# ==============================================================================
# RoPE-Frequency-Aware KV Page Routing — Experiment v2
#
# This extends the original notebook to cover everything progress.md flagged
# as "unproven / not yet run" against the README:
#
#   [4.5]  Mandatory outlier-sensitivity instrumentation      -> Section 6
#   [4.4]  Cost metrics (selector latency, metadata bytes,
#          KV bytes loaded, bandwidth-normalized recall)      -> Section 4 + 7
#   [4.4]  Downstream task quality (perplexity impact)        -> Section 8
#   [4.2]  Full fraction sweep incl. 0% floor and 15%         -> CONFIG
#   [scale] Variance across independent context windows       -> Section 3/9
#   [near]  Literal-spec near/far labeling as a cross-check   -> Section 5b
#
# Everything from the original script (Tier-1/Tier-2 routing, Q4 quadrant
# test, label-circularity fix, sensitivity sweep over labeling params) is
# preserved and generalized to run per-window, then aggregated with mean/std.
#
# NOTE: This is meant to run in Colab/Jupyter with a GPU. It has NOT been
# executed — review before running. The perplexity-impact section (8) patches
# HF attention internals and includes a self-check (assert_reconstruction_
# fidelity) that will warn loudly if your transformers version's attention
# forward doesn't match the assumptions here (GQA repeat_interleave, RoPE
# half-split convention, eager softmax attention). If that check fails, trust
# everything EXCEPT Section 8 and adapt the patch to your version.
# ==============================================================================

import os
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ["HF_HUB_DISABLE_XET"] = "1"
# os.environ["HF_TOKEN"] = "hf_..."   # uncomment + fill in if you hit rate limits

# !pip install torch transformers datasets accelerate
import re
import sys
import copy
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import Dict, Tuple, List, Set, Optional
import gc
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQ_LEN = 2048
PAGE_SIZE = 64
W_PAGES = 2
TOP_K_PAGES = 4
EVAL_QUERIES = 256

LAYERS_TO_HOOK = [4, 10, 16]

# >>> CHANGED: now matches README Section 4.2 exactly (0% floor + 15% added).
FRACTION_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0]

# Default (primary-report) labeling config.
DEFAULT_CONTEXT_TOKENS = 64
DEFAULT_PERCENTILE = 50

# Sensitivity sweep grid for the Q4-labeling parameters (unchanged from v1).
SENSITIVITY_CONTEXT_TOKENS = [32, 64, 128]
SENSITIVITY_PERCENTILES = [25, 50, 75]

# >>> NEW: independent, non-overlapping context windows pulled from a larger
# corpus so headline numbers get a mean +/- std instead of a single point
# estimate (progress.md gap #4: "no variance/confidence interval").
NUM_EVAL_WINDOWS = 3
CORPUS_NAME = "Salesforce/wikitext"
CORPUS_CONFIG = "wikitext-103-raw-v1"   # bigger than wikitext-2 so we can carve out multiple non-overlapping 2048-tok windows
CORPUS_SPLIT = "test"

# >>> NEW: Section 4.5 outlier-sensitivity instrumentation.
OUTLIER_CLIP_PERCENTILE = 99.0  # "top-1%-magnitude-clipped" per README 4.5

# >>> NEW: Section 4.4 cost metrics.
DTYPE_BYTES = 2  # fp16
SELECTOR_LATENCY_REPEATS = 3  # re-time each query's routing step this many times and take the min (reduces timing noise)
NOMINAL_HBM_BANDWIDTH_GB_S = 3350.0  # e.g. A100 80GB SXM nameplate; used ONLY for a labeled, coarse bandwidth-utilization estimate, not a benchmark

# >>> NEW: Section 8 downstream perplexity impact.
RUN_PERPLEXITY_IMPACT = True
PPL_METHODS = ['A', 'C', 'B_best']  # 'B_best' resolved per-window from the Q4 sweep

# >>> NEW: Section 5b alternate near/far labeling, matching the README's
# literal quadrant-table wording ("near = within/just outside Tier 1") rather
# than the median-split-of-the-routable-pool proxy used in v1. Both are
# reported; progress.md flagged this as a stated deviation worth checking.
STRICT_NEAR_PAGE_MARGIN = 2  # pages beyond Tier-1's boundary still counted as "near"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}", flush=True)


def load_with_retry(load_fn, name, max_retries=3, backoff=5):
    for attempt in range(1, max_retries + 1):
        try:
            return load_fn()
        except Exception as e:
            print(f"[{name}] attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}", flush=True)
            if attempt == max_retries:
                raise
            print(f"Retrying in {backoff}s...", flush=True)
            time.sleep(backoff)


# ==============================================================================
# 1. LOAD MODEL & CORPUS (kept loaded until every window + PPL pass is done)
# ==============================================================================
print(f"Loading {MODEL_NAME}...", flush=True)
tokenizer = load_with_retry(lambda: AutoTokenizer.from_pretrained(MODEL_NAME), "tokenizer")
model = load_with_retry(
    lambda: AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="auto"),
    "model"
)
model.eval()

print(f"Loading corpus {CORPUS_NAME}/{CORPUS_CONFIG}...", flush=True)
dataset = load_with_retry(
    lambda: load_dataset(CORPUS_NAME, CORPUS_CONFIG, split=CORPUS_SPLIT),
    "dataset"
)
full_text = "\n".join(dataset["text"])
full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
print(f"Corpus has {full_ids.shape[0]} tokens available.", flush=True)

needed = NUM_EVAL_WINDOWS * MAX_SEQ_LEN
if full_ids.shape[0] < needed:
    raise ValueError(
        f"Corpus too short for {NUM_EVAL_WINDOWS} non-overlapping windows of "
        f"{MAX_SEQ_LEN} tokens ({full_ids.shape[0]} < {needed}). "
        f"Lower NUM_EVAL_WINDOWS or switch CORPUS_CONFIG."
    )

# Non-overlapping windows, spread across the corpus rather than all from the
# very start, so they're not adjacent/correlated text.
window_starts = np.linspace(0, full_ids.shape[0] - MAX_SEQ_LEN, NUM_EVAL_WINDOWS, dtype=int)
eval_windows = [full_ids[s:s + MAX_SEQ_LEN].unsqueeze(0) for s in window_starts]
print(f"Prepared {NUM_EVAL_WINDOWS} eval windows at token offsets: {list(window_starts)}", flush=True)


# ==============================================================================
# 2. HOOKS — capture q, pre-RoPE k, post-RoPE k, V, and exact attention
#    (V is >>> NEW, needed for Section 8's real attention-output reconstruction)
# ==============================================================================
def make_capture_hook(store: dict):
    def llama_attention_hook(module, args, kwargs, output):
        hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']
        bsz, q_len, _ = hidden_states.size()

        cfg = module.config
        num_heads = getattr(module, 'num_heads', None) or cfg.num_attention_heads
        num_kv_heads = getattr(module, 'num_key_value_heads', None) or getattr(cfg, 'num_key_value_heads', num_heads)
        head_dim = getattr(module, 'head_dim', None) or (cfg.hidden_size // cfg.num_attention_heads)
        n_rep = num_heads // num_kv_heads

        query_states = module.q_proj(hidden_states)
        key_states = module.k_proj(hidden_states)
        value_states = module.v_proj(hidden_states)  # >>> NEW

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)  # >>> NEW
        if n_rep > 1:
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)  # >>> NEW

        store['pre_rope_k'] = key_states.detach().clone()

        if kwargs.get('position_embeddings', None) is not None:
            cos, sin = kwargs['position_embeddings']
            cos, sin = cos[:, :q_len], sin[:, :q_len]
        else:
            position_ids = kwargs.get('position_ids', None)
            if position_ids is None:
                position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
            cos, sin = module.rotary_emb(key_states, position_ids[:, :q_len])

        def apply_rotary(x, cos, sin):
            cos_b = cos.unsqueeze(1)
            sin_b = sin.unsqueeze(1)
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            rotated = torch.cat((-x2, x1), dim=-1)
            return (x * cos_b) + (rotated * sin_b)

        query_states = apply_rotary(query_states, cos, sin)
        post_rope_k = apply_rotary(key_states, cos, sin)

        store['q'] = query_states.detach().clone()
        store['post_rope_k'] = post_rope_k.detach().clone()
        store['v'] = value_states.detach().clone()  # >>> NEW

        attn_weights = torch.matmul(query_states, post_rope_k.transpose(2, 3)) / np.sqrt(head_dim)
        mask = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool, device=device), diagonal=1)
        attn_weights.masked_fill_(mask, float('-inf'))
        attn_probs = F.softmax(attn_weights, dim=-1)
        store['exact_attn'] = attn_probs.detach().clone()

        # >>> NEW: keep baseline logits/loss so Section 8 doesn't need an
        # extra dense forward pass to get the perplexity baseline.
        store['head_dim'] = head_dim
        store['num_heads'] = num_heads
    return llama_attention_hook


# ==============================================================================
# 3. ROUTING ENGINE
# ==============================================================================
def get_low_freq_indices(dim: int, fraction: float) -> torch.Tensor:
    num_dims_to_keep = max(1, int((dim // 2) * fraction))
    first_half_indices = torch.arange((dim // 2) - num_dims_to_keep, dim // 2)
    second_half_indices = torch.arange(dim - num_dims_to_keep, dim)
    return torch.cat([first_half_indices, second_half_indices]).to(device)


def compute_page_bounds(k_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_len = (k_tensor.shape[1] // PAGE_SIZE) * PAGE_SIZE
    k_trunc = k_tensor[:, :valid_len, :]
    k_pages = k_trunc.view(k_trunc.shape[0], -1, PAGE_SIZE, k_trunc.shape[2])
    return k_pages.min(dim=2)[0], k_pages.max(dim=2)[0]


def score_pages_quest_style(q_slice, page_mins, page_maxs) -> torch.Tensor:
    q_pos = F.relu(q_slice).unsqueeze(1)
    q_neg = -F.relu(-q_slice).unsqueeze(1)
    return (q_pos * page_maxs).sum(dim=-1) + (q_neg * page_mins).sum(dim=-1)


# ==============================================================================
# 4. PHASE 1 — ROUTING DECISIONS + COST INSTRUMENTATION [README 4.4]
#    Selector latency and metadata bytes are measured HERE, isolated to just
#    the scoring+topk step, since that's the actual per-decode-step cost a
#    deployed router would pay (not the ground-truth bookkeeping, which only
#    exists for evaluation).
# ==============================================================================
def compute_routing_decisions(q, k_pre, k_post, v, exact_attn, seq_len_local):
    num_heads, _, head_dim = q.shape
    num_pages = seq_len_local // PAGE_SIZE
    page_centers = torch.arange(num_pages, device=device) * PAGE_SIZE + (PAGE_SIZE // 2)
    valid_len = (k_pre.shape[1] // PAGE_SIZE) * PAGE_SIZE
    k_pre_pages = k_pre[:, :valid_len, :].view(num_heads, num_pages, PAGE_SIZE, head_dim)
    page_means_pre = k_pre_pages.mean(dim=2)

    query_indices = list(range(seq_len_local - EVAL_QUERIES, seq_len_local))
    bounds_A = compute_page_bounds(k_post)

    decisions = {}
    for method in ['A', 'C'] + [f'B_{frac}' for frac in FRACTION_SWEEP]:
        per_query = []
        bound_widths = []
        selector_latencies_ms = []

        # >>> NEW: static metadata-bytes-per-page for this method (doesn't
        # depend on the query, so compute once).
        if method == 'A':
            dims_used, factors_stored = head_dim, 2  # min + max
        elif method == 'C':
            dims_used, factors_stored = head_dim, 1  # mean only
        else:
            frac = float(method.split('_')[1])
            dims_used = max(1, int((head_dim // 2) * frac)) * 2  # low-freq subset spans both halves
            factors_stored = 2  # min + max

        for q_idx in query_indices:
            q_head_vecs = q[:, q_idx, :]
            current_page_idx = q_idx // PAGE_SIZE
            tier1_start_page = max(0, current_page_idx - W_PAGES)
            tier1_pages = list(range(tier1_start_page, current_page_idx + 1))
            routable_pages_idx = torch.arange(0, tier1_start_page, device=device)
            if len(routable_pages_idx) <= TOP_K_PAGES:
                continue

            # ---- timed routing step (this is the only part a real system pays for at decode time) ----
            best_dt = None
            for _ in range(SELECTOR_LATENCY_REPEATS):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                if method == 'A':
                    scores = score_pages_quest_style(
                        q_head_vecs, bounds_A[0][:, routable_pages_idx, :], bounds_A[1][:, routable_pages_idx, :])
                elif method == 'C':
                    q_norm = F.normalize(q_head_vecs.unsqueeze(1), dim=-1)
                    p_norm = F.normalize(page_means_pre[:, routable_pages_idx, :], dim=-1)
                    scores = (q_norm * p_norm).sum(dim=-1)
                else:
                    fraction = float(method.split('_')[1])
                    if fraction == 0.0:
                        # >>> NEW: true "no signal" sanity floor per README 4.2
                        # ("At 0%, B degenerates to no routing signal"). A
                        # fixed-seed random score, NOT 1 low-freq dim, is the
                        # honest reading of "no signal" -- with the old
                        # max(1,...) floor, 0% still kept 1 real dimension.
                        g = torch.Generator(device='cpu').manual_seed(hash((q_idx, method)) % (2**31))
                        scores = torch.rand(num_heads, len(routable_pages_idx), generator=g).to(device)
                    else:
                        idx = get_low_freq_indices(head_dim, fraction)
                        q_sub = q_head_vecs[:, idx]
                        mins_sub = bounds_A[0][:, routable_pages_idx][:, :, idx]
                        maxs_sub = bounds_A[1][:, routable_pages_idx][:, :, idx]
                        scores = score_pages_quest_style(q_sub, mins_sub, maxs_sub)

                topk_indices = scores.topk(TOP_K_PAGES, dim=-1).indices
                selected_pages = routable_pages_idx[topk_indices]

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000.0
                best_dt = dt if best_dt is None else min(best_dt, dt)
            selector_latencies_ms.append(best_dt)

            if method.startswith('B_') and float(method.split('_')[1]) > 0.0:
                fraction = float(method.split('_')[1])
                idx = get_low_freq_indices(head_dim, fraction)
                mins_sub = bounds_A[0][:, routable_pages_idx][:, :, idx]
                maxs_sub = bounds_A[1][:, routable_pages_idx][:, :, idx]
                bound_widths.append((maxs_sub - mins_sub).mean().item())

            dists_all = (q_idx - page_centers[routable_pages_idx].float()).abs()
            dist_median = dists_all.median().item()
            is_far_per_page = dists_all > dist_median

            # >>> NEW: alternate literal-spec near/far flag, kept alongside
            # the median-split one (README 5b / progress.md gap #6).
            dist_in_pages = (current_page_idx - routable_pages_idx).float()
            is_far_per_page_strict = dist_in_pages > (W_PAGES + STRICT_NEAR_PAGE_MARGIN)

            gt_attn = exact_attn[:, q_idx, :]
            gt_page_mass = torch.zeros(num_heads, len(routable_pages_idx), device=device)
            for i, p in enumerate(routable_pages_idx.tolist()):
                start_tok, end_tok = p * PAGE_SIZE, min((p + 1) * PAGE_SIZE, q_idx + 1)
                gt_page_mass[:, i] = gt_attn[:, start_tok:end_tok].sum(dim=-1)

            tier1_mass = 0.0
            for p in tier1_pages:
                start_tok, end_tok = p * PAGE_SIZE, min((p + 1) * PAGE_SIZE, q_idx + 1)
                tier1_mass += gt_attn[:, start_tok:end_tok].sum().item()

            selected_mass = 0.0
            for h in range(num_heads):
                for p in selected_pages[h].tolist():
                    start_tok, end_tok = p * PAGE_SIZE, min((p + 1) * PAGE_SIZE, q_idx + 1)
                    selected_mass += gt_attn[h, start_tok:end_tok].sum().item()

            per_query.append({
                'q_idx': q_idx,
                'routable_pages_idx': routable_pages_idx,
                'selected_pages': selected_pages,
                'gt_page_mass': gt_page_mass,
                'is_far_per_page': is_far_per_page,
                'is_far_per_page_strict': is_far_per_page_strict,  # >>> NEW
                'recall_mass_sum': selected_mass + tier1_mass,
                'recall_evals': num_heads,
                'n_routable': len(routable_pages_idx),
            })

        # >>> NEW: cost summary for this method (README 4.4 secondary metrics)
        avg_routable = np.mean([r['n_routable'] for r in per_query]) if per_query else 0.0
        metadata_bytes_per_step = avg_routable * dims_used * factors_stored * num_heads * DTYPE_BYTES
        kv_bytes_per_step = TOP_K_PAGES * PAGE_SIZE * head_dim * num_heads * 2 * DTYPE_BYTES  # K+V, same across methods by construction

        decisions[method] = {
            'per_query': per_query,
            'bound_widths': bound_widths,
            'selector_latency_ms_mean': float(np.mean(selector_latencies_ms)) if selector_latencies_ms else 0.0,
            'selector_latency_ms_p50': float(np.percentile(selector_latencies_ms, 50)) if selector_latencies_ms else 0.0,
            'metadata_bytes_per_step': metadata_bytes_per_step,
            'kv_bytes_per_step': kv_bytes_per_step,
        }
    return decisions


# ==============================================================================
# 5. PHASE 2 — Q4 LABELING & AGGREGATION (per-window; cheap relabeling)
# ==============================================================================
def score_with_labeling(decisions, get_lexical_sim, threshold, use_strict_near=False):
    results = {}
    for method, d in decisions.items():
        total_recall, num_recall_evals = 0.0, 0
        q4_samples, q4_eligible = [], 0
        q_stats = {'Q1': [], 'Q2': [], 'Q3': []}

        for rec in d['per_query']:
            q_idx = rec['q_idx']
            routable = rec['routable_pages_idx'].tolist()
            is_far = rec['is_far_per_page_strict'] if use_strict_near else rec['is_far_per_page']
            gt_page_mass = rec['gt_page_mass']
            selected_pages = rec['selected_pages']
            num_heads = gt_page_mass.shape[0]

            total_recall += rec['recall_mass_sum']
            num_recall_evals += rec['recall_evals']

            lex_sims = torch.tensor(
                [get_lexical_sim(q_idx, p) for p in routable], device=device)
            is_high_sim = lex_sims > threshold

            for h in range(num_heads):
                page_mass_h = gt_page_mass[h]
                mass_median = page_mass_h.median()
                is_important = page_mass_h > mass_median
                true_q4_mask = is_important & is_far & (~is_high_sim)
                true_q4_target_mass = page_mass_h[true_q4_mask].sum().item()

                selected_p = selected_pages[h].tolist()
                if true_q4_target_mass > 0:
                    q4_eligible += 1
                    recovered = sum(
                        page_mass_h[i].item()
                        for i, p in enumerate(routable)
                        if p in selected_p and true_q4_mask[i]
                    )
                    q4_samples.append(recovered / true_q4_target_mass)

                for p in selected_p:
                    if p not in routable:
                        continue
                    i = routable.index(p)
                    far = is_far[i].item()
                    high_sim = is_high_sim[i].item()
                    pm = page_mass_h[i].item()
                    if not far and high_sim: q_stats['Q1'].append(pm)
                    elif not far and not high_sim: q_stats['Q2'].append(pm)
                    elif far and high_sim: q_stats['Q3'].append(pm)

        results[method] = {
            'recall': (total_recall / num_recall_evals) * 100 if num_recall_evals else 0.0,
            'Q4_target_recovery': np.mean(q4_samples) * 100 if q4_samples else 0.0,
            'Q4_eligible_count': q4_eligible,
            'Q1_mass': np.mean(q_stats['Q1']) if q_stats['Q1'] else 0.0,
            'Q2_mass': np.mean(q_stats['Q2']) if q_stats['Q2'] else 0.0,
            'Q3_mass': np.mean(q_stats['Q3']) if q_stats['Q3'] else 0.0,
            'avg_bound_width': np.mean(d['bound_widths']) if d['bound_widths'] else None,
            'selector_latency_ms_mean': d['selector_latency_ms_mean'],
            'metadata_bytes_per_step': d['metadata_bytes_per_step'],
            'kv_bytes_per_step': d['kv_bytes_per_step'],
        }
    return results


# ==============================================================================
# 6. >>> NEW — SECTION 4.5: MANDATORY OUTLIER-SENSITIVITY INSTRUMENTATION
#    For every B fraction: compute per-page bound width using true min/max
#    vs. top-1%-magnitude-clipped min/max, and track the divergence. Purely
#    diagnostic (per README: "recorded as a diagnostic time series, not
#    acted upon within this experiment").
# ==============================================================================
def compute_outlier_diagnostics(k_post, head_dim):
    """Returns a dict: fraction -> {'true_width', 'clipped_width', 'divergence_pct'}"""
    lo_q = (100.0 - OUTLIER_CLIP_PERCENTILE) / 100.0
    hi_q = OUTLIER_CLIP_PERCENTILE / 100.0

    out = {}
    for frac in FRACTION_SWEEP:
        if frac == 0.0:
            continue  # no index at 0% (random-selection floor); nothing to diagnose
        idx = get_low_freq_indices(head_dim, frac)
        k_sub = k_post[:, :, idx]  # [num_heads, seq_len, sub_dim]

        valid_len = (k_sub.shape[1] // PAGE_SIZE) * PAGE_SIZE
        k_trunc = k_sub[:, :valid_len, :]
        k_pages = k_trunc.view(k_trunc.shape[0], -1, PAGE_SIZE, k_trunc.shape[2])  # [H, P, PAGE, D]

        true_min, true_max = k_pages.min(dim=2)[0], k_pages.max(dim=2)[0]
        true_width = (true_max - true_min).mean().item()

        lo = torch.quantile(k_pages.float(), lo_q, dim=2, keepdim=True)
        hi = torch.quantile(k_pages.float(), hi_q, dim=2, keepdim=True)
        k_clipped = k_pages.float().clamp(min=lo, max=hi)
        clipped_min, clipped_max = k_clipped.min(dim=2)[0], k_clipped.max(dim=2)[0]
        clipped_width = (clipped_max - clipped_min).mean().item()

        divergence_pct = 100.0 * (true_width - clipped_width) / true_width if true_width > 0 else 0.0
        out[frac] = {
            'true_width': true_width,
            'clipped_width': clipped_width,
            'divergence_pct': divergence_pct,
        }
    return out


# ==============================================================================
# 7. >>> NEW — SECTION 8: DOWNSTREAM PERPLEXITY IMPACT
#    Patches the hooked layers' attention to use ONLY {Tier-1 window} U
#    {top-K selected pages for a given method}, for eval-window query
#    positions, in a genuine second forward pass, then compares next-token
#    loss on those positions against the true dense baseline (computed once,
#    from the Section 2 capture pass -- no extra forward needed for that).
#
#    Self-check: with an all-allowed (fully dense) mask, this reconstruction
#    must reproduce the model's real output almost exactly. That validates
#    the manual re-implementation of attention (GQA repeat_interleave, RoPE
#    convention, o_proj) before trusting the masked/sparse variant.
# ==============================================================================
def build_sparse_additive_mask(q_len, num_heads, eval_query_indices, selected_pages_by_qidx,
                                w_pages, page_size, dev):
    causal = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool, device=dev), diagonal=1)
    mask = torch.zeros(num_heads, q_len, q_len, device=dev)
    mask.masked_fill_(causal.unsqueeze(0), float('-inf'))

    eval_set = set(eval_query_indices)
    for q_idx in eval_query_indices:
        current_page = q_idx // page_size
        tier1_start_page = max(0, current_page - w_pages)
        tier1_start_tok = tier1_start_page * page_size

        row = torch.full((num_heads, q_len), float('-inf'), device=dev)
        row[:, tier1_start_tok:q_idx + 1] = 0.0  # Tier-1 window always allowed

        sel = selected_pages_by_qidx.get(q_idx)
        if sel is not None:
            for h in range(num_heads):
                for p in sel[h].tolist():
                    s, e = p * page_size, min((p + 1) * page_size, q_idx + 1)
                    row[h, s:e] = 0.0
        mask[:, q_idx, :] = row
    return mask.unsqueeze(0)  # [1, num_heads, q_len, q_len] -- broadcasts over batch


def make_ppl_patch_hook(selected_pages_by_qidx, eval_query_indices, w_pages, page_size, dev,
                         validate_capture: Optional[dict] = None):
    def hook(module, args, kwargs, output):
        hidden_states = args[0] if len(args) > 0 else kwargs['hidden_states']
        bsz, q_len, _ = hidden_states.size()
        cfg = module.config
        num_heads = getattr(module, 'num_heads', None) or cfg.num_attention_heads
        num_kv_heads = getattr(module, 'num_key_value_heads', None) or getattr(cfg, 'num_key_value_heads', num_heads)
        head_dim = getattr(module, 'head_dim', None) or (cfg.hidden_size // cfg.num_attention_heads)
        n_rep = num_heads // num_kv_heads

        q_states = module.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        k_states = module.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        v_states = module.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        if n_rep > 1:
            k_states = k_states.repeat_interleave(n_rep, dim=1)
            v_states = v_states.repeat_interleave(n_rep, dim=1)

        if kwargs.get('position_embeddings', None) is not None:
            cos, sin = kwargs['position_embeddings']
            cos, sin = cos[:, :q_len], sin[:, :q_len]
        else:
            position_ids = kwargs.get('position_ids', None)
            if position_ids is None:
                position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
            cos, sin = module.rotary_emb(k_states, position_ids[:, :q_len])

        def apply_rotary(x, cos, sin):
            cos_b, sin_b = cos.unsqueeze(1), sin.unsqueeze(1)
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            rotated = torch.cat((-x2, x1), dim=-1)
            return x * cos_b + rotated * sin_b

        q_states = apply_rotary(q_states, cos, sin)
        k_states = apply_rotary(k_states, cos, sin)

        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / np.sqrt(head_dim)
        add_mask = build_sparse_additive_mask(q_len, num_heads, eval_query_indices,
                                               selected_pages_by_qidx, w_pages, page_size, dev)
        masked_weights = attn_weights + add_mask
        attn_probs = F.softmax(masked_weights, dim=-1)
        context = torch.matmul(attn_probs, v_states)
        context = context.transpose(1, 2).contiguous().view(bsz, q_len, num_heads * head_dim)
        attn_output = module.o_proj(context)

        if validate_capture is not None and validate_capture.get('pending', False):
            # Fully-dense reconstruction (same math, causal-only mask) must
            # match the model's real output. This runs once to validate the
            # manual re-implementation before the masked result is trusted.
            dense_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / np.sqrt(head_dim)
            causal = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool, device=dev), diagonal=1)
            dense_weights = dense_weights.masked_fill(causal, float('-inf'))
            dense_probs = F.softmax(dense_weights, dim=-1)
            dense_context = torch.matmul(dense_probs, v_states).transpose(1, 2).contiguous().view(
                bsz, q_len, num_heads * head_dim)
            dense_output = module.o_proj(dense_context)
            orig = output[0] if isinstance(output, (tuple, list)) else output
            validate_capture['max_diff'] = (dense_output - orig).abs().max().item()
            validate_capture['pending'] = False

        if isinstance(output, tuple):
            return (attn_output,) + tuple(output[1:])
        elif isinstance(output, list):
            return [attn_output] + list(output[1:])
        else:
            return attn_output
    return hook


def run_perplexity_pass(inputs, method_selected_by_layer: Dict[int, Dict[int, torch.Tensor]],
                         eval_query_indices, tag: str, validate=False):
    """One extra forward pass with attention patched at LAYERS_TO_HOOK.
    method_selected_by_layer[layer_idx] = {q_idx: selected_pages_tensor[num_heads, TOP_K]}
    """
    validate_capture = {'pending': True} if validate else None
    handles = []
    for layer_idx in LAYERS_TO_HOOK:
        h = model.model.layers[layer_idx].self_attn.register_forward_hook(
            make_ppl_patch_hook(method_selected_by_layer[layer_idx], eval_query_indices,
                                 W_PAGES, PAGE_SIZE, device, validate_capture),
            with_kwargs=True
        )
        handles.append(h)

    with torch.no_grad():
        out = model(**inputs)
    for h in handles:
        h.remove()

    if validate_capture is not None:
        max_diff = validate_capture.get('max_diff', float('nan'))
        ok = max_diff < 1e-2  # fp16 tolerance
        print(f"  [validate:{tag}] dense-reconstruction max|diff| vs real model output = {max_diff:.5f} "
              f"-> {'OK' if ok else 'WARNING: reconstruction mismatch, treat PPL numbers with caution'}")

    return out.logits


def token_nll(logits, labels, positions):
    """Per-position next-token negative log-likelihood, restricted to `positions`."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    nll_all = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        reduction='none'
    ).view(shift_labels.shape)
    # position p in original sequence predicts token p+1, i.e. lives at shift index p
    idx = torch.tensor([p for p in positions if p < nll_all.shape[1]], device=nll_all.device)
    return nll_all[0, idx]


# ==============================================================================
# 8. MAIN LOOP — run everything per window, aggregate at the end
# ==============================================================================
window_results = []  # list of dicts: one per window, holding everything needed for cross-window aggregation

for w_i, inputs_ids in enumerate(eval_windows):
    print("\n" + "#" * 70)
    print(f"# WINDOW {w_i + 1}/{NUM_EVAL_WINDOWS}  (corpus offset {window_starts[w_i]})")
    print("#" * 70)

    inputs = {"input_ids": inputs_ids.to(device), "attention_mask": torch.ones_like(inputs_ids).to(device)}
    seq_len = inputs["input_ids"].shape[1]
    input_ids_cpu = inputs["input_ids"][0].detach().cpu()

    # --- capture pass (Section 2) ---
    per_layer_tensors = {}
    handles = []
    for layer_idx in LAYERS_TO_HOOK:
        store = {}
        per_layer_tensors[layer_idx] = store
        h = model.model.layers[layer_idx].self_attn.register_forward_hook(make_capture_hook(store), with_kwargs=True)
        handles.append(h)

    print(f"Running forward pass on {seq_len} tokens to capture exact state...", flush=True)
    with torch.no_grad():
        baseline_out = model(**inputs)
    for h in handles:
        h.remove()

    baseline_logits = baseline_out.logits.detach()

    # --- lexical (model-free) content-similarity signal (Section 2b, unchanged logic) ---
    _word_re = re.compile(r"\w+")

    def _word_set(token_ids: torch.Tensor) -> Set[str]:
        txt = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        return set(w.lower() for w in _word_re.findall(txt))

    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 0.0
        union = len(a | b)
        return len(a & b) / union if union else 0.0

    num_pages_total = seq_len // PAGE_SIZE
    page_word_sets: List[Set[str]] = [
        _word_set(input_ids_cpu[p * PAGE_SIZE:(p + 1) * PAGE_SIZE]) for p in range(num_pages_total)
    ]
    query_indices_global = list(range(seq_len - EVAL_QUERIES, seq_len))

    query_ctx_caches: Dict[int, Dict[int, Set[str]]] = {}
    lexical_sim_caches: Dict[int, Dict[Tuple[int, int], float]] = {}
    all_sims_cache: Dict[int, List[float]] = {}

    def build_context_signal(context_tokens: int):
        if context_tokens in query_ctx_caches:
            return
        ctx_cache: Dict[int, Set[str]] = {}
        for q_idx in query_indices_global:
            start = max(0, q_idx - context_tokens + 1)
            ctx_cache[q_idx] = _word_set(input_ids_cpu[start:q_idx + 1])
        query_ctx_caches[context_tokens] = ctx_cache

        sim_cache: Dict[Tuple[int, int], float] = {}
        all_sims: List[float] = []
        for q_idx in query_indices_global:
            current_page_idx = q_idx // PAGE_SIZE
            tier1_start_page = max(0, current_page_idx - W_PAGES)
            for p in range(tier1_start_page):
                v = jaccard(ctx_cache[q_idx], page_word_sets[p])
                sim_cache[(q_idx, p)] = v
                all_sims.append(v)
        lexical_sim_caches[context_tokens] = sim_cache
        all_sims_cache[context_tokens] = all_sims

    def get_lexical_sim(context_tokens, q_idx, page_idx) -> float:
        return lexical_sim_caches[context_tokens][(q_idx, page_idx)]

    def get_threshold(context_tokens, percentile) -> float:
        return float(np.percentile(all_sims_cache[context_tokens], percentile))

    for ct in sorted(set(SENSITIVITY_CONTEXT_TOKENS + [DEFAULT_CONTEXT_TOKENS])):
        build_context_signal(ct)

    default_threshold = get_threshold(DEFAULT_CONTEXT_TOKENS, DEFAULT_PERCENTILE)

    # --- routing decisions + cost metrics (Section 4) ---
    layer_decisions = {}
    layer_primary_results = {}
    layer_outlier_diag = {}

    for layer_idx in LAYERS_TO_HOOK:
        t0 = time.time()
        store = per_layer_tensors[layer_idx]
        q, k_pre, k_post, v, exact_attn = (store['q'][0], store['pre_rope_k'][0],
                                            store['post_rope_k'][0], store['v'][0], store['exact_attn'][0])
        head_dim = store['head_dim']

        decisions = compute_routing_decisions(q, k_pre, k_post, v, exact_attn, seq_len)
        layer_decisions[layer_idx] = decisions

        primary = score_with_labeling(
            decisions,
            lambda qi, p, ct=DEFAULT_CONTEXT_TOKENS: get_lexical_sim(ct, qi, p),
            default_threshold,
        )
        layer_primary_results[layer_idx] = primary

        # Section 6: outlier diagnostics (uses full post-RoPE K, independent of routing decisions)
        layer_outlier_diag[layer_idx] = compute_outlier_diagnostics(k_post, head_dim)

        print(f"\n--- Layer {layer_idx} (window {w_i + 1}) ---")
        print(f"{'Method':<10} | {'Recall':<8} | {'Q4 Recov':<9} | {'Sel.Lat(ms)':<12} | "
              f"{'MetaB/step':<11} | {'KVB/step':<10} | {'BoundW':<8} | {'Outlier div%'}")
        r = primary['C']
        print(f"{'C(PreRoPE)':<10} | {r['recall']:6.2f}% | {r['Q4_target_recovery']:7.2f}% | "
              f"{r['selector_latency_ms_mean']:10.4f} | {r['metadata_bytes_per_step']:9.0f} | "
              f"{r['kv_bytes_per_step']:8.0f} | {'N/A':<8} | N/A")
        for frac in FRACTION_SWEEP:
            r = primary[f'B_{frac}']
            bw = f"{r['avg_bound_width']:.3f}" if r['avg_bound_width'] is not None else "N/A"
            odiv = f"{layer_outlier_diag[layer_idx][frac]['divergence_pct']:.2f}%" if frac in layer_outlier_diag[layer_idx] else "N/A"
            print(f"{'B_' + str(int(frac*100)) + '%':<10} | {r['recall']:6.2f}% | {r['Q4_target_recovery']:7.2f}% | "
                  f"{r['selector_latency_ms_mean']:10.4f} | {r['metadata_bytes_per_step']:9.0f} | "
                  f"{r['kv_bytes_per_step']:8.0f} | {bw:<8} | {odiv}")
        r = primary['A']
        print(f"{'A(Full)':<10} | {r['recall']:6.2f}% | {r['Q4_target_recovery']:7.2f}% | "
              f"{r['selector_latency_ms_mean']:10.4f} | {r['metadata_bytes_per_step']:9.0f} | "
              f"{r['kv_bytes_per_step']:8.0f} | {'N/A':<8} | N/A")
        print(f"[layer {layer_idx} done in {time.time()-t0:.1f}s]")

    # --- strict-near-labeling cross-check (Section 5b) ---
    layer_strict_results = {}
    for layer_idx in LAYERS_TO_HOOK:
        layer_strict_results[layer_idx] = score_with_labeling(
            layer_decisions[layer_idx],
            lambda qi, p, ct=DEFAULT_CONTEXT_TOKENS: get_lexical_sim(ct, qi, p),
            default_threshold,
            use_strict_near=True,
        )

    # --- sensitivity sweep (Section 5, unchanged logic) ---
    sensitivity_summary = []
    for layer_idx in LAYERS_TO_HOOK:
        decisions = layer_decisions[layer_idx]
        for ct in SENSITIVITY_CONTEXT_TOKENS:
            for pct in SENSITIVITY_PERCENTILES:
                thr = get_threshold(ct, pct)
                results = score_with_labeling(decisions, lambda qi, p, ct=ct: get_lexical_sim(ct, qi, p), thr)
                best_frac = sorted(FRACTION_SWEEP, key=lambda f: results[f'B_{f}']['Q4_target_recovery'], reverse=True)[0]
                q4_C = results['C']['Q4_target_recovery']
                q4_B = results[f'B_{best_frac}']['Q4_target_recovery']
                c_elig = results['C']['Q4_eligible_count']
                verdict = "INCONCLUSIVE" if c_elig == 0 else ("B WINS" if q4_B > q4_C else "C WINS/TIE")
                sensitivity_summary.append((layer_idx, ct, pct, q4_C, q4_B, best_frac, c_elig, verdict))

    # --- global best B fraction for this window (for the perplexity pass) ---
    frac_scores = {f: [] for f in FRACTION_SWEEP}
    for layer_idx in LAYERS_TO_HOOK:
        for f in FRACTION_SWEEP:
            frac_scores[f].append(layer_primary_results[layer_idx][f'B_{f}']['Q4_target_recovery'])
    global_best_frac = max(frac_scores, key=lambda f: np.mean(frac_scores[f]))
    print(f"\nWindow {w_i+1}: global-best B fraction across layers = {global_best_frac*100:.0f}%")

    # --- Section 8: perplexity impact ---
    ppl_results = {}
    if RUN_PERPLEXITY_IMPACT:
        print(f"\nRunning perplexity-impact forward passes for methods: {PPL_METHODS} ...")
        baseline_nll = token_nll(baseline_logits, inputs["input_ids"], query_indices_global)
        baseline_ppl = torch.exp(baseline_nll.mean()).item()
        print(f"  Baseline (dense) windowed PPL over eval positions: {baseline_ppl:.3f}")

        for method_tag in PPL_METHODS:
            method_key = f'B_{global_best_frac}' if method_tag == 'B_best' else method_tag
            method_selected_by_layer = {}
            for layer_idx in LAYERS_TO_HOOK:
                recs = layer_decisions[layer_idx][method_key]['per_query']
                method_selected_by_layer[layer_idx] = {r['q_idx']: r['selected_pages'] for r in recs}

            logits = run_perplexity_pass(inputs, method_selected_by_layer, query_indices_global,
                                          tag=f"{method_tag}@w{w_i+1}",
                                          validate=(w_i == 0 and method_tag == PPL_METHODS[0]))
            nll = token_nll(logits, inputs["input_ids"], query_indices_global)
            ppl = torch.exp(nll.mean()).item()
            delta = ppl - baseline_ppl
            print(f"  {method_tag:<8} (routed, Tier1+top{TOP_K_PAGES}) windowed PPL: {ppl:.3f}  "
                  f"(delta vs dense baseline: {delta:+.3f})")
            ppl_results[method_tag] = {'ppl': ppl, 'delta_vs_dense': delta}

        ppl_results['dense_baseline'] = {'ppl': baseline_ppl, 'delta_vs_dense': 0.0}

    window_results.append({
        'window_idx': w_i,
        'primary': layer_primary_results,
        'strict_near': layer_strict_results,
        'sensitivity': sensitivity_summary,
        'outlier': layer_outlier_diag,
        'ppl': ppl_results,
        'global_best_frac': global_best_frac,
    })

    # free this window's large tensors before moving on
    del per_layer_tensors, baseline_out, baseline_logits
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

del model
gc.collect()
if device.type == 'cuda':
    torch.cuda.empty_cache()


# ==============================================================================
# 9. >>> NEW — CROSS-WINDOW AGGREGATION (mean +/- std instead of one point estimate)
# ==============================================================================
print("\n" + "=" * 70)
print(f"CROSS-WINDOW SUMMARY (n={NUM_EVAL_WINDOWS} windows) — Q4 Target Recovery, mean +/- std")
print("=" * 70)
header = f"{'Layer':<8}" + "".join(f"{f'B_{int(f*100)}%':<12}" for f in FRACTION_SWEEP) + f"{'A(100%)':<12}{'C(PreRoPE)':<12}"
print(header)
for layer_idx in LAYERS_TO_HOOK:
    row = f"{layer_idx:<8}"
    for f in FRACTION_SWEEP:
        vals = [wr['primary'][layer_idx][f'B_{f}']['Q4_target_recovery'] for wr in window_results]
        row += f"{np.mean(vals):5.1f}+/-{np.std(vals):<4.1f} "
    vals_a = [wr['primary'][layer_idx]['A']['Q4_target_recovery'] for wr in window_results]
    vals_c = [wr['primary'][layer_idx]['C']['Q4_target_recovery'] for wr in window_results]
    row += f"{np.mean(vals_a):5.1f}+/-{np.std(vals_a):<4.1f} {np.mean(vals_c):5.1f}+/-{np.std(vals_c):<4.1f}"
    print(row)

print("\n" + "=" * 70)
print("CROSS-WINDOW SUMMARY — STRICT literal-spec near/far labeling (5b cross-check)")
print("=" * 70)
print(header)
for layer_idx in LAYERS_TO_HOOK:
    row = f"{layer_idx:<8}"
    for f in FRACTION_SWEEP:
        vals = [wr['strict_near'][layer_idx][f'B_{f}']['Q4_target_recovery'] for wr in window_results]
        row += f"{np.mean(vals):5.1f}+/-{np.std(vals):<4.1f} "
    vals_a = [wr['strict_near'][layer_idx]['A']['Q4_target_recovery'] for wr in window_results]
    vals_c = [wr['strict_near'][layer_idx]['C']['Q4_target_recovery'] for wr in window_results]
    row += f"{np.mean(vals_a):5.1f}+/-{np.std(vals_a):<4.1f} {np.mean(vals_c):5.1f}+/-{np.std(vals_c):<4.1f}"
    print(row)

print("\n" + "=" * 70)
print("CROSS-WINDOW SUMMARY — Section 4.5 outlier bound-width divergence (%), mean +/- std")
print("=" * 70)
print(f"{'Fraction':<10}" + "".join(f"{layer_idx:<12}" for layer_idx in LAYERS_TO_HOOK))
for f in FRACTION_SWEEP:
    if f == 0.0:
        continue
    row = f"{f*100:.0f}%       "
    for layer_idx in LAYERS_TO_HOOK:
        vals = [wr['outlier'][layer_idx][f]['divergence_pct'] for wr in window_results]
        row += f"{np.mean(vals):6.2f}+/-{np.std(vals):<5.2f}"
    print(row)
print("(Divergence = (true_width - clipped_width) / true_width. Per README 4.5, this is a")
print(" diagnostic only -- high divergence at the frequency fractions that also look good in")
print(" the Q4 test is the trigger for a follow-up study on percentile-based bounds, not for")
print(" any action within this experiment.)")

if RUN_PERPLEXITY_IMPACT:
    print("\n" + "=" * 70)
    print("CROSS-WINDOW SUMMARY — Downstream perplexity impact (windowed PPL, mean +/- std)")
    print("=" * 70)
    all_tags = ['dense_baseline'] + PPL_METHODS
    print(f"{'Method':<14}{'PPL':<16}{'Delta vs dense':<16}")
    for tag in all_tags:
        ppls = [wr['ppl'][tag]['ppl'] for wr in window_results if tag in wr['ppl']]
        deltas = [wr['ppl'][tag]['delta_vs_dense'] for wr in window_results if tag in wr['ppl']]
        if not ppls:
            continue
        print(f"{tag:<14}{np.mean(ppls):6.3f}+/-{np.std(ppls):<6.3f}  {np.mean(deltas):+6.3f}+/-{np.std(deltas):<6.3f}")
    print("\nInterpretation: 'A' and 'B_best' should sit close to each other and both close to")
    print("dense baseline if the routing scheme (any dimensionality) is preserving what matters")
    print("for next-token prediction. 'C' having a larger delta than 'A'/'B_best' would be the")
    print("downstream-quality echo of the Q4 attention-mass result above.")

print("\n" + "=" * 70)
print("CROSS-WINDOW SUMMARY — Sensitivity verdict, aggregated over all windows")
print("=" * 70)
all_sens = [s for wr in window_results for s in wr['sensitivity']]
n_total = len(all_sens)
n_b_wins = sum(1 for s in all_sens if s[-1] == "B WINS")
n_inconclusive = sum(1 for s in all_sens if s[-1] == "INCONCLUSIVE")
n_c_wins = n_total - n_b_wins - n_inconclusive
print(f"Total (window x layer x context x percentile) combos: {n_total}")
print(f"  B WINS:        {n_b_wins} ({100*n_b_wins/n_total:.0f}%)")
print(f"  C WINS/TIE:    {n_c_wins} ({100*n_c_wins/n_total:.0f}%)")
print(f"  INCONCLUSIVE:  {n_inconclusive} ({100*n_inconclusive/n_total:.0f}%)")
