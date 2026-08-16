# Query-Aware Frequency-Domain KV Cache Routing 
# Technical Documentation

---

## 1. Purpose of This Document

This document explains, in implementation-level detail, what problem this project solves, what solution it proposes, and how that solution is built and validated. It is written as a technical reference for someone who needs to understand the system well enough to reproduce it, extend it, or evaluate whether the approach is sound — not as a marketing summary.

---

## 2. The Problem

### 2.1 Background: why KV-cache routing is needed at all

In transformer decoding, every generated token attends to every previous token's key and value vectors (the "KV cache"). As context length grows, this cache grows linearly, and reading it back on every decode step becomes the dominant memory-bandwidth cost of inference. For long-context serving, this is the primary bottleneck — not compute, but the sheer volume of KV data that must be read per step.

**Paged, query-aware KV-cache routing** (the Quest-style approach this project builds on) reduces this cost by not attending to the full cache on every step. Instead:

1. The cached sequence is split into fixed-size **pages** (64 tokens each in this project).
2. A small **Tier-1 local window** — the 2 pages nearest the current token — is always attended to in full. Recent context is almost always relevant, so this is not worth routing.
3. Everything older than Tier-1 is the **routable pool**. For this pool, a cheap **per-page relevance score** is computed from the current query vector and a compact summary of each page's keys. Only the top-4 highest-scoring pages are actually loaded and attended to.

This turns attention cost from "read the whole cache" into "read Tier-1 plus a handful of selected pages" — which is the entire point of the technique.

### 2.2 Why the obvious implementation is expensive

The most accurate version of this router scores each page using **full-dimension, post-RoPE min/max bounds** of that page's keys, compared against the **full query vector**. This is accurate — it is query-aware and preserves fine-grained per-dimension information — but it is expensive in two specific ways:

- **Metadata storage**: for every page, on every head, you must store a `min` vector and a `max` vector spanning the *entire* head dimension. This metadata must be read on every decode step just to decide what to load — before any actual KV data is even touched.
- **Scoring compute**: the score itself is a ReLU-split dot product that runs over the full head dimension, for every candidate page, every head, every query.

A cheaper alternative that is sometimes used in practice replaces this with a **single pre-RoPE mean key vector per page**, scored by cosine similarity to the query. This is far cheaper to store (one vector instead of a min/max pair) and cheap to score, but it throws away two things:
- The position/rotation information that RoPE encodes (it operates on **pre-RoPE** keys).
- All per-token variance inside a page, since the page is collapsed to a single "average" direction.

### 2.3 The actual question this project investigates

Is there a middle ground: a routing signal that is nearly as accurate as the expensive full-dimension router, but far cheaper to store and compute, and reliably better than the cheap mean-vector alternative?

### 2.4 The specific hypothesis

RoPE rotates each pair of key/query dimensions by an angle proportional to `position × frequency`. Different dimension-pairs are assigned different frequencies:

- **Low-frequency dimension pairs** rotate slowly across positions. Their values change gradually across a page, so a page's min/max bounds on these dimensions stay tight and meaningful even after RoPE has been applied.
- **High-frequency dimension pairs** rotate through many full cycles within a single 64-token page. Their per-page min/max bounds end up spanning close to the dimension's entire possible range — essentially noise for routing purposes.

**Hypothesis**: if only the lowest-frequency fraction of the post-RoPE head dimension is kept in the page bounds (a tunable fraction `f`, tested at `{0%, 5%, 10%, 15%, 25%, 50%, 75%, 100%}` of the half-dimension, applied symmetrically to both RoPE halves), most of the accuracy of the full-dimension router should be recoverable, while metadata and compute cost shrink roughly linearly with `f`.

---

## 3. The Solution

### 3.1 What was built

An empirical validation harness — not a training run — that:

1. Extracts real attention tensors from a real, frozen language model (`TinyLlama-1.1B-Chat-v1.0`, fp16) processing real text (WikiText-2).
2. Computes what each of three routing methods **would have selected** at each query position.
3. Compares those selections against ground-truth attention mass to measure how much "important" attention each method actually captures.
4. Separately measures the **real** effect on next-token perplexity by actually masking attention according to each method's selections and running a real forward pass.

Nothing here is simulated. The model is a real checkpoint, the text is real, and the "routed" forward pass is a genuine forward pass with attention masked exactly as a production router would mask it.

### 3.2 The three methods compared

| Method | Signal | Query-aware? | Storage cost |
|---|---|---|---|
| **A — Full (baseline)** | Full-dimension, post-RoPE min/max page bounds vs. full query | Yes | Highest |
| **B — Frequency-domain (proposed)** | Low-frequency fraction `f` of post-RoPE min/max bounds vs. matching query dimensions | Yes | Scales linearly with `f` |
| **C — Pre-RoPE mean (baseline)** | Single pre-RoPE mean key vector per page, cosine similarity to query | No (position-blind) | Lowest |

At `f = 1.0`, Method B is numerically identical to Method A. At `f = 0`, Method B degenerates to genuine random page selection — an explicit sanity floor representing "no signal at all," not a degenerate one-dimensional signal.

### 3.3 Configuration used

- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, fp16.
- **Document windows**: 3 independent, non-overlapping 2048-token windows from real WikiText-2 test text, spaced 6000 tokens apart, so results are reported as mean ± std across genuinely different text.
- **Paging**: `PAGE_SIZE = 64` tokens, Tier-1 window = 2 pages, top-K routed pages = 4.
- **Evaluation positions**: the last 256 token positions of each window.
- **Layers probed**: 4, 10, 16 (early / mid / late).
- **Frequency-fraction sweep**: `{0%, 5%, 10%, 15%, 25%, 50%, 75%, 100%}` of the low-frequency half-dimension.
- **Outlier-sensitivity clip**: 1st/99th percentile, used only as a diagnostic on bound width, never fed back into routing decisions.

---

## 4. Implementation Design — How Each Piece Works

### 4.1 Step 1: Real tensor extraction

A forward hook is registered on each hooked layer's self-attention module. During a single real forward pass, the hook:

1. Recomputes `q_proj` / `k_proj` on the real hidden states (handling grouped-query attention head repetition) to obtain **pre-RoPE** key states.
2. Applies the model's own RoPE rotation (matching the model's `cos`/`sin` convention) to obtain **post-RoPE** query and key states.
3. Computes the **exact**, dense, causally-masked attention probability matrix via `softmax(QKᵀ/√d)`. This is the ground truth every routing method's selections are scored against.

Output per hooked layer: `pre_rope_k`, `post_rope_k`, `q` (post-RoPE), `exact_attn`.

### 4.2 Step 2: An independent relevance signal for evaluation

To evaluate not just *whether* a page was selected but *why it mattered*, a model-free signal is built: for each query position, the last 64 tokens are taken as the query's "topic," decoded to text, and compared against each candidate page's decoded text using **Jaccard word-overlap similarity**. This signal is entirely decoupled from any routing method's own Q/K-based scoring — it cannot structurally favor A, B, or C, because none of them use it to make decisions. It exists purely to *label* pages for evaluation purposes. A threshold (default: 50th percentile of observed similarities) splits pages into "high" vs. "low" lexical similarity.

### 4.3 Step 3: The Q1–Q4 evaluation framework

Every routable page, for every query, is classified along two independent axes:

- **Distance**: near (just past Tier-1) vs. far.
- **Topical relevance**: high lexical similarity vs. low lexical similarity (from Step 2).

This produces four quadrants:

| | High lexical similarity | Low lexical similarity |
|---|---|---|
| **Near** | Q1 | Q2 |
| **Far** | Q3 | **Q4** |

**Q4 is the case that matters most**: pages that are far away, topically unrelated by a naive text-overlap measure, yet still receive real, non-trivial attention mass from the model. A naive recency-based or similarity-based heuristic would never surface these pages. A page counts as a true Q4 target if its ground-truth attention mass exceeds the per-query median page mass (i.e., it's actually important) **and** it is far **and** it is lexically dissimilar.

**Q4 target recovery** — the fraction of that hard-to-find attention mass that a method actually selects — is the project's central accuracy metric.

### 4.4 Step 4: Scoring logic for each method

For every evaluated query and every method:

- **Method A**: `score_pages_quest_style` — a ReLU-split dot product between the full-dimension query and each page's full-dimension post-RoPE min/max bounds.
- **Method C**: cosine similarity between the query and each page's pre-RoPE mean key vector.
- **Method B_f**: identical scoring logic to A, but restricted to `get_low_freq_indices(head_dim, f)` — the lowest-frequency `f`-fraction of each RoPE half — applied consistently to both the query and the page bounds.

For all methods, the top-4 scoring pages per head are selected, and the ground-truth attention mass captured by (Tier-1 + selected pages) is recorded as **recall**.

Three additional measurements are collected in this same step to support the cost/benefit case:

- **Outlier diagnostics**: for each frequency fraction, how much would bound width shrink if the top/bottom 1% of token magnitudes were clipped — a check on whether B's bounds are driven by a handful of extreme values or by genuinely informative dimension behavior.
- **Selector latency**: real, GPU-synchronized wall-clock timings of the actual page-scoring computation for each method, averaged over 30 trials.
- **Metadata bytes**: exact per-page metadata footprint in fp16 for each method — A stores full min+max (largest), C stores one mean vector (smallest), B scales linearly with the chosen fraction.

### 4.5 Step 5: Labeling and aggregation (decoupled from routing)

Routing decisions (Step 4) do not depend on how "high lexical similarity" is defined. This separation is deliberate: the expensive part (touching the model, computing real Q/K/attention) is done once and cached; the cheap part (applying a labeling threshold and aggregating recall/Q4-recovery statistics) can then be re-run many times over different labeling parameters without re-running the model. This is what makes the later 27-combination sensitivity sweep computationally feasible.

### 4.6 Step 6: Real downstream perplexity measurement

This is the strongest evidence produced by the project, because it does not depend on the Q4-labeling heuristic at all — it measures actual model output quality.

A forward hook recomputes attention exactly as the model normally would, but for the evaluated query positions, masks out every key position **except** Tier-1 and that method's selected pages (all other positions get `-inf` before softmax). This genuinely restricts what the model can attend to, at the same granularity a production router would enforce. Positions outside the evaluated set are left untouched, avoiding any confound from modifying the whole sequence.

Procedure per document:
1. Compute baseline next-token loss with the model completely unmodified.
2. For methods A, C, and B (at its Q4-recovery-optimal fraction for that document/layer), patch **one layer's** attention with that method's actual selections and re-run a full forward pass, reading real teacher-forced cross-entropy over the evaluated positions.
3. Report `Δnats = loss_routed − loss_baseline`.

### 4.7 Step 7: Cross-document and sensitivity aggregation

- **Cross-document summary**: Q4 recovery and perplexity delta are aggregated as mean ± std across the 3 independent text windows, per layer.
- **Labeling-robustness sweep**: the *cached* routing decisions are re-scored under 9 combinations of `context_tokens ∈ {32, 64, 128}` × `percentile ∈ {25, 50, 75}`, across all 3 layers (27 combinations total). Each combination lets Method B re-select its own best fraction, since the optimal fraction can shift depending on how "hard" pages are defined. Each combination is classified as `B WINS`, `C WINS/TIE`, or `INCONCLUSIVE` (zero eligible Q4 cases), and an overall verdict is printed.

---

## 5. Results

### 5.1 Representative per-document, per-layer detail (Document 1, Layer 4)

| Method | Dims kept | Recall | Q4 Recovery | Selector latency (µs) | Metadata bytes |
|---|---|---|---|---|---|
| C (pre-RoPE mean) | — | 13.12% | 14.13% | 580.0 | 131,072 |
| B, 0% (random floor) | 0% | 24.20% | 14.66% | 102.2 | 0 |
| B, 5% | 5% | 91.43% | 37.29% | 238.2 | 8,192 |
| B, 25% | 25% | 96.71% | 42.23% | 235.5 | 65,536 |
| B, 75% | 75% | 96.81% | 45.09% | 237.6 | 196,608 |
| B, 100% | 100% | 96.78% | 45.16% | 231.4 | 262,144 |
| A (full, baseline) | 100% | 96.78% | 45.16% | 141.8 | 262,144 |

Real perplexity impact at this (document, layer): baseline `ppl = 15.06`; A: Δ = −0.004 nats; C: Δ = +0.301 nats (measurable degradation); B at 100%: Δ = −0.004 nats (indistinguishable from A). This pattern — B closing most of the gap to A by the 25–75% fraction range, C lagging far behind on Q4 recovery while visibly hurting perplexity — repeats across all documents and layers tested.

**DOCUMENT-1:**
<img width="1286" height="886" alt="document1_method_ABC_benchmark" src="https://github.com/user-attachments/assets/92fa63c6-fade-4300-8d47-a1ac26a3607d" />

**DOCUMENT-2:**
<img width="2315" height="1594" alt="document2_method_ABC_benchmark" src="https://github.com/user-attachments/assets/50ca6159-cbd5-44a0-81ca-94d4e8fb595b" />

**DOCUMENT-3:**
<img width="2315" height="1594" alt="document3_method_ABC_benchmark" src="https://github.com/user-attachments/assets/7236ff39-ef73-4aa7-95e2-e197e8512484" />




### 5.2 Cross-document summary (mean ± std across 3 real WikiText windows)

| Layer | Method | Q4 recovery % | Perplexity Δ (nats) |
|---|---|---|---|
| 4 | C | 10.95 ± 2.58 | +0.4696 ± 0.2730 |
| | A | 63.51 ± 13.72 | −0.00003 ± 0.0029 |
| | B (best) | 63.64 ± 13.77 | −0.00009 ± 0.0029 |
| 10 | C | 16.00 ± 2.83 | +0.0546 ± 0.0241 |
| | A | 71.37 ± 15.75 | +0.0012 ± 0.0018 |
| | B (best) | 71.42 ± 15.77 | +0.0009 ± 0.0017 |
| 16 | C | 20.83 ± 3.06 | +0.0801 ± 0.0389 |
| | A | 63.57 ± 15.04 | +0.0016 ± 0.0096 |
| | B (best) | 65.14 ± 14.99 | +0.0006 ± 0.0114 |

B beats C in Q4 recovery on all 3 documents at every layer tested. At its best fraction, B is statistically indistinguishable from the far more expensive A on both Q4 recovery and real perplexity. C, despite being cheapest to store, recovers roughly a third as much hard-to-find attention mass and measurably raises real perplexity.

The best fraction for B varied by document and layer (50–100%), with 75% and 100% most common — meaning the low-frequency half of the head dimension is not always sufficient on its own, but a majority of the full dimension can typically be dropped without accuracy loss relative to A.

Outlier bound-width diagnostics at each document's best fraction showed 8–11% divergence between true and top-1%-clipped bound widths — a modest, not extreme, sensitivity to outlier magnitude, suggesting B's bounds are not dominated by a handful of extreme activations.
<img width="960" height="540" alt="Q4 Recovery Across Layers" src="https://github.com/user-attachments/assets/2fcd901d-c761-4909-bd1c-d252429ed3f2" />


### 5.3 Labeling-parameter sensitivity sweep

Across all 27 tested `(context_tokens, percentile)` combinations:

- **B wins: 24/27 (89%)**
- **C wins/ties: 3/27 (11%)** — all three at `context_tokens = 128, percentile = 25`, the most permissive Q4 definition, one occurrence per layer.
- **Inconclusive: 0/27**

The result leans toward B but is not uniform: some labeling combinations flip the outcome. This is reported as a boundary condition, not an unconditional win — the B > C conclusion is robust under the large majority of reasonable labeling choices, but not universal.

---

## 6. Interpretation

1. **RoPE frequency structure is a real, exploitable signal for cheap KV routing.** Keeping only the low-frequency fraction of post-RoPE key/query dimensions in the page bounds preserves most of the routing information a full-dimension router uses. Accuracy saturates well before 100% of dimensions are kept, and even a modest 25–50% fraction clearly outperforms the position-blind mean-vector baseline.
2. **Position information matters more than raw semantic similarity for this task.** Method C is cheap but consistently the worst performer on both Q4 recovery and real perplexity, and is the only method that measurably hurts real model quality relative to the unsparsified baseline.
3. **The gains are not an artifact of one arbitrary evaluation choice.** Both the 3-document cross-validation and the 27-combination labeling sensitivity sweep point the same direction, with the sweep explicitly flagging the one region (very loose Q4 definitions) where the result is less clear-cut.
4. **Cost savings are real and roughly linear in the kept fraction.** Metadata footprint scales linearly with the chosen fraction — e.g. at 25% it is roughly a quarter of A's footprint — while recovering the great majority of A's Q4 recovery and perplexity performance in that same range.

---

## 7. Limitations

- Only one model family/size was tested (`TinyLlama-1.1B`); frequency-domain routing behavior could differ at other scales, architectures, or RoPE configurations (e.g. NTK-aware scaling, ALiBi variants).
- Only 3 layers (4, 10, 16) and 3 document windows were evaluated — sufficient for a first-pass variance estimate, not a large-scale statistical study.
- The Q4 "hard case" framework depends on a specific lexical-similarity proxy for topical relevance. The sensitivity sweep shows the qualitative conclusion holds broadly, but it is not perfectly invariant to this choice.
- Perplexity evaluation patches only one layer at a time. Cascading effects of routing every layer simultaneously in a real serving system are not measured.

---

## 8. Implementation Reference (by Notebook Cell)

| Cell | Purpose |
|---|---|
| 1 | Configuration: model name, paging/Tier-1/top-K sizes, frequency-fraction sweep, labeling defaults and sensitivity grids, outlier-clip thresholds, number of document windows, device selection. |
| 2 | Loads model (fp16) and tokenizer with retry logic; loads real WikiText-2 test split; slices into non-overlapping 2048-token windows. |
| 3 | Forward hook on each hooked layer's self-attention module, extracting pre-RoPE K, post-RoPE Q/K, and exact dense attention probabilities from a real forward pass. |
| 4 | `LexicalSignal`: model-free, per-document Jaccard word-overlap similarity between query context and candidate page text, used for Q1–Q4 labeling. |
| 5 | Routing-engine primitives: low-frequency dimension selection (`get_low_freq_indices`), page min/max bound computation, Quest-style page scoring, outlier-clip diagnostics, selector-latency benchmarking, metadata-byte accounting per method. |
| 6 | `compute_routing_decisions`: for every evaluated query, computes near/far classification, ground-truth per-page attention mass, and top-K selections for A, C, and every B fraction. |
| 7 | `score_with_labeling`: applies a given `(context_tokens, percentile)` labeling configuration to cached routing decisions to compute recall, Q4 recovery, Q4-eligible counts, and Q1–Q3 mass statistics. |
| 8 | `make_routed_attention_hook` / `eval_next_token_loss`: real forward-pass patching enforcing each method's actual page selections in one layer's attention, measuring real teacher-forced cross-entropy. |
| 9 | Main loop: extraction → routing → outlier diagnostics → labeling → real perplexity evaluation for every document × layer combination, with `[SUCCESS]/[NEGATIVE RESULT]/[INCONCLUSIVE]` verdict per run. |
| 10 | Cross-document summary: mean ± std of Q4 recovery and perplexity delta per layer, plus outlier bound-width divergence at each document's best fraction. |
| 11 | Labeling-robustness sweep: re-scores cached routing decisions across 27 `(context_tokens, percentile)` × layer combinations, tallies win/tie/inconclusive counts, prints overall robustness verdict. |

---

## 9. Glossary

- **Tier-1 window**: the fixed-size local window of most-recent pages that always receives full attention, regardless of routing.
- **Routable pool**: all pages older than Tier-1, from which the router selects the top-K to additionally attend to.
- **Page bounds**: per-page, per-dimension min/max of key vectors within a page, used as a compact summary for Quest-style scoring.
- **Q1–Q4 quadrants**: a 2×2 classification of routable pages by (near/far) × (lexically similar/dissimilar), used to isolate the "hard" case (Q4: far and dissimilar) that a naive router would miss.
- **Q4 target recovery**: the fraction of ground-truth attention mass in true-Q4-labeled pages that a routing method actually selects.
- **Frequency fraction (f)**: the proportion of the low-frequency half of the post-RoPE head dimension retained for Method B's page-bound scoring.
- **Nats**: natural-log units of cross-entropy loss; perplexity = exp(mean cross-entropy in nats).
