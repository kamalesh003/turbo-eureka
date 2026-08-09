# RoPE-Frequency-Aware KV Page Routing

## Intention

Long-context LLM decoding is increasingly bottlenecked not by compute, but by **memory traffic**: at every generated token, one new query vector must be compared against a historical KV cache that can span tens or hundreds of thousands of tokens, and reading that cache from GPU HBM dominates decode latency.

The intention of this idea is **not** to make KV reads faster, and **not** to propose a new caching architecture. Hierarchical query-aware page selection (cheap filter → top-K candidate pages → exact attention on survivors) is already established by systems like Quest. The intention is narrower and more specific:

> **To test whether a compact routing index built from only the low-frequency RoPE dimensions of post-RoPE keys can select the same important KV pages as a full-dimensional index — at substantially lower routing cost — and to determine where, if anywhere, this beats the existing alternatives.**

In one sentence: *don't make every KV read faster — make the decision about which KV pages deserve to be read cheaper, using RoPE's own frequency structure as the mechanism for that decision.*

---

## Core Idea

Reorganize the KV cache from passive tensor storage into a query-routable structure with two clearly separated concerns:

1. **Routing** (approximate) — decide which physical KV pages are worth reading.
2. **Attention** (exact) — once pages are selected, read and attend over their complete, unmodified K/V.

Nothing about the underlying Transformer or its attention computation changes. Only the process of deciding *which historical pages get physically loaded* changes.

```text
                         Query Q
                            │
                            ▼
                ┌─────────────────────┐
                │ Compact KV-page     │
                │ routing index       │
                │ (low-frequency      │
                │ post-RoPE dims only)│
                └──────────┬──────────┘
                           │
                       rank pages
                           │
                           ▼
                         Top-K
                           │
                           ▼
                  Read full K/V only
                  from selected pages
                           │
                           ▼
                    Exact attention
```

---

## The Specific Mechanism

RoPE applies rotations at different frequencies to different dimension-pairs of each key vector. Within a page covering a narrow range of token positions, the **low-frequency** dimensions rotate very little across that range — their values stay relatively stable and therefore make a tight, discriminative page-level summary. The **high-frequency** dimensions rotate almost a full cycle within the same narrow range, so a page-level summary built from them is inherently loose, regardless of how it's computed.

The proposed routing index therefore:

- Applies RoPE normally (nothing about the model changes).
- For each physical KV page, stores **min/max statistics computed only over the low-frequency dimension subset** of the post-RoPE keys.
- At each decode step, projects the live query onto the same frequency subset and scores each page's min/max bounds against it — exactly Quest's scoring method, just on fewer dimensions.
- Ranks all pages by this cheap score and keeps the top-K.
- Reads the **full, unrestricted** K/V (all dimensions, all channels) only for the selected top-K pages, and runs ordinary exact attention on them.

The high-frequency dimensions are never discarded from the model — they are excluded only from the cheap routing index, and participate fully once a page is selected.

```text
Full K
 │
 ├───────────────┐
 │               │
 ▼               ▼
Low-frequency    High-frequency
RoPE dimensions  RoPE dimensions
 │               │
 ▼               X (excluded from index only)
min/max index
 │
 ▼
page score
 │
 ▼
Top-K pages
 │
 ▼
full K/V (all dimensions)
 │
 ▼
exact attention
```

**Routing is always rank-then-top-K, never a fixed score threshold** — because attention scores are softmax-normalized across the whole candidate set, so no fixed cutoff has principled meaning; the same score can matter enormously in one query's context and be negligible in another's.

---

## What Is Explicitly Not Claimed as Novel

- Paged KV storage
- Query-aware KV page selection generally (Quest already does this)
- KV cache as an "active"/routable memory structure
- KV compression, sparse attention, or RoPE-aware compression as general categories

## What Is Claimed as Novel

The specific, narrow combination:

> **Post-RoPE key vectors** + **low-frequency-only dimension subset** + **min/max page statistics** + **top-K routing.**

No published method sits at exactly this point:

| Approach | RoPE kept for routing? | Frequency subset? | Page min/max? |
|---|---|---|---|
| Quest | ✅ | ❌ (full-dim) | ✅ |
| ShadowKV | ✅ | ❌ | ❌ (mean/cosine landmarks) |
| Full Attention Strikes Back | ❌ (scores pre-RoPE) | — | ❌ |
| SALS / EliteKV | pre-/transformed representations | different objective | ❌ |
| **This proposal** | ✅ | ✅ | ✅ |

---

## The Central Hypothesis (Falsifiable)

> Can a low-frequency post-RoPE page index identify the important KV pages with comparable top-K recall to a full-dimensional page index, while substantially reducing routing computation and metadata bandwidth?

This is explicitly framed as an open, testable question — not an assumed result.

---

## The Critical Experiment

Compare three selectors head-to-head:

- **A — Full-dimensional post-RoPE min/max** (Quest's method, the strong baseline)
- **B — Low-frequency post-RoPE min/max** (this proposal)
- **C — Pre-RoPE scoring** (the strongest published neighbor, from "Full Attention Strikes Back")

Sweep the fraction of frequency dimensions used by B: 0%, 5%, 10%, 15%, 25%, 50%, 75%, 100% (finer granularity at the low end, since that's where any interesting cliff in recall is likely to sit).

For each configuration, measure:

- Page Recall@K and attention-mass recall (quality of routing)
- Selector latency and metadata size/bandwidth (cost of routing)
- KV bytes actually read, end-to-end decode latency, GPU HBM utilization
- Downstream output quality (perplexity, long-context task accuracy)

**The result that matters** is not whether B beats A — a smaller index is trivially cheaper at some point on the sweep. It's whether B's recall-vs-cost curve reaches or exceeds **C's** curve. C is the real bar, because it represents a prior team's considered choice to abandon RoPE for scoring rather than subset it. If B never matches C, that's a legitimate negative result (confirms C's design choice was right). If B matches or beats C, that's the actual contribution.

---

## Scope Boundaries

- **Core, required result**: the three-way selector comparison above.
- **Secondary, optional, only after the core result lands**: adaptive top-K based on routing-score concentration (spend more bandwidth only on queries whose page scores are ambiguous).
- **Systems-engineering, not scientific claims**: GPU-resident metadata layout, coalesced page reads, GQA/MQA compatibility, fused routing+attention kernels — worth doing for a real implementation, but not part of what the experiment is meant to establish.

---

## One-Sentence Abstract

> We investigate a frequency-selective post-RoPE representation for KV-page routing, using compact min/max summaries over low-frequency RoPE components to reduce the cost of query-dependent top-K page selection in long-context decoding.
