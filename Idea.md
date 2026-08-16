# RoPE-Frequency-Aware KV Page Routing — Detailed Resolved Proposal

## 0. Motivating Summary

Long-context decoding is memory-bandwidth-bound: every generated token requires reading a historical KV cache from HBM, and that read dominates latency far more than the attention compute itself. Existing query-aware routers (Quest, ShadowKV) already solve "read only the KV pages that matter" — but they pay for a full-dimensional routing index on every page, every step. This proposal asks a narrower question: **can the routing index itself be shrunk to just the RoPE dimensions that stay geometrically stable within a page, without losing recall relative to the best existing alternatives?**

The resolved design below folds in three structural fixes — a local bypass window, a sharpened theoretical wedge against the strongest competitor, and mandatory outlier instrumentation — so that the experiment tests exactly one variable at a time.

---

## 1. Core Objective (Restated Precisely)

> Determine whether a min/max routing index built **only** from the low-frequency subset of post-RoPE key dimensions matches the page-selection quality of a full-dimensional index (Quest-style), and — as the sharper, falsifiable claim — **beats Pre-RoPE routing specifically on far-away, weakly-similar target pages**, where positional decay is the only usable signal.

This is not a general "sparse attention is good" claim. It is a claim about which *subset of dimensions* carries the routing-relevant signal, isolated from every other design choice (page size, top-K, attention kernel, etc., all held fixed across conditions).

---

## 2. Resolved Architecture

### 2.1 Tier 1 — Always-On Local Window

**Design.** The most recent `W` tokens (rounded up to the nearest 1–2 physical pages) are never routed — they are always loaded and always included in exact attention, every decode step, regardless of what the router would have selected.

**Why this is structural, not optional.** High-frequency RoPE dimensions complete near-full rotations within a single page span, so a query's dot product against nearby keys stays sharply discriminative for recent tokens specifically *because* of the high-frequency channels. Tier 2's index deliberately excludes those channels. Without Tier 1, any perplexity degradation observed downstream would be ambiguous — it could reflect a genuine long-range routing failure (the thing being tested) or simply the model losing its most recent context (an artifact of the design, not a finding). Tier 1 removes that ambiguity by construction.

**Parameterization.** `W` is fixed once (e.g., 1–2 pages' worth of tokens, matched to whatever local-window size the baseline systems (Quest, ShadowKV) already assume, if any) and held constant across configurations A, B, and C. It is explicitly *not* part of the swept variable set — sweeping it would reintroduce a confound between "how much is locally exempted" and "how good is the long-range index," which is precisely what Tier 1 is meant to prevent.

### 2.2 Tier 2 — Low-Frequency Long-Range Index

Unchanged in mechanism from the original proposal, now scoped to apply only outside the Tier-1 window:

- RoPE is applied normally to keys; nothing about the model's forward pass changes.
- For each physical page outside the local window, store min/max bounds computed **only over the low-frequency dimension subset** of the post-RoPE keys.
- At decode time, project the live query onto the same low-frequency subset.
- Score each page's bounds against the projected query (identical scoring function to Quest — just fewer dimensions).
- Rank and keep top-K pages.
- Load **full-dimensional, unmodified** K/V for the selected pages only, and run ordinary exact attention.

High-frequency dimensions are never discarded from the model itself — only from the index used to decide *which pages to read*.

### 2.3 Combined Data Flow

```text
                         Query Q
                            │
              ┌─────────────┴─────────────┐
              │                           │
     Tier 1: local window            Tier 2: routing index
     (last W tokens, always          (low-freq post-RoPE
      loaded, no routing)             min/max, all older pages)
              │                           │
              │                      rank pages → top-K
              │                           │
              │                  read full K/V of top-K
              │                           │
              └─────────────┬─────────────┘
                             ▼
                    exact attention over
                 {local window ∪ selected pages}
```

---

## 3. Theoretical Wedge Against Baseline C (Pre-RoPE Scoring)

This is the part of the proposal that was previously a hand-wave ("let's see empirically") and is now a specific, checkable mechanism.

**The claim.** RoPE rotates each dimension-pair of a key vector by an angle proportional to `position × θ_d`, where `θ_d` decreases geometrically with dimension index `d`. For low-frequency `d`, `θ_d` is tiny, so:

- **Within a page** (a narrow span of positions), the rotation is nearly constant → tight, stable min/max bounds, good for identifying *which page this is*.
- **Across pages** (a wide span of positions), the rotation still accumulates monotonically, just slowly → the *center* of a page's low-frequency bounds drifts systematically as a function of distance from the query.

So the low-frequency index carries two signals simultaneously: a content fingerprint (from the key values themselves) and a coarse, monotonic function of relative distance (from the residual rotation). Pre-RoPE scoring (baseline C) strips positional information out entirely before scoring, retaining only content similarity.

**The testable consequence.** If this mechanism is real, method B (low-frequency post-RoPE) should have a measurable recall advantage over method C specifically in the region where content similarity alone is uninformative but distance is informative: **target pages that are far from the query and weakly similar in content, yet still attended to** (e.g., a fact restated in different words far earlier in a document). In every other quadrant (near/high-sim, near/low-sim, far/high-sim), B and C are expected to be roughly comparable, because either the local window already covers it (near) or content similarity alone suffices (high-sim). The "far / low-sim" quadrant is the only place the mechanism predicts a gap — which is why it becomes the decisive stratum in the metric design (Section 5).

**Falsification condition.** If B does not outperform C in the far/low-sim quadrant, the mechanism is wrong (or too weak to matter at realistic page granularities), and the original team's choice to abandon RoPE for scoring (baseline C's design decision) is vindicated. That is an acceptable, informative negative result — not a failure of the experiment.

---

## 4. Experimental Protocol

### 4.1 Conditions

| Selector | Dimensions used | What it represents |
|---|---|---|
| **A** | Full post-RoPE, all dims | Strong baseline (Quest) |
| **B** | Post-RoPE, low-frequency subset only, fraction swept | Proposed method |
| **C** | Pre-RoPE keys | Strongest published alternative (Full Attention Strikes Back) |

All three share: same page size, same top-K, same Tier-1 local window, same underlying model and attention kernel. Only the routing index differs.

### 4.2 Swept Variable

Fraction of low-frequency dimensions retained in B's index: **0%, 5%, 10%, 15%, 25%, 50%, 75%, 100%**, sliced sequentially from lowest frequency upward (never a random subset — frequency order is the whole point). Denser sampling at the low end because that is where a discriminative-vs-noisy cliff, if it exists, is expected to sit. At 0%, B degenerates to "no routing signal" (a sanity floor); at 100%, B collapses to A (a sanity ceiling / consistency check).

### 4.3 Held Constant

- `W` (Tier-1 local window size) — fixed once, identical across A, B, C.
- Page size, top-K, model checkpoint, attention kernel, dataset/task suite.

### 4.4 Metrics

**Primary: Attention-mass recall.** For each decode step, compute the fraction of the true (full-context) attention distribution's mass that lands on the pages actually selected by the router. This is preferred over raw Page Recall@K because a router can retrieve a page that ultimately receives negligible attention weight — attention-mass recall directly measures whether the *right* pages (in terms of what the model would actually use) were kept.

**Critical stratification.** Attention-mass recall is broken out into four quadrants per query-target pair:

| | High content-similarity | Low content-similarity |
|---|---|---|
| **Near (within/just outside Tier 1)** | Q1 | Q2 |
| **Far (long-range)** | Q3 | **Q4 — decisive quadrant** |

Success for method B is defined specifically as: **B's attention-mass recall in Q4 meets or exceeds C's**, at some point on the frequency-fraction sweep, at a routing cost (latency + metadata bandwidth) equal to or lower than C's. Parity or superiority in Q1–Q3 is expected and reassuring but not the deciding test.

**Secondary / cost metrics** (measured alongside, for the full recall-vs-cost curve):
- Selector latency (wall-clock per decode step, routing computation only).
- Metadata size / bandwidth (bytes of index read per step).
- KV bytes actually loaded end-to-end.
- End-to-end decode latency and GPU HBM utilization.
- Downstream task quality: perplexity and long-context benchmark accuracy (e.g., needle-in-haystack-style tasks, since these are exactly the far/low-sim scenario Q4 targets).

### 4.5 Mandatory Instrumentation — Outlier Sensitivity

Because B's index uses strictly fewer dimensions than A's, any single outlier activation within the retained low-frequency subset has proportionally more influence over a page's min/max bounds than it would in a full-dimensional index (where its distortion is diluted across many more dimensions).

**Protocol.** For every configuration of B (each point on the frequency sweep), compute per-page bound width twice: once using true min/max, once using a top-1%-magnitude-clipped min/max. Track the divergence between the two as a function of frequency fraction. This is recorded as a diagnostic time series, not acted upon within this experiment.

**Decision rule for follow-up work (explicitly out of scope here):** if clipped vs. unclipped bound widths diverge sharply at the frequency fractions that otherwise look most promising in the Q4 recall test, that is the trigger to investigate percentile-based bounds as a mitigation in a subsequent study — not this one.

---

## 5. Decision Tree for Interpreting Results

1. **Does B ever reach A's recall** (any quadrant, any frequency fraction) below 100% dimensions? If yes at fraction `f`, that is the operating point of interest for cost comparison; if only at ~100%, the low-frequency hypothesis is unsupported — the full dimensionality was doing real work.
2. **Does B beat or match C specifically in Q4** at that same (or lower) cost? 
 - **Yes** → the macro-positional-decay claim holds; this is the paper's core positive result.
 - **No** → legitimate negative result; report it as confirmation that Pre-RoPE scoring's design choice (discard position, keep pure semantics) was correct, and note where B still adds value, if anywhere (e.g., cost savings in Q1–Q3 despite no Q4 edge).
3. **Outlier diagnostic** informs whether any future work on this idea should start with percentile bounds instead of true min/max — independent of the primary yes/no outcome above.

---

## 6. What Remains Explicitly Out of Scope

Unchanged from the original scope boundaries — kept here for completeness:

- Adaptive top-K based on score concentration (secondary, only after core result lands).
- GPU-resident metadata layout, coalesced reads, GQA/MQA compatibility, fused kernels — real-implementation concerns, not part of what this experiment needs to establish.
- Outlier mitigation itself (percentile bounds, clipping strategies) — instrumented but deferred, per Section 4.5.

---

## 7. One-Paragraph Abstract (Final)

We propose a two-tier KV routing scheme for long-context decoding: an always-on local window handles near-neighbor precision via full-dimensional attention, while a compact index built strictly from low-frequency post-RoPE key dimensions routes selection of all older KV pages. We hypothesize this low-frequency index retains a coarse, monotonic positional-decay signal that content-only Pre-RoPE routing discards, and predict a specific, falsifiable advantage in attention-mass recall for targets that are positionally distant but only weakly similar in content — the one quadrant where positional signal is not redundant with semantic signal. The experiment sweeps the retained frequency fraction against two established baselines (full-dimensional post-RoPE routing and Pre-RoPE routing), reports stratified recall-vs-cost curves, and separately instruments — without yet mitigating — the index's sensitivity to activation outliers introduced by dimensionality reduction.
