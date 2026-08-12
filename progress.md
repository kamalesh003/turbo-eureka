I read through both the source and the full run output. Here's the breakdown against the README's protocol.

## What This Run Actually Proves

**1. The core Q4 falsification test passes, and passes non-trivially.**
At all three probed layers, Method B beats Baseline C in the decisive far/low-sim quadrant, by a large margin — not a marginal edge:

| Layer | C (Q4 recovery) | B best (Q4 recovery) |
|---|---|---|
| 4 | 7.77% | 46.46% (at 10% dims) |
| 10 | 14.73% | 46.06% (at 100% dims) |
| 16 | 14.23% | 47.53% (at 75% dims) |

That's roughly a 3–6× gap, consistently in B's favor. This is the specific, falsifiable prediction from Section 3 of the README, and it did not falsify.

**2. The circularity problem in defining Q4 was caught and fixed, and the result survives it.**
Your first attempt at this (implied by the code comments) must have had label leakage — using the routing methods' own scores to define "low content-similarity," which would bias the test toward whichever method built the labels. The fixed version uses a **model-free lexical Jaccard signal** (word-overlap on decoded token text) as the content-similarity proxy, fully decoupled from Q/K vectors. This was exactly the gap I flagged as unresolved before — it's now closed, and it's the right fix (an independent proxy, not a shared embedding space).

**3. Robustness check on the Q4 labeling parameters — done, and honestly reported.**
27 combinations of (layer × context-window × similarity-percentile) were tested. B wins in 25/27 (93%), with 2 borderline losses both at the widest context window (128 tokens) and loosest threshold (p25). The notebook's own verdict — "leans toward B, not uniform, report as boundary condition" — is the correct, non-oversold way to state this.

**4. Sanity checks pass.** B at 100% dims exactly matches Baseline A across all three layers (e.g., 96.78% / 96.78% at layer 4) — this is the internal consistency check the README implicitly wants, and it confirms the implementation isn't silently buggy.

**5. Recall saturates fast** — B reaches ~91–95% aggregate recall using only 5% of low-frequency dimensions. That's a genuine, useful finding, though it also means aggregate Recall isn't the discriminating metric here (it saturates too easily); Q4 recovery is doing all the real work in this experiment, which is fine since that was always the intended decisive metric.

## What's Still Unproven / Not Yet Run

**1. Section 4.5 (Mandatory Outlier Instrumentation) — not implemented at all.** I grepped the whole notebook for clip/outlier/percentile-bound logic — nothing. The README calls this out explicitly as *mandatory*, not optional. This is the single clearest gap between what was promised and what was run.

**2. No cost metrics.** Section 4.4's "Secondary/cost metrics" — selector latency, metadata bandwidth, KV bytes loaded, end-to-end decode latency, HBM utilization — are entirely absent. Only a coarse per-layer wall-clock ("layer done in 54.8s" for the *whole* experiment loop) is logged, which isn't selector-only latency. Without this, you can't yet make the cost-normalized claim the README's Section 4.4 success criterion actually requires ("meets or exceeds C's recall **at equal or lower routing cost**").

**3. No downstream task quality.** No perplexity, no needle-in-haystack-style benchmark — the README names these explicitly as the natural Q4 downstream proxy. Right now the claim is purely about attention-mass recovery, one level removed from output quality.

**4. Tiny scale relative to the paper's own framing.** The motivating summary is about *long-context* decoding, but this run uses a single 2048-token WikiText document, 32 total pages, 3 of the model's layers (out of TinyLlama's 22), one 1.1B model, one dataset, one document, no repeated seeds. There's no variance/confidence interval — each number is a point estimate from one run. This is a solid pilot, not yet the "real" experiment the README describes (multiple models, longer contexts, a task suite).

**5. Fraction sweep is close but not exact to spec.** README wanted `{0%, 5%, 10%, 15%, 25%, 50%, 75%, 100%}` (including the 0% sanity floor and 15%); the run used `{5%, 10%, 25%, 50%, 75%, 100%}`. Minor, but the 0% floor is a specified sanity check that's missing.

**6. "Near" is operationalized as a proxy, worth flagging.** The far/near split is a median-distance split *within the already-non-local routable pool*, not "within/just outside Tier 1" as the README's quadrant table literally describes. It's a defensible choice (Tier 1 already owns true-near), but it's a deviation from the literal spec and should be stated as such in any writeup.

## Bottom line

The one thing the whole proposal hinges on — **does the low-frequency post-RoPE signal beat Pre-RoPE specifically in Q4** — has a real, non-circular, sensitivity-checked positive result across 3 layers. That's the load-bearing finding, and it's solid as far as it goes.

What's not yet proven is everything the README frames as required *before* this becomes a publishable/actionable result: the mandatory outlier sensitivity diagnostic (Section 4.5), the cost side of the recall-vs-cost tradeoff, and scale (more layers, more documents, longer contexts, downstream task numbers). Right now this is "the mechanism is real at small scale," not yet "the mechanism is real and worth the systems investment."
