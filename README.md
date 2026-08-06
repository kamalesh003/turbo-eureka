# turbo-eureka

# DRAA v2

## Decentralized Retrieval-Augmented Adaptation via Anchor-Composed LoRA Routing

*A fully specified, implementable revision of the original DRAA concept.*

---

# 1. Motivation

| Existing Paradigm | Limitation |
|---|---|
| RAG | Retrieves knowledge (documents), not expertise. |
| LoRA | Requires storing thousands of adapters, one per task. |
| Hypernetworks (weight generation) | No published method reliably generates a competent full LoRA from a single query embedding for unseen open-domain tasks. |
| Federated Learning | Requires aggregation servers and synchronized rounds. |
| Mixture-of-Experts | Experts are fixed and live inside a single model. |

**Core idea:** retrieve distributed *expertise*, not documents or full weights — but do it by **composing pre-trained, verified LoRA anchors** using a small learned router, not by generating weights from nothing. Composition over a known basis is a well-posed problem with existing proof points (LoRA merging, task arithmetic); blind weight generation from an embedding is not. Everything in this design is built to make that one substitution work end to end.

---

# 2. High-Level Architecture

```text
                       User Query
                            │
                            ▼
              Semantic Instruction Encoder
                   (Sentence Transformer)
                            │
                            ▼
                  Query Embedding z
                            │
                            ▼
                  Adapter Cache Lookup
              (base_model_hash, LSH-bucket(z))
                    │              │
                  HIT             MISS
                    │              │
                    │              ▼
                    │   Distributed Capability Discovery
                    │      (Gossip directory + DHT fallback)
                    │              │
                    │   ┌──────────┼──────────┐
                    │   ▼          ▼          ▼
                    │ Medical    Vision   Statistics
                    │  Peer       Peer       Peer
                    │   │          │          │
                    │   ▼          ▼          ▼
                    │ Coefficient Router (H_i, per peer)
                    │   │          │          │
                    │   ▼          ▼          ▼
                    │ Anchor Composition (ΔW_i from local bank)
                    │   └──────────┼──────────┘
                    │              ▼
                    │      Fusion Router (α_i(z))
                    │              │
                    │              ▼
                    │      Merged Adapter ΔW
                    │              │
                    └──────────────┤
                                   ▼
                        Integrity / Consistency Check
                                   │
                                   ▼
                      Frozen Foundation Model + ΔW
                                   │
                                   ▼
                              Final Response
                                   │
                                   ▼
                    Cache write + feedback-driven update
```

---

# 3. Peer Architecture

```text
Peer
├── Local Dataset
├── Foundation Model (frozen, version-pinned)
├── Anchor Bank             { A_i[1] ... A_i[N] }   — pre-trained LoRAs, canonical schema
├── Coefficient Router H_i  — small MLP, z → β ∈ R^N
├── Capability Encoder      — derives capability vector c_i from H_i state + canary scores
├── Capability Index        — local ANN over known peer capability vectors
├── Neighbor Directory       (gossip peer list)
├── Adapter Cache           — keyed by (base_model_hash, LSH-bucket(z))
├── Gossip Manager          — versioned, TTL'd, signed capability broadcast
├── Local Trainer           — trains/updates H_i and anchor bank offline
├── GPM Memory              — stored gradient-subspace basis for H_i, for continual learning
├── Identity + Stake        — signing key, bonded deposit (Sybil resistance)
└── Canary Test Set         — held-out benchmark queries with known-good answers, per peer's domain
```

A peer never shares its dataset, raw gradients, or full model weights. It shares: its capability vector, its base-model hash, and — per query, on demand — a coefficient vector `β_i` (tens of floats) or the resulting composed `ΔW_i`, both signed.

---

# 4. Local Learning Phase

Each peer specializes independently, offline, with no cross-peer coordination.

```text
Local Domain Data
        │
        ▼
Train / curate Anchor Bank A_i[1..N]
   (each anchor: a competent LoRA for one well-scoped sub-task,
    fixed target-module schema, fixed rank r)
        │
        ▼
Train Coefficient Router H_i
   (input: query embeddings from labeled/successful task history
    output: coefficient vector β over the anchor bank
    objective: composed ΔW_i performs well on held-out queries in-domain)
        │
        ▼
Evaluate against Canary Test Set
        │
        ▼
Derive Capability Vector c_i
        │
        ▼
Gossip capability update (versioned, signed)
```

This replaces the original's "train a hypernetwork to emit full weights" with a training problem that has a bounded, well-defined output space (N coefficients), making convergence and evaluation tractable per peer.

---

# 5. Anchor Bank Design

- Each peer curates **N anchor LoRAs** (tens to low hundreds), each trained on a well-scoped sub-task within the peer's domain using standard LoRA fine-tuning.
- All anchors across the entire network share a **canonical target-module schema**: a fixed list of transformer submodules (e.g. `q_proj, k_proj, v_proj, o_proj, mlp.up, mlp.down` per block) and a fixed rank `r`. This is what makes composition and cross-peer fusion mathematically well-defined later.
- Anchors are versioned and hash-pinned to a specific **Base Model Registry** entry (§7); an anchor is only usable against the exact base model it was trained against.
- Anchor quality is periodically re-validated against the peer's own canary set; an anchor that regresses is flagged and excluded from composition until retrained.

---

# 6. Capability Representation

A peer's capability vector `c_i` is derived from:

- Embeddings of its local training/anchor corpus.
- Recent canary benchmark scores (measured, not self-reported).
- A summary statistic of `H_i`'s current parameters (e.g., a learned projection), so the vector reflects what the router *currently does*, not just what data it was trained on.
- Recency-weighted successful-query history.

```text
c_i = E( anchor_corpus_i , canary_scores_i , H_i_summary , task_history_i )
```

`c_i` is **not** self-declared text ("I know medicine") and is **not** trusted at face value by other peers — see §14 for how it's verified before being routed on.

---

# 7. Base Model Registry

**This is a hard precondition the original design omitted.** LoRA fusion across peers only works if every contributing peer's adapters target the identical base model.

```text
Registry Entry
{
  model_id
  architecture_version
  weight_hash          (signed by a trusted publisher / reproducible build hash)
}
```

- A peer's capability advertisement always includes its `base_model_hash`.
- The discovery layer (§9) and fusion router (§11) only ever match peers sharing the requester's `base_model_hash`.
- Peers on a different base model version simply form a separate routing partition — no error, no cross-partition fusion attempted.
- No continuous synchronization between peers is required; this is a **one-time compatibility contract**, checked at discovery time.

---

# 8. Gossip-Based Capability Exchange

Peers gossip only lightweight, versioned metadata — never datasets, anchors, or full weights.

```text
Capability Record
{
  peer_id
  capability_vector c_i
  base_model_hash
  version               (monotonic counter)
  ttl
  signature
}
```

Protocol:

- **Anti-entropy gossip**: periodic full-state reconciliation between randomly chosen peer pairs, not just push-only propagation. This bounds staleness to `O(log N)` rounds with high probability (standard epidemic-protocol result), given an explicit fanout (recommend 3–6 peers per round) and round interval.
- Records carry a **monotonic version**; a peer receiving an older version than it already has simply discards it.
- Records expire on TTL lapse and are dropped locally with no global coordination — this is what actually implements clean peer-leave semantics (§16).

---

# 9. Distributed Capability Discovery

```text
Query → z
    │
    ▼
Local Adapter Cache check (base_model_hash, LSH-bucket(z)) — fast path
    │  MISS
    ▼
ANN search over locally-known capability vectors (from gossip state)
    │
    ▼
If match set is thin / query is high-value:
    Fallback to Kademlia-style DHT lookup
    (keyed by LSH of the capability vector — deterministic O(log N) lookup,
     used as a guarantee layer on top of gossip's cheap approximate answer)
    │
    ▼
Matched Peers (filtered to matching base_model_hash)
```

Gossip is the default cheap path; the DHT exists specifically to give a bounded-latency, deterministic answer when the epidemic-propagated view is insufficient — the original design had no such fallback and no way to reason about whether the "best" peer was actually reachable.

---

# 10. Query-Conditioned Anchor Composition (replaces free-form hypernetwork generation)

Each matched peer computes a small coefficient vector, not a full weight tensor:

```text
z  →  H_i(z)  →  β_i ∈ R^N        (small MLP, output dim = N, tens not millions)

ΔW_i = Σ_k  β_i[k] · A_i[k]       (weighted sum over the peer's own pre-trained anchor bank)
```

This is the single most important structural change from the original proposal. `H_i` no longer has to invent a competent set of weights for an unseen task from a sentence embedding alone — a mapping with no demonstrated precedent — it only has to select and weight among a bounded, pre-validated basis. This is directly analogous to established LoRA-merging / task-arithmetic techniques, which are shown to work; the router's job is the well-posed part of the problem.

If a peer's domain doesn't meaningfully overlap the query, `β_i` is trained to be near-zero across the board — a peer simply contributes almost nothing rather than being excluded by hard architectural rules.

---

# 11. Two-Level Fusion

**Level 1 — intra-peer** (§10): compose `ΔW_i` from that peer's own anchor bank.

**Level 2 — inter-peer**: weight contributions across matched peers.

```text
ΔW = Σ_i  α_i(z) · ΔW_i
```

This sum is well-defined only because every `ΔW_i`, from every peer, shares the canonical target-module/rank schema fixed in §5 — this was the specific point where the original design's math and its own architectural description contradicted each other.

`α_i(z)` is produced by a separately trained **Fusion Router**: a small model taking `z` plus the capability vectors of the matched peer set, outputting a softmax over peers. It can be trained via distillation against queries with known best-peer labels, or refined online via a contextual-bandit routing objective using downstream feedback.

Example — medical query: `Medical 0.70, Vision 0.20, Statistics 0.10`.
Different query: `Vision 0.60, Medical 0.20, Programming 0.20`.

---

# 12. Integrity Check

Before returning a result, the fused adapter passes a lightweight sanity check:

```text
ΔW  →  run frozen base model + ΔW on the actual query
     →  compare against frozen base model alone
     →  flag pathological divergence (cheap perplexity/consistency probe)
```

This catches a corrupted or adversarial peer contribution without requiring a full trust model to prevent every possible failure mode. Each contributing peer's `β_i` (or resulting `ΔW_i`) is signed, so a flagged failure can be attributed and penalized (§14).

---

# 13. LoRA Injection

```text
Merged ΔW → injected into Frozen Foundation Model → Inference → Response
```

The base model itself is never modified; injection is standard LoRA application, discarded or cached after use.

---

# 14. Security Layer

Three separate, concrete mechanisms — not a single unspecified "trust score":

**Sybil resistance**
Peer identity requires either a bonded stake (slashed on detected misbehavior) or a rate-limited, resource-costly join process. Free, costless self-registration of capability claims is Sybil-able by construction and is not permitted.

**Capability verification**
New or updated capability claims are checked against **known-answer canary queries** drawn from a held-out per-domain benchmark. A peer's advertised capability vector is trusted in proportion to its *measured* canary accuracy, not its self-description. Canary sets are a maintained, ongoing content-curation cost — not a one-time build cost — and should be budgeted as such.

**Adapter integrity**
Every `β_i` / `ΔW_i` a peer returns is signed. The fusion router's integrity check (§12) provides a real-time defense; canary re-checks and stake slashing provide the longer-term deterrent.

---

# 15. Cost Model (explicit, not hand-waved)

- Gossip traffic: capability records, a few KB each — cheap, as in the original.
- Per-query per-peer compute: evaluating `H_i(z)` is a small MLP forward pass — sub-millisecond, negligible next to a foundation-model forward pass. This is a direct consequence of §10's redesign; the original's full-weight-generation hypernetwork would not have had this property.
- Dominant recurring cost: the merge-and-inference step, which happens **once per query regardless of how many peers contributed** — the same order of cost as ordinary single-adapter LoRA inference.
- **Adapter cache**, keyed by `(base_model_hash, LSH-bucket(z))`, storing the final fused `ΔW`. A cache hit skips per-peer coefficient generation, composition, and fusion entirely.
- Storage cost reintroduced versus the original: each peer stores its own anchor bank (N × 1–10MB LoRAs). This is the honest tradeoff for a composition mechanism that's actually falsifiable and works — the original's "zero adapter storage" claim was only achievable by assuming an unproven generation step.

---

# 16. Continual Learning

Applied specifically to `H_i` (the small coefficient router), where it's actually tractable — not to a full weight-generating hypernetwork, where it would not be:

```text
New local training signal
        │
        ▼
Compute gradient update for H_i
        │
        ▼
Project orthogonal to peer's stored GPM basis
   (Gradient Projection Memory — low-rank basis of past update subspaces)
        │
        ▼
Apply projected update to H_i
        │
        ▼
Re-validate against Canary Test Set
        │
        ▼
Recompute Capability Vector c_i
        │
        ▼
Gossip versioned capability update
```

This is Gradient Projection Memory, a documented continual-learning technique, applied to a genuinely small network — which is what makes it computationally tractable here.

---

# 17. Peer Join / Leave

**Join:** a new peer builds its anchor bank and `H_i` offline, passes canary evaluation, posts stake/identity proof, advertises its capability vector via gossip. No retraining or restart required elsewhere on the network.

**Leave:** the peer stops gossiping updates; its capability record's TTL lapses at neighboring peers, who drop it locally. No global coordination step is required — this is a direct consequence of the TTL/versioning design in §8, not an assumption.

---

# 18. End-to-End Workflow

```text
User Query
    │
    ▼
Semantic Instruction Encoder → z
    │
    ▼
Adapter Cache lookup (base_model_hash, LSH-bucket(z))
    │
    ├── HIT → cached ΔW
    │
    ▼ MISS
Discovery (Gossip + DHT fallback), filtered to matching base_model_hash
    │
    ▼
Per matched peer: β_i = H_i(z);  ΔW_i = Σ_k β_i[k]·A_i[k]
    │
    ▼
Fusion Router: α_i(z)  →  ΔW = Σ_i α_i(z)·ΔW_i
    │
    ▼
Integrity check vs. frozen base model
    │
    ▼
Inject ΔW → Foundation Model → Inference → Response
    │
    ▼
Cache write
    │
    ▼
Feedback → GPM-constrained H_i update → canary re-check → capability vector update → gossip
```

---

# 19. Mathematical Formulation

Anchor bank for peer `i`: `A_i = {A_i[1], ..., A_i[N]}`, each a LoRA delta on the canonical schema.

Coefficient generation:
```
β_i = H_i(z),   H_i: R^d → R^N   (small MLP)
```

Intra-peer composition:
```
ΔW_i = Σ_{k=1}^{N} β_i[k] · A_i[k]
```

Inter-peer fusion, over the matched peer set `P` (all sharing `base_model_hash`):
```
ΔW = Σ_{i ∈ P} α_i(z) · ΔW_i,      α_i(z) from the Fusion Router, Σ α_i = 1
```

Capability vector:
```
c_i = E(anchor_corpus_i, canary_scores_i, H_i_summary, task_history_i)
```

Discovery:
```
P = ANN(z, {c_i}) ∩ {i : base_model_hash_i = base_model_hash_query}
```

Inference:
```
y = F(x; ΔW),   F = frozen foundation model
```

Continual update (GPM), for peer `i`'s router:
```
g = ∇_θ L(H_i)
g_proj = g − proj_{basis_i}(g)      (orthogonal projection onto stored subspace's complement)
θ ← θ − η · g_proj
```

---

# 20. Core Contributions

1. **Anchor-Composed Query Routing** — LoRA synthesis reframed as learned composition over a bounded, pre-validated anchor basis rather than unconstrained weight generation from a query embedding, grounded in established LoRA-merging/task-arithmetic results.
2. **Base-Model-Aware Decentralized Fusion** — an explicit compatibility contract (Base Model Registry) and canonical target-module/rank schema that make cross-peer adapter summation mathematically well-defined.
3. **Two-Tier Gossip + DHT Discovery** — versioned, TTL'd epidemic propagation for cheap default routing, backed by a deterministic DHT fallback for bounded-latency guarantees.
4. **Measured, Not Self-Reported, Trust** — Sybil-resistant identity plus canary-benchmark-verified capability claims, replacing unspecified "trust scores" with a concrete, auditable mechanism.
5. **Tractable Continual Adaptation** — Gradient Projection Memory applied to a small coefficient-router network, keeping catastrophic-forgetting mitigation computationally practical.
6. **Falsifiable Core Claim** — the single load-bearing hypothesis (learned anchor composition beats simple nearest-anchor retrieval) is isolated into a cheap, single-peer, offline experiment that can validate or kill the approach before any distributed infrastructure is built.

---

# 21. Why This Version Is Buildable

Every mechanism in this design maps to a technique with independent evidence that it works: LoRA composition and merging, epidemic-protocol gossip with anti-entropy, Kademlia-style DHT lookup, stake-based Sybil resistance, canary-based capability verification, and Gradient Projection Memory for continual learning. The one genuinely open question — whether a learned coefficient router over an anchor bank measurably outperforms plain nearest-anchor retrieval — is now a small, well-defined, offline experiment (§9 of the prior review), not an assumption buried under several layers of distributed-systems infrastructure. Nothing else in the architecture depends on an unproven generative capability.
