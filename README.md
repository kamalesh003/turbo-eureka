# turbo-eureka

# DRAA v2 — Decentralized Retrieval-Augmented Adaptation (Resolved)

### A Practically Buildable Revision of the Original DRAA Proposal

This is not a rewrite for polish. Every change below exists because a specific flaw in the original made the system either impossible to implement or mathematically inconsistent. Where the original made an unproven claim, this version replaces it with a mechanism built from techniques that are independently demonstrated to work, and states plainly where research risk remains.

---

## 1. The central fix: replace "generate from scratch" with "retrieve, then compose"

**Original flaw:** A domain hypernetwork `H_i` was expected to map a single sentence-embedding `z` (a few hundred to ~1500 floats) directly to a full LoRA delta `ΔW_i` (potentially millions of parameters) for an arbitrary unseen query. This mapping is underdetermined — there is no published result showing a hypernetwork can generate a competent task-specific LoRA from an embedding alone, for open-domain queries it has never seen. This was the load-bearing assumption of the entire system, and it doesn't hold.

**Fix:** Stop generating weights from nothing. Generate them by **composing a small set of pre-trained anchor LoRAs**, selected and weighted by the hypernetwork. This is a much better-posed problem, and it's grounded in techniques that already work:

- **LoRA composition / task arithmetic** — combining multiple pre-trained LoRAs via learned or heuristic weights is demonstrated to work (LoraHub-style composition, TIES-merging, task-vector arithmetic). The hypernetwork's job shrinks from "invent ΔW" to "pick weights over a known basis," which is a much smaller and better-conditioned function to learn.
- Each peer maintains a bank of **N anchor LoRAs** (tens to low hundreds, not thousands) trained offline on curated sub-tasks within its domain — e.g. a Medical peer might hold anchors for "radiology report interpretation," "differential diagnosis," "drug interaction," etc.
- At query time, the peer's hypernetwork `H_i` takes `z` (the query embedding) and outputs a **coefficient vector** `β ∈ R^N` over its own anchor bank, not a full weight tensor:

```
β_i = H_i(z)              # small MLP, output dim = N (tens), not millions
ΔW_i = Σ_k β_i[k] · A_i[k] # weighted sum of pre-trained anchor LoRAs
```

This is orders of magnitude smaller as a learning problem, reuses proven merging math, and is falsifiable in a small offline experiment before any distributed infrastructure is built (see §9).

- **Storage cost this reintroduces:** each peer now stores N anchor LoRAs locally instead of zero. This is the honest tradeoff — the original's "no adapter storage" claim was only achievable because the generation step was unrealistic. N × (LoRA size, typically 1–10MB) is cheap local storage, not a scalability problem.

---

## 2. Fix the base-model synchronization gap

**Original flaw:** §2 and §11 apply every peer's LoRA to "the frozen foundation model," while §15/§16 claim peers can join and leave with "no synchronization required." These are inconsistent — fusing adapters across peers only works if all peers share the identical base model (same weights, same architecture, same layer names).

**Fix:** State the precondition explicitly instead of hiding it.

- The network maintains a small **Base Model Registry**: a signed manifest of `{model_id, architecture_version, weight_hash}` that all participating peers must match to be routable together.
- A peer's capability advertisement includes its base-model hash. The fusion router only merges adapters from peers matching the requester's base model.
- Peers on a different base model version form a **separate routing partition** — this is normal, not an error state, and is the honest version of "no synchronization required": no *continuous* synchronization is needed, but a *one-time compatibility contract* is.

---

## 3. Fix the fusion math / partial-adapter dimension mismatch

**Original flaw:** §9 describes peers generating adapters for *different layers* (medical → "clinical reasoning layers," vision → "visual encoder adaptation"), while §20's fusion formula (`ΔW = Σ α_i ΔW_i`) is a scalar-weighted sum, which is only valid if every `ΔW_i` has the *same shape and targets the same layers*. As written, the architecture contradicts itself.

**Fix:** Define a **canonical LoRA schema** shared network-wide, and route fusion at the correct granularity.

- Fix a shared target-module list up front (e.g., `{q_proj, k_proj, v_proj, o_proj, mlp.up, mlp.down}` for each transformer block, with a fixed rank `r` per module) — this is the standard practice in real LoRA merging work and is what makes `Σ α_i ΔW_i` well-defined at all.
- Peers whose specialty genuinely concentrates in different modules (e.g. a Vision peer only meaningfully updates the visual encoder’s projection layers) simply produce **near-zero coefficients for irrelevant modules** rather than being architecturally restricted to a module subset. This turns "different peers touch different layers" from an architectural inconsistency into an emergent, learned sparsity pattern — which is compatible with the shared-schema summation.
- Fusion becomes **two-level**, matching how the peers actually differ:
  1. **Intra-peer**: `ΔW_i = Σ_k β_i[k]·A_i[k]` (composition over that peer's own anchor bank, §1).
  2. **Inter-peer**: `ΔW = Σ_i α_i(z)·ΔW_i` (weighting across peers, as in the original), now valid because every `ΔW_i` shares the canonical schema.
- `α_i(z)` (inter-peer weight) is produced by a lightweight, separately-trained **Fusion Router** — a small model taking `z` and the set of matched peers' capability vectors, outputting a softmax over peers. This can be trained via distillation from held-out queries with known best-peer labels, or via a bandit/contextual-routing objective online.

---

## 4. Fix gossip staleness and give discovery a deterministic fallback

**Original flaw:** §6–7 describe gossip-propagated capability vectors with no fanout, TTL, convergence bound, or partition handling. Under gossip alone, a peer whose capability just changed can be invisible to distant queriers for an unbounded time, and there's no way to reason about "the query definitely reached the best-matched peer."

**Fix:** Use gossip for cheap approximate discovery, backed by a structured layer for guarantees.

- **Capability vectors are versioned** (`{peer_id, vector, version, ttl, signature}`). Anti-entropy gossip (periodic full-state reconciliation between random peer pairs, not just push) bounds staleness to `O(log N)` rounds with high probability — standard epidemic-protocol result; the design must pick and state explicit fanout (e.g. 3–6 peers/round) and round interval.
- For queries where "best possible match" matters more than latency, back gossip with a **Kademlia-style DHT** keyed by locality-sensitive hashes (LSH) of the capability vector. This gives `O(log N)` deterministic lookup instead of relying purely on epidemic spread, at the cost of standard DHT maintenance overhead. Gossip stays as the cheap/default path; DHT lookup is the fallback when gossip's approximate answer set looks thin or the query is high-value.
- Expired (TTL-lapsed) capability entries are dropped locally — this is what actually implements §16's "no global coordination needed on leave," rather than an unstated assumption.

---

## 5. Fix the security model — Sybil resistance and adapter integrity

**Original flaw:** §17 asserts a "trust score" with no computation mechanism, no Sybil defense, and no check that a peer's returned `ΔW_i` is actually beneficial rather than adversarial or low-quality.

**Fix, as three separate concrete mechanisms:**

- **Sybil resistance:** peer identity requires either (a) proof-of-stake — a bonded deposit slashed on detected misbehavior, or (b) a rate-limited, resource-costly join process (proof-of-work or verified external identity). Pure open self-registration of "capability vectors" with no cost is Sybil-able by construction; the original design has no answer to an attacker spinning up 1,000 fake "medical experts."
- **Capability verification, not self-report:** new/updated capability claims are spot-checked with **known-answer canary queries** drawn from a held-out benchmark set per domain. A peer's advertised capability vector is only trusted commensurate with its measured accuracy on canaries, not its self-description. This directly replaces the "trust score gradually decreases" hand-wave with a measurable quantity.
- **Adapter-output integrity:** returned `ΔW_i` (or `β_i`, the smaller coefficient vector under the v2 design) is signed by the peer. The fusion router additionally runs a lightweight **sanity check pass** — e.g., verify the fused model's output on the actual query doesn't diverge pathologically from the frozen base model's output (a cheap perplexity/consistency check) before returning to the user — to catch a compromised or adversarial peer without needing a full trust model to prevent every failure mode.

---

## 6. Fix the cost model — the real bottleneck is inference-time compute, not bandwidth

**Original flaw:** §18 compares "capability vector (KB)" against "sending a 7B model," which isn't the real cost tradeoff. The recurring cost is running `H_i` (or now, the much cheaper coefficient-generator) and the merge operation, per query, on every matched peer.

**Fix — make the cost model explicit and design against it:**

- Under the v2 design, `H_i` outputs a length-N coefficient vector via a small MLP — this is cheap (sub-millisecond, negligible compared to a single foundation-model forward pass), which is *why* §1's fix matters for cost, not just correctness.
- The dominant cost is the **merge-and-inference** step (applying `ΔW` and running the foundation model forward pass), which happens once per query regardless of how many peers contributed — this cost is the same order as ordinary single-adapter LoRA inference, not multiplied by the number of matched peers.
- **Adapter caching (original §12) is kept and is now cheap to key correctly**: cache key = `(base_model_hash, LSH-bucket(z))`, value = the fused `ΔW`. Cache hit skips both the per-peer coefficient generation and the fusion step entirely.

---

## 7. Fix continual learning — name the actual mechanism

**Original flaw:** §14 asserts "orthogonal memory tracking" without specifying what space the projection happens in, or how it applies to a hypernetwork rather than a directly-trained model.

**Fix:** Apply **Gradient Projection Memory (GPM)** — a documented continual-learning technique — at the correct point in the v2 architecture:

- Each peer stores a low-rank basis of the **gradient subspace** used by past updates to its anchor-selection network (the small `H_i` MLP from §1), not the foundation model.
- When `H_i` is updated after new local training, new gradients are projected to be orthogonal to this stored basis before being applied, which is the standard GPM mechanism for reducing catastrophic forgetting in a small trained network — and it's tractable here specifically *because* `H_i` is now small (§1's fix), where doing this against a full weight-generation hypernetwork would have been computationally impractical.
- The **capability vector** update is a downstream summary of this process (e.g., derived from `H_i`'s current parameters plus recent canary performance), not an independently-invented "orthogonal projection of capability vectors" as the original vaguely implied.

---

## 8. Revised end-to-end pipeline

```
User Query
    │
    ▼
Semantic Instruction Encoder (Sentence Transformer) → z
    │
    ▼
Adapter Cache Lookup (base_model_hash, LSH-bucket(z))
    │
    ├── HIT → return cached ΔW → skip to Inference
    │
    ▼ MISS
Distributed Discovery (Gossip directory + DHT fallback)
    │
    ▼
Matched Peers (only peers on matching Base Model Registry entry)
    │
    ▼
Per-Peer: β_i = H_i(z)              [small MLP → coefficients over anchor bank]
Per-Peer: ΔW_i = Σ_k β_i[k]·A_i[k]  [compose from pre-trained anchors, canonical schema]
    │
    ▼
Fusion Router: α_i(z) over matched peers
    │
    ▼
ΔW = Σ_i α_i(z)·ΔW_i   [valid: all ΔW_i share canonical module/rank schema]
    │
    ▼
Integrity check (consistency probe vs. frozen base model)
    │
    ▼
Inject ΔW into Frozen Foundation Model → Inference → Response
    │
    ▼
Cache ΔW at (base_model_hash, LSH-bucket(z))
    │
    ▼
Feedback → GPM-constrained update of H_i → canary re-verification → versioned capability-vector gossip update
```

---

## 9. What's still genuinely unproven, and how to test it before building anything else

Be precise about remaining research risk instead of implying the whole thing is settled:

1. **Does anchor-composition beat single-anchor retrieval?** i.e., does learning `β_i` over N anchors actually outperform simply retrieving the single closest-matching anchor LoRA by embedding similarity? This is the one experiment that validates or kills the hypernetwork component. Run it offline, single-peer, before building any distributed layer: take one domain, a fixed anchor bank, hold out queries, compare (a) nearest-anchor retrieval, (b) learned-coefficient composition, (c) a trained-from-scratch full LoRA per query (upper bound, infeasible in production but useful as a ceiling). If (b) doesn't measurably beat (a), the hypernetwork adds complexity without benefit and the system should just do retrieval.
2. **Fusion router quality** — whether `α_i(z)` learned from limited routing feedback actually beats a simpler heuristic (e.g., cosine similarity between `z` and each peer's capability vector, softmax-normalized) needs its own small-scale test before justifying a separately trained router.
3. **Canary-based trust** assumes a maintained, hard-to-game benchmark set per domain exists or can be built — this is an ongoing content-curation cost, not a one-time engineering cost, and should be budgeted as such.

---

## 10. Summary of what changed and why

| Component | Original | v2 Fix | Reason |
|---|---|---|---|
| Adapter generation | Full ΔW from embedding alone | Learned coefficients over pre-trained anchor bank | Original mapping is underdetermined / unproven at scale |
| Base model handling | Implied fully independent peers | Explicit Base Model Registry, hash-matched routing | Fusion requires identical base weights; original was inconsistent |
| Fusion math | Scalar sum over free-form partial adapters | Canonical module/rank schema + two-level (intra-peer, inter-peer) fusion | Original math only valid if shapes match; original text implied they wouldn't |
| Discovery | Pure gossip, no bounds | Gossip + DHT fallback, versioned/TTL'd entries | No convergence or staleness guarantees otherwise |
| Security | "Trust score," unspecified | Stake/cost-based Sybil resistance + canary verification + signed, sanity-checked adapters | Original had no defense against fake capability claims |
| Cost model | KB vs. GB bandwidth comparison | Explicit accounting of per-query inference/merge cost, cache-key design | Real bottleneck is compute, not bandwidth |
| Continual learning | "Orthogonal memory tracking," unspecified | GPM on the small coefficient-generator network specifically | Original didn't specify space or target; only tractable on a small network |

The system as revised is buildable with existing, demonstrated components (LoRA composition/merging, epidemic + DHT discovery, GPM continual learning, stake-based Sybil resistance) rather than resting on a single unproven generative capability. The one remaining open research question — whether learned anchor-composition beats simple retrieval — is now isolated, cheap to test, and doesn't require any distributed infrastructure to answer.
