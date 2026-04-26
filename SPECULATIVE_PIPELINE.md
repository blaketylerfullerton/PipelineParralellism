# Speculative Pipelining for WAN Distributed LLM Inference

## What It Is

Speculative pipelining is an optimization that overlaps two normally sequential operations: **draft token generation** and **network round-trip verification**. In the current pipeline, Stage 0 goes idle the moment it fires its activation message across the WAN — it just waits. Speculative pipelining fills that idle window with useful work.

It is distinct from speculative decoding (which is already implemented). Speculative decoding reduces the number of network round-trips by batching K draft tokens into one. Speculative pipelining reduces the cost of each individual round-trip by hiding the draft generation time inside the network wait.

Think of it as two levels of speculation stacked:

- **Level 1 (already done):** Speculative decoding — send K tokens per round-trip instead of 1.
- **Level 2 (this document):** Speculative pipelining — pre-generate draft tokens for round N+1 while round N is in flight.

---

## The Current Timeline (No Pipelining)

For each speculative decode round, Stage 0 executes these steps sequentially:

```
┌──────────────────┐
│ draft_k_tokens() │  ~10–100ms depending on draft model size
└────────┬─────────┘
         │
┌────────▼──────────────┐
│ stage_model forward() │  Stage 0's target model layers — ~20–50ms
└────────┬──────────────┘
         │
┌────────▼──────────┐
│ send_msg (ZMQ)    │  ~1ms + serialization
└────────┬──────────┘
         │
┌────────▼──────────────────────────────────────────────────────┐
│ recv_msg (blocking wait)                                       │
│                                                                │
│   ← WAN transit (~40ms) + Stage N compute + WAN return (~40ms)│
│                                                                │
│   Stage 0 is completely idle here                              │
└────────┬──────────────────────────────────────────────────────┘
         │
┌────────▼──────────────┐
│ accept / reject logic │  <1ms
└───────────────────────┘
```

Stage 0 is idle for the entire `recv_msg` window — roughly 80ms of WAN round-trip plus Stage N's compute time. The draft step that precedes the next round runs only after the response arrives, adding its full cost to the critical path.

**Total wall-clock time per round:**
```
draft + stage0_forward + WAN_RTT + stageN_compute + accept/reject
```

---

## The Pipelined Timeline

With speculative pipelining, Stage 0 launches the next draft in a background thread the moment it fires `send_msg`. The GIL is released while the main thread blocks on ZMQ's C-level `poll()` call, so PyTorch can execute on the background thread in parallel.

```
┌──────────────────┐
│ draft_k_tokens() │  (round N)
└────────┬─────────┘
         │
┌────────▼──────────────┐
│ stage_model forward() │
└────────┬──────────────┘
         │
┌────────▼──────────┐
│ send_msg (ZMQ)    │
└────────┬──────────┘
         │                    ┌──────────────────────────────────┐
         │                    │  background thread               │
         │                    │  draft_k_tokens() for round N+1  │
         │                    │  (using speculative seed)        │
         │                    └──────────────────────────────────┘
┌────────▼──────────────────────────────────────────────────────┐
│ recv_msg (blocking wait)                                       │
│   WAN transit + Stage N compute + WAN return                  │
└────────┬──────────────────────────────────────────────────────┘
         │
    join background thread (usually already done)
         │
┌────────▼──────────────┐
│ accept / reject logic │
└────────┬──────────────┘
         │
    if prefetch seed matched: use pre-drafted tokens → skip draft next round
    if mismatch: discard prefetch → re-draft normally (no regression)
```

**Total wall-clock time per round (on cache hit):**
```
draft + stage0_forward + WAN_RTT + stageN_compute + accept/reject
                                   ↑
                        draft for round N+1 hides inside this window
```

On a hit, `draft` disappears from the critical path for round N+1.

---

## The Seed Problem

This is the central complication. To pre-draft round N+1 during round N's WAN wait, you need to know what token to start drafting from — the "seed." But that seed is determined by the acceptance outcome of round N, which only arrives with the Stage N response.

The possible outcomes of round N's accept/reject:
- **M < K tokens accepted:** next seed = a correction token sampled by the target model (unknown in advance)
- **M = K+1 tokens accepted (all + bonus):** next seed = a bonus token sampled by the target model (also unknown)

In every case, the actual next seed is a function of target model logits you haven't seen yet.

The practical approach is to assume an **optimistic seed** — the last draft token generated in round N (`draft_tokens_t[0, K-1]`). This guess is correct when the acceptance cascade aligns with the assumption. When it is wrong, the pre-drafted tokens are discarded and Stage 0 re-drafts from the correct seed after the response arrives. The worst case is identical to today's behavior.

### Hit Rate Expectations

With a well-calibrated draft model and per-token acceptance rate of ~0.75:

| K (draft length) | Estimated hit rate |
|---|---|
| 4 | ~20–30% |
| 5 | ~15–25% |
| 8 | ~10–20% |

Hit rate decreases with larger K because the optimistic seed assumption becomes less likely to hold over longer chains. However, larger K also means longer draft time — which is exactly what you most want to hide. The two effects partially cancel in a useful way.

---

## Why This Matters More for This Project Than Most

Most speculative pipelining discussions assume a single machine where draft runs on CPU and verify runs on GPU. In that case the draft is fast (CPU) and verify is fast (GPU), so the overlap window is narrow.

This project runs across WAN with ~40ms each way — 80ms+ of round-trip before any compute is counted. That dead time dwarfs what you see in single-machine setups. Any work that fits inside that window is effectively free.

### Scale of the win by draft model size

| Draft model | Draft time (K=5, CPU) | WAN RTT | Draft as % of round | Expected speedup on hit |
|---|---|---|---|---|
| GPT-2 small (124M) | ~10ms | ~80ms | ~7% | ~2% |
| Llama-3.2-1B | ~50–80ms | ~80ms | ~30% | ~8–12% |
| Llama-3.2-3B (as draft) | ~120–200ms | ~80ms | ~50%+ | ~15–25% |

For the current config using `unsloth/Llama-3.2-1B` as the draft model, speculative pipelining is a meaningful optimization because the draft model's forward pass is a substantial fraction of the total round-trip time.

---

## Interaction with Existing Optimizations

### KV Cache (Phase 1.1)
No conflict. The pre-draft uses a snapshot of the draft model's KV cache, leaving the main KV untouched. On a miss the snapshot is discarded. On a hit the snapshot's extended KV is adopted and trimmed to the correct sequence length.

### int8 Activation Compression (Phase 1.2)
No interaction. The pre-draft runs entirely locally; compression only affects the wire payload sent to Stage N.

### Existing Speculative Decoding (Phase 2)
Speculative pipelining is additive on top of speculative decoding, not a replacement. Spec decoding reduces round-trips per token. Spec pipelining reduces the marginal cost of each round-trip. They compose independently.

### MoE Routing (Phase 3)
Speculative pipelining becomes more valuable in the MoE phase because expert routing introduces additional latency hops. Each additional hop means more idle time for Stage 0 to fill. The same threading approach applies; the seed problem is identical.

---

## Implementation Sketch

Two files change:

**`draft.py`**: Add `start_prefetch(seed, k)` which snapshots the KV cache and launches a daemon thread running `draft_k_tokens` on the speculative seed. Add `consume_prefetch(actual_seed_id, assumed_seed_id)` which joins the thread and returns the result if the seed matched, or `None` on mismatch.

**`worker.py`** (Stage 0 speculative decode branch): After `send_msg`, call `draft_model.start_prefetch(draft_tokens_t[:, -1:], k)`. Block on `recv_msg` as normal. After accept/reject, call `consume_prefetch` and either adopt the pre-draft for the next iteration or discard it.

The background thread uses a cloned KV, so there is no shared mutable state between the thread and the main thread's accept/reject logic. The join happens before any KV manipulation.

A `pipeline_hits / pipeline_steps` counter should be added to the end-of-generation summary to measure real-world effectiveness.

---

## Projected Outcomes

### Conservative (GPT-2 small draft, 2-machine WAN at 40ms RTT)
- Hit rate: ~25%
- Draft time saved per hit: ~10ms
- Per-round savings: 0.25 × 10ms = 2.5ms on a ~150ms round
- **TPS improvement: ~2%**

This is a small but free win for the current GPT-2 config.

### Target config (Llama-3.2-1B draft, 2-machine WAN at 40ms RTT)
- Hit rate: ~25%
- Draft time saved per hit: ~60ms
- Per-round savings: 0.25 × 60ms = 15ms on a ~250ms round
- **TPS improvement: ~6%**

### Optimistic (Llama-3.2-1B draft, high acceptance rate ~0.85)
- Hit rate: ~35%
- Draft time saved per hit: ~60ms
- Per-round savings: 0.35 × 60ms = 21ms on a ~250ms round
- **TPS improvement: ~8–10%**

### With tree speculation upgrade (see below)
- Hit rate: ~55–65%
- **TPS improvement: ~15–20%**

---

## Upgrade Path: Token Tree Speculation

The seed problem can be partially solved by drafting a **small tree** of candidates during the WAN wait instead of a single chain. For example:

- Draft 2 tokens from seed `draft[K-1]` (all-accepted assumption)
- Draft 2 tokens from seed `draft[K-2]` (rejected-at-last assumption)
- Draft 2 tokens from seed `last_tok` (all-rejected assumption)

The correct branch of the tree is selected once the acceptance outcome is known. This raises hit rate from ~25% to ~55–65% at the cost of running 3× more draft inference during the wait window — but since the window is the WAN RTT, the extra compute is mostly free.

This is architecturally similar to EAGLE-2 and Medusa, adapted for the pipeline-parallel WAN setting. It is a natural next step after validating the single-seed implementation.

---

## Summary

| Property | Value |
|---|---|
| Implementation difficulty | Medium (threading + KV snapshot logic) |
| Risk of regression | None (miss path is identical to today) |
| Files changed | `draft.py`, `worker.py` |
| Benefit for GPT-2 config | ~2% TPS |
| Benefit for Llama-1B config | ~6–10% TPS |
| Benefit with tree upgrade | ~15–20% TPS |
| Stacks with existing optimizations | Yes, fully additive |
| Priority vs Phase 3 MoE | Lower — MoE is a larger multiplier |

Speculative pipelining is a low-risk, additive optimization that becomes increasingly valuable as the draft model grows and as WAN latency increases. It is not the highest-leverage item on the roadmap (Phase 3 MoE is), but it costs nothing on a miss and compounds with every other optimization already in place.
