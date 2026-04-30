# Distributed Inference — Latency Mitigation Plan

Goal: make pipeline parallelism across WAN-connected CPU machines (WireGuard, different ISPs) actually usable. Starting point was the GPT-2 pipeline in this repo — dense, no KV cache, raw-bytes ZMQ, fully sequential round-trip per token. Current state is Llama-3.2-3B with KV cache, int8 wire compression, speculative decoding (k=6, ~77% acceptance), and cascade fast-path (~29% local steps).

Realistic target after the full stack: **3–8 TPS for a 70B-equivalent model on 2–3 CPU machines across different internets.** Interactive-ish, not snappy. Good enough for agent workflows.

**Current measured baseline (2× s-8vcpu-16gb DO droplets, same region, WireGuard, Llama 3.2-3B bf16):**
- **1.10 TPS** end-to-end
- Draft (Llama 1B, k=6): ~1,200 ms per round
- Stage 0 forward: ~650 ms per round
- Stage 1 forward + wait: ~3,200 ms per round (network RTT negligible — same region)
- Speculative acceptance rate: **77.1%** (excellent)
- Cascade hit rate: **29%** (2/7 steps locally)
- **Bottleneck: CPU memory bandwidth.** Network is not the constraint.

---

## Bottleneck analysis — what's actually slow now

Per spec round, the wall-clock breakdown:

| Component | ms | % of round |
|---|---|---|
| Draft (1B, k=6) | ~1,200 | 24% |
| Stage 0 forward (verifier) | ~650 | 13% |
| Stage 1 forward + wait | ~3,200 | 63% |
| WAN RTT | ~80 | <2% |

The original plan was written assuming WAN was the enemy. **It isn't.** Llama-3.2-3B in bf16 is 6 GB total → ~3 GB/stage of weights that have to be streamed from RAM through the CPU caches every decode step. On an 8-vCPU droplet with ~30–40 GB/s memory bandwidth, just touching the weights costs ~75–100 ms; with attention, KV reads, and a 7-token verification batch, ~3,200 ms is roughly what the hardware is capable of.

This re-orders the priority of every remaining item in this plan. The two highest-leverage levers from here are:

1. **Hide the 1,200 ms draft inside the 3,200 ms verifier wait** — speculative pipelining (Phase 2.5).
2. **Reduce the bytes streamed per token** — weight quantization (Phase 1.6, *new*).

MoE (Phase 3) is *not* obviously the next move on this hardware: see the dedicated section for the honest tradeoff.

---

## Phase 1 — Free wins (DONE)

### ~~1.1 KV cache~~ ✅ DONE

**What we did:** Updated `model.py` to use HuggingFace `DynamicCache` (transformers 5.x API) in all three stage modules (`Stage0Module`, `MiddleModule`, `LastModule`). Each block is called with `past_key_values=cache, use_cache=True` — the cache mutates in place and accumulates K/V tensors across steps. `Stage0Module` offsets positional IDs by `cache.get_seq_length()` so decode steps land at the correct position. Added `is_prefill` flag to activation messages in `utils.py` so downstream stages know when to reset their cache. In `worker.py`, Stage 0 feeds the full sequence on step 0 then only `generated[:, -1:]` (1 token) on every subsequent step; all other stages maintain their own `DynamicCache` and reset it on `is_prefill=True`.

**Result:** Hidden state transmitted over the wire drops from `(1, N, 768)` to `(1, 1, 768)` after the first step — ~30x smaller payload on a 30-token generation. Compute drops from O(N²) to O(N) amortized.

| | Before | After |
|---|---|---|
| ms / token | 192 ms | 112 ms |
| speedup | — | 1.7× |

**Before (no KV cache):**

![Before KV cache](images/before_kv.png)

**After (KV cache):**

![After KV cache](images/after_kv.png)

### ~~1.2 Activation compression on the wire~~ ✅ DONE

**What we did:** Added `encode_activation(tensor, mode)` / `decode_activation(encoded)` / `tensor_from_activation_msg(msg)` to `utils.py`. Three modes — `fp32` (baseline), `fp16` (2x), `int8` (4x, per-tensor scale). `make_activation_msg` now accepts a `codec` param. `worker.py` reads `compression.mode` from `config.yaml` and passes it through. All hidden-state sends use the configured codec; the token-return channel (single int) stays fp32. `config.yaml` defaults to `int8`.

**Measured round-trip error (GPT-2 medium, hidden=768):**

| codec | bytes/token | max error |
|---|---|---|
| fp32 | 3072 | 0 |
| fp16 | 1536 (2×) | 0.001 |
| int8 | 768 (4×) | 0.013 |

![After Adding Compression](images/after_compression.png)

> **Note (post-bottleneck-shift):** Wire compression is now near-zero leverage — RTT is <2% of the round. Keep `int8` set for the WAN/multi-ISP scenario, but don't expect further gains here.

### ~~1.2.5 Switch to Llama 3.2-3B~~ ✅ DONE

**What we did:** Added `Stage0Llama`, `MiddleLlama`, `LastLlama` to `model.py`. `_run_llama_layers` replicates the `LlamaModel.forward` per-layer loop for a stage subset, computing RoPE position embeddings and causal masks locally. `_relabel_layer_idx` resets each sliced layer's `self_attn.layer_idx` so per-stage `DynamicCache` stays dense. `get_stage` / `get_tokenizer` dispatch on `config.model.arch` (`gpt2` vs `llama`). After slicing the stage, the full model is deleted and `gc.collect()` is called to free the unused weight slice. `config.yaml` now points at `unsloth/Llama-3.2-3B` (target) and `unsloth/Llama-3.2-1B` (draft) with `dtype: bfloat16`.

### ~~1.3 Overlap send with compute (partial — buffer tuning only)~~ ✅ DONE

**What we did:** Raised `ZMQ_SNDBUF`/`ZMQ_RCVBUF` to 4 MB on all push/pull sockets in `utils.py`. A background-thread `AsyncSender` was attempted but ZMQ sockets are not thread-safe — using a socket from any thread other than the one that created it caused a 5× TPS regression. The proper async approach is `asyncio` + `zmq.asyncio`, deferred to Phase 3 where it's needed for MoE scatter-gather routing anyway.

### 1.4 Network-layer tuning (deprioritized)

WAN is currently <2% of the round budget. Deferred until ISP-diverse testing (M5) shows network actually matters again.

---

## Phase 1.6 — Weight quantization *(NEW — highest-leverage open item)*

**Why this exists:** the original plan only covered *activation* (wire) compression. With the bottleneck now firmly inside RAM-to-CPU bandwidth, **reducing the bytes-per-weight is the single biggest available win.** Llama-3.2-3B at:

| precision | bytes / param | size / stage | est. Stage 1 ms | est. TPS |
|---|---|---|---|---|
| bf16 (current) | 2 | ~3 GB | 3,200 | 1.10 |
| int8 | 1 | ~1.5 GB | ~1,800 | ~1.7 |
| Q4_K_M (GGUF) | ~0.5 | ~750 MB | ~1,000 | ~2.3 |

These are upper bounds — they assume bandwidth is the only thing happening. Real numbers depend on whether the kernel implementations are any good at int8/Q4 on CPU. **That kernel quality question is the trigger for the Path A vs Path B decision below.**

### Concrete changes

- Try `torch.ao.quantization.quantize_dynamic` int8 dynamic quant on the stage Linear layers first — cheapest experiment, no infra change. If HF Linear int8 CPU kernels are mediocre (likely), the gain will be sub-2× and that's the signal.
- If torch dynamic quant disappoints: the right move is GGUF quantization, which means leaving HuggingFace altogether (see Path B below).
- New `quantization.weight_mode` field in `config.yaml`: `bf16 | int8 | q4`.

### Sequencing

Do this **after** Phase 2.5 speculative pipelining, not before — see the timing-coupling note in 2.5 for why the order matters.

---

## ~~Phase 2 — Speculative decoding~~ ✅ DONE

**Concept:** run a tiny draft model (Llama-3.2-1B) locally on Stage 0. Draft K=6 tokens fast without touching the network. Send all candidates through the pipeline in one forward. The big model emits logits for all positions in one shot — accept the longest matching prefix using standard speculative sampling (Leviathan et al. 2023, Algorithm 1), reject the rest.

**What we did:** `draft.py` with `DraftModel.draft_peek` / `draft_k_tokens` / `advance_cache` / `trim_cache`. Stage-0 generation loop in `worker.py:172-289` drafts k tokens, sends them with the activation message, and accept/rejects on the returned target probabilities. Last stage in `worker.py:367-378` returns target probs + samples for all draft positions plus a bonus.

**Measured:** 77.1% per-token acceptance, ~4.6 tokens accepted per round on average, k=6.

### ~~4.2 Cascade / early exit~~ ✅ DONE

**What we did:** After the first draft token, Stage 0 checks its top-1 probability. If `p0 >= cascade.confidence_threshold` (default 0.9), Stage 0 runs its layers on `last_tok` only, fires the hidden to downstream stages as `is_cascade=True` (fire-and-forget, no recv), and moves on. Downstream stages handle `is_cascade` by updating their KV and not sending a response. On a cascade miss, Stage 0 drafts k-1 more tokens and falls through to the speculative path. End-of-generation summary prints `Cascade: X/Y steps handled locally (Z%)`.

Stacks with speculative decoding: local cascade handles easy tokens, speculative handles medium, full pipeline handles hard tokens.

---

## Phase 2.5 — Speculative pipelining *(next implementation target)*

### Concept

Stage 0 currently goes idle the moment it fires its activation message — it just blocks on `recv_msg` waiting for Stage 1 to verify. **That idle window is ~3,200 ms** in the current config (Stage 1 verifier compute), and the next round's draft will cost ~1,200 ms. If we launch the next draft in a background thread the moment we send the verifier batch, Python's GIL is released while the main thread blocks on ZMQ's C-level `poll()`, so PyTorch can run the draft in parallel.

```
draft (1200ms) → stage0 fwd (650ms) → send → recv-wait (3200ms) → accept/reject
                                              │
                                              └── background: draft for round N+1
                                                  (hides inside the 3200ms wait)
```

On a hit, the draft cost disappears from the next round's critical path entirely.

### The seed problem

To pre-draft round N+1 during round N's wait, you need to know what token to start drafting from — the "seed." But the seed depends on round N's accept/reject outcome, which only arrives with the verifier response.

The pragmatic solution: assume an **optimistic seed** (the last drafted token, `draft_tokens[K-1]`). If accept/reject confirms it, adopt the prefetched draft. If not, discard and re-draft normally — the worst case is identical to today's behavior, so this is risk-free.

Hit-rate expectations at ~0.77 per-token acceptance:

| K | est. hit rate |
|---|---|
| 4 | ~25–35% |
| 6 (current) | ~20–30% |
| 8 | ~15–25% |

### Why this is more valuable than the original projection said

The original `SPECULATIVE_PIPELINE.md` (now retired into this doc) projected ~6–10% TPS gain because it modeled the overlap window as the WAN RTT (~80 ms). That was correct under network-bound assumptions. Under the current compute-bound regime:

```
hit_rate × draft_time_saved / round_time
= 0.25 × 1,200 ms / 5,050 ms
≈ 6% per round (averaged over hits + misses)
```

But that's the *average* — on hits it's ~24% per round. So:

| scenario | est. TPS gain |
|---|---|
| Current (k=6, 1B draft, bf16, ~25% hit) | **~20–25%** |
| With tree speculation (~55–65% hit) | ~35–45% |
| After Phase 1.6 quantization (verifier wait shrinks to ~1,000 ms) | drops back to ~6–10% |

**Order matters.** Do speculative pipelining *before* weight quantization to capture the full gain while the verifier wait window is still big. After quantization, the draft no longer fits inside the wait, and this optimization's relative value collapses.

### Implementation sketch

- **`draft.py`**: add `start_prefetch(seed, k)` that snapshots the KV cache and launches a daemon thread running `draft_k_tokens` from the optimistic seed. Add `consume_prefetch(actual_seed_id, assumed_seed_id)` that joins the thread and returns the cached result on match, `None` on mismatch.
- **`worker.py`** (Stage 0 spec branch, around `worker.py:230`): after `send_msg`, call `draft_model.start_prefetch(draft_tokens_t[:, -1:], k)`. Block on `recv_msg` as normal. After accept/reject, call `consume_prefetch` and either adopt the pre-draft or discard.
- The thread holds a *cloned* KV cache, so there is no shared mutable state. Join happens before any KV manipulation in the main thread.
- Add a `pipeline_hits / pipeline_steps` counter to the end-of-generation summary.

### Tree speculation upgrade (later, ~1–2 weeks)

The seed problem can be partially eliminated by drafting a *tree* during the wait instead of a single chain — e.g. 2 tokens from each of 3 candidate seeds (`draft[K-1]`, `draft[K-2]`, `last_tok`). Hit rate jumps from ~25% to ~55–65% at the cost of 3× more draft inference inside the same wait window — and that compute is mostly free because the window is already paid for. Architecturally this is EAGLE-2 / Medusa adapted for pipeline-parallel WAN. Natural follow-up after the single-seed implementation is validated.

---

## Path A vs Path B — fork in the road

This decision is the most important strategic call in the project right now. Make it deliberately, ideally after Phase 2.5 lands and before Phase 1.6 weight quantization.

### Path A — stay HuggingFace, hand-rolled

Implement Phase 1.6 with `torch.ao.quantization` int8 dynamic quant. Accept that HF's CPU Linear kernels are 3–5× slower than llama.cpp's AVX-512 paths. The research narrative stays clean: every component (KV cache, codec, spec decode, cascade, spec pipelining, MoE eventually) is built from scratch and explainable.

Trade-off: the absolute numbers will look weak compared to llama.cpp on the same hardware, even with all the cleverness above. Reviewer reaction may be "interesting design, but why is it 3× slower than llama.cpp baseline?"

### Path B — port onto llama.cpp's RPC backend

Fork llama.cpp's RPC server and lift this repo's novel layers (cascade, WAN-tuned speculative pipelining, eventually MoE-WAN routing) on top of it. Inherit GGUF quantization (Q4_K_M, Q5_K_M), AVX-512/NEON kernels, and battle-tested KV cache management for free.

Trade-off: lose the educational from-scratch story for the kernel layer. The novel contributions (architectural — cascade, pipelining, WAN-aware routing) survive intact.

### How to decide

Run Path A's int8 dynamic quant experiment first (1 day of work). If it gets to ~1.7 TPS as the table predicts, Path A is viable. If the kernels disappoint and it stalls below 1.4 TPS, that's the signal that you've hit the limit of what HF's CPU paths will give you, and Path B becomes the only way to credibly hit M4–M6.

---

## Phase 3 — Architectural shift to MoE *(repositioned)*

### Original framing (kept for context)

In an MoE model (Mixtral, DeepSeek-V3, Qwen-MoE), each token routes to only ~2 of N experts per MoE layer. Sharding *experts* across machines instead of *layers* means each token only hits the 2 machines holding its routed experts.

### Honest reassessment for current hardware

MoE's value proposition is "fewer active parameters per token" — a *compute* win. On a CPU+WAN system that's already memory-bandwidth-bound:

- Mixtral 8×7B at Q4 still has ~13B active params per token. Bandwidth requirement per token is similar to a 13B dense model — not obviously better than Llama-3-7B Q4 dense.
- Scatter-gather routing adds 1–3 hops per layer where dense pipeline parallelism has 0–1. WAN-hop count goes *up*.
- Shared attention path (RMSNorm, attention QKV, output proj) gets duplicated on every machine, costing memory.

**Bottom line:** dense + Q4 + speculative + pipelining will likely beat dense → MoE on 2× CPU droplets. MoE is the right move when:
- Compute is plentiful relative to memory (i.e., GPUs)
- You have ≥3 machines so expert sharding actually saves bandwidth
- You're targeting model sizes where dense Q4 won't fit on one machine

For consumer-hardware (M6) experiments and ≥70B-class models, MoE is still the answer. For 3B–13B on 2× CPU droplets, it probably isn't.

**Decision:** keep MoE as the long-term architectural milestone (M4) but stop treating it as the next-up phase. Phases 1.6 and 2.5 should land first; reassess MoE after measuring the dense+Q4 ceiling.

---

## Phase 4 — Other items (pick what fits)

### 4.1 Split prefill vs decode

One machine does prefill (compute-bound, latency-insensitive), another does decode (latency-sensitive). Requires streaming KV cache between them once per request. Worth it only with asymmetric hardware. Skipped on symmetric DO droplets.

### 4.3 Microbatching / 1F1B scheduling

Only helps for multi-user serving (agent workflows, API server). Skip for single-user interactive.

### 4.4 CUDA graphs / persistent kernels

CPU-only right now. Becomes relevant if any machine gets a GPU.

---

## Sequencing

| Phase | Effort | Result | Measured / projected TPS |
|---|---|---|---|
| Baseline (GPT-2, no cache) | — | — | ~0.3 TPS |
| ~~1.1 KV cache~~ ✅ | 2–3 days | 1.7× | ~0.5 TPS |
| ~~1.2 int8 activations~~ ✅ | 1 day | 1.3× | ~0.7 TPS |
| ~~1.3 Buffer tuning~~ ✅ | 2 days | latency jitter only | ~0.7 TPS |
| ~~1.4 Switch to Llama 3.2-3B~~ ✅ | 1 day | real model | — |
| ~~2. Speculative decoding~~ ✅ | 1–2 weeks | 77% acceptance, k=6 | **1.10 TPS** |
| ~~4.2 Cascade~~ ✅ | 3 days | 29% steps local | included above |
| **2.5 Speculative pipelining** | 1–2 days | hides 1.2s draft in 3.2s wait | **~1.35 TPS** |
| **1.6 Weight quantization (int8)** | 3–5 days | ~2× memory bandwidth | **~1.8 TPS** |
| **Path A vs B decision point** | — | choose based on int8 result | — |
| 1.6b GGUF Q4 (Path B only) | 1–2 weeks | port onto llama.cpp RPC | ~2.5–3 TPS |
| 2.6 Tree speculation | 1–2 weeks | acceptance ~85% effective | ~+25% |
| 3. MoE architecture | 2–4 weeks | only after Q4 dense saturates | TBD |
| 4.1 Prefill/decode split | 1 week | asymmetric hardware only | — |
| 4.3 Microbatching / 1F1B | 1–2 weeks | multi-user only | — |

**Current status:** bottleneck is CPU memory bandwidth, not network. Speculative pipelining is the next easy win and must land *before* quantization to capture the full overlap gain. Weight quantization is the next big lever. MoE is deferred until Q4 dense is exhausted.

---

## Milestones / exit criteria

- **~~M1 — Phase 1 complete~~** ✅: KV cache + int8 activations on Llama 3.2-3B across two WireGuard droplets.
- **~~M2 — Speculative decoding working~~** ✅: 77.1% acceptance, 1.10 TPS on 2× s-8vcpu-16gb, Llama 3.2-3B with 1B draft.
- **M3 — Speculative pipelining**: draft hidden inside verifier wait, ~1.35 TPS target, `pipeline_hits` counter validates ≥20%.
- **M3.5 — Weight quantization**: int8 dynamic quant in HF (Path A) gets to ≥1.7 TPS, OR triggers move to Path B.
- **M4 — Q4 dense ceiling**: whichever path wins, measure the absolute ceiling for dense-pipeline-parallel + spec + cascade. This is the number MoE has to beat.
- **M5 — WAN across different ISPs**: two machines on genuinely different networks (not same-region DO), ≥2 TPS. Real test of the research thesis.
- **M6 — Consumer hardware**: laptops / desktops over the public internet, not cloud VMs.
- **M7 — MoE sharded inference (only if M4 saturates)**: Mixtral 8x7B Q4 or Qwen-MoE across 3 machines, must beat M4's dense number to be worth the engineering cost.

---

## Papers / references

- **Phase 1.2 (activations)**: *SmoothQuant* (Xiao et al. 2023), *ZeroQuant* — activation quantization math.
- **Phase 1.6 (weights)**: *LLM.int8()* (Dettmers et al.), *GPTQ* (Frantar et al.), llama.cpp `k-quants` source for Q4_K_M / Q5_K_M.
- **Phase 2**: Leviathan et al. 2023 (*Fast Inference from Transformers via Speculative Decoding*), *EAGLE-2*, *Medusa*.
- **Phase 2.5 (pipelining)**: no canonical paper — adapted from spec-decode literature for WAN pipeline-parallel; closest prior art is `Petals` (Borzunov et al. 2023) on the WAN side.
- **Phase 2.6 (tree spec)**: *EAGLE-2*, *Medusa*, *SpecInfer*.
- **Phase 3 (MoE)**: *Switch Transformer* (Fedus et al.), *Mixtral of Experts*, *Petals*.
- **Phase 4.1**: *DistServe*, *Splitwise* — prefill/decode disaggregation.
- **Phase 4.3**: *PipeDream* (Narayanan et al.), *GPipe* (Huang et al.).
