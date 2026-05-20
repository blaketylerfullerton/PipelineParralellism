# Experiment — Does Speculative Pipelining Pay Over WAN?

**Status:** analytical gate implemented; empirical run pending M3.5 llama.cpp RPC
**Owner:** Blake
**Date drafted:** 2026-05-11
**Decision deadline:** end of M3.5 (two-machine llama.cpp RPC working)

---

## TL;DR

Relay's central research bet is that **hiding draft compute inside the verifier
round-trip becomes net-positive once the WAN's share of the per-token budget
grows** (which is exactly what happens after the move to llama.cpp Q4_K_M).
Current speculative numbers don't show that yet — in fact `spec k=2` loses to
`no-spec` on both the local HF baseline (6.53 vs 7.27 TPS) and the
DigitalOcean baseline (2.77 vs 3.15 TPS).

This document defines the **smallest experiment that can either confirm or
kill that hypothesis**, with kill/confirm criteria written down *before* the
runs so we can't move goalposts after the fact.

---

## What We're Doing

Run a controlled three-way comparison on a two-machine llama.cpp RPC setup
with Q4_K_M Llama-3.2-3B, sweeping injected WAN RTT, and measuring TPS for:

1. **no-spec** — straight pipelined decode, one verifier round per token
2. **vanilla spec** — draft `k` tokens serially, then verifier round
3. **pipelined spec** — draft `k` tokens *during* the verifier round-trip

The goal is not to ship a feature. The goal is to answer one question:

> Is there any realistic RTT regime where pipelined spec beats no-spec
> by enough to justify keeping it in the design?

---

## Why We're Doing It

Three reasons, in order of importance.

### 1. The research thesis lives or dies here

The README claims speculative pipelining is "more relevant after kernel
acceleration, not less." That claim is currently *unsupported* by Relay's own
numbers. Every other contribution (multi-ISP measurement, WAN-aware
orchestration) is more engineering than insight without a positive
speculative-pipelining result. If this hypothesis dies, the thesis pivots to
a smaller measurement-paper story. Better to find out now than after another
three months of orchestration work.

### 2. The hypothesis is testable cheaply

We don't need real multi-ISP infrastructure to falsify it. We need two
machines with controllable RTT. If the technique can't win at *any* injected
RTT on a clean two-machine setup, it's not going to suddenly start winning
across residential ISPs with jitter and packet loss layered on top.

### 3. The result reshapes downstream priorities

- **Confirmed:** speculative pipelining becomes the central plot of the paper;
  real-WAN and larger-model experiments are extensions of a known positive
  result.
- **Killed:** drop draft/verifier complexity from the llama.cpp orchestrator,
  focus the project on WAN-aware stage placement and the multi-ISP
  characterization story.
- **Ambiguous (most likely):** escalate to a model where the verifier forward
  is large enough that the WAN share of the round naturally grows — i.e. the
  70B-class regime the thesis actually targets.

---

## Background — Why the Current Numbers Don't Settle It

| Environment | Variant | TPS | Notes |
|---|---|---:|---|
| Local HF bf16 | no-spec | 7.27 | fastest local |
| Local HF bf16 | spec k=2 | 6.53 | spec loses by ~10% |
| 2× DO HF bf16 | no-spec | 3.15 | fastest WAN |
| 2× DO HF bf16 | spec k=2 | 2.77 | spec loses by ~12% |

Both environments lived in the bf16 / CPU-bound regime where the verifier
forward dominated the round (`~3.2 s` for stage 1, vs `~80 ms` of WAN RTT).
In that regime, draft cost shows up as pure overhead — there's nothing to
hide it behind.

The Q4_K_M regime is qualitatively different. Per-token compute drops ~12×,
so the *relative* share of WAN RTT in the round grows from <2% to something
material. That's the regime where hiding draft compute inside the round-trip
could finally net out positive. None of the existing numbers test that
regime.

---

## Analytical Model (do this first, before any code)

```
T_round(no-spec)        ≈ T_v + R
T_round(vanilla spec)   ≈ T_d·k + T_v + R           (draft is serial)
T_round(pipelined spec) ≈ max(T_d·k, T_v + R)       (draft hidden in RTT)
expected tokens / round ≈ 1 + α·k  (geometric approx)
```

Variables:

| symbol | meaning | how to measure |
|---|---|---|
| `T_v` | verifier forward, one stage, Q4_K_M | `llama-bench` single-machine |
| `T_d` | draft forward, one token | bench the chosen draft model alone |
| `R` | one-way activation latency (≈ RTT/2 here) | `tc netem` controlled |
| `α` | draft acceptance rate at given `k` | single-machine spec run |
| `k` | draft length | hyperparameter, sweep `{2, 4, 6}` |

**Deliverable for this stage:** `deploy/speculative_pipelining_model.py` takes
the measured `T_v`, `T_d`, `α` values and writes predicted TPS vs `R` for all
three configurations to:

- `logs/speculative-pipelining/model/prediction.csv`
- `logs/speculative-pipelining/model/prediction.svg`
- `logs/speculative-pipelining/model/decision.json`

Example:

```bash
./deploy/speculative_pipelining_model.py \
  --verifier-ms <T_v> \
  --draft-ms <T_d> \
  --alpha-by-k 2:<alpha2>,4:<alpha4>,6:<alpha6> \
  --rtts-ms 0,50,100,200,400
```

If the model says no crossover exists at any `R ≤ 500 ms`, we don't run the
empirical experiment — hypothesis dies on paper.

Estimated effort: half a day.

---

## Empirical Setup

### Hardware

Two machines connected by WireGuard:

- **Stage 0:** local MacBook Air (M4) or one DO droplet
- **Stage 1:** a second DO droplet in the same region

Same-region same-provider is fine for this experiment. We're not measuring
real WAN; we're measuring *response to controllable RTT*. Real multi-ISP
comes after.

### Software prerequisites

These are M3.5 deliverables. None of this experiment runs until they're
done.

- [ ] llama.cpp `rpc-server` running on Stage 1
- [ ] llama.cpp client on Stage 0 driving a two-stage Q4_K_M split
- [ ] Relay's draft / verifier loop ported from `src/draft.py` (HF) to the
      llama.cpp orchestration layer. The pieces that transfer:
      `DraftModel.start_prefetch`, `consume_prefetch`, and the daemon-thread
      scaffolding. What changes: the verifier call is now `llama_decode()`
      rather than a stage-N forward, and the KV snapshot is on the draft
      model only.
- [ ] `tc netem` configured on the WireGuard interface for RTT injection

### RTT sweep

Inject RTT on the WireGuard interface:

```
sudo tc qdisc add dev wg0 root netem delay <X>ms
```

Sweep: `R ∈ {0, 25, 50, 100, 200} ms`.

The 200 ms point matters — that's roughly residential-cross-continent
territory and represents the regime the thesis actually cares about.

### Configurations

| name | spec enabled | pipeline_prefetch | `k` |
|---|---|---|---:|
| no-spec | false | — | — |
| vanilla-spec | true | false | 4 |
| pipelined-spec | true | true | 4 |

Hold `k=4` constant for the main sweep. Once a configuration is confirmed
positive, do a follow-up `k ∈ {2, 4, 6}` sweep at the best RTT point.

### Protocol

- Same 5 prompts across all runs (use a fixed prompt set, ~30 tokens each)
- Generate 256 tokens per run
- 5 seeds per (config × RTT) cell
- Total runs: 3 configs × 5 RTTs × 5 seeds = **75 runs**

Report median TPS per cell plus a 95% bootstrap CI. Plot TPS vs `R` for all
three configurations on one chart, overlaid with the analytical-model
prediction from Stage 0.

Use `deploy/speculative_pipelining_analyze.py` for the median/CI and decision
summary once the run CSV exists:

```bash
./deploy/speculative_pipelining_analyze.py logs/speculative-pipelining/runs.csv
```

By default, the analyzer reads
`logs/speculative-pipelining/model/decision.json` and only emits `confirmed`
when the empirical threshold also matches the Stage 0 analytical threshold
within ~2×. If that file is missing, or if the Stage 0 model was not confirmed,
an empirical TPS win is still treated as `ambiguous`.

The input CSV schema is:

```csv
config,r_ms,seed,tps,draft_accept_pct
no-spec,0,0,38.1,
vanilla-spec,0,0,34.7,55.2
pipelined-spec,0,0,36.9,55.2
```

---

## Kill / Confirm Criteria

Written down **before** running.

### Killed if

Pipelined spec ≤ no-spec at every injected RTT, including `R = 200 ms`.
No regime exists where the technique pays at this model scale. Drop the
mechanism from the llama.cpp orchestrator and pivot the paper story.

### Confirmed if

There exists an RTT threshold `R*` such that pipelined spec beats *both*
no-spec and vanilla spec by **≥ 20% TPS** at `R ≥ R*`, **and** the observed
threshold matches the Stage 0 analytical prediction within ~2×. Matching
the prediction matters as much as the win — it's what makes the result a
mechanism rather than a fluke.

### Ambiguous if

Pipelined spec wins, but only narrowly (<20%), or only at unrealistic RTTs
(>500 ms), or only at suspicious acceptance rates. **Default action on
ambiguous: escalate to a larger model** (target: a 70B-class or 30B MoE
quantization that doesn't fit on a single consumer machine). The thesis
target was always that regime — small models were never going to make the
WAN share dominate the round.

---

## What This Buys Us

| outcome | months saved or unlocked |
|---|---|
| killed | save 2–3 months of orchestration work built on a dead hypothesis; pivot to a still-publishable measurement story |
| confirmed | central plot for the paper; subsequent experiments extend a known-positive result rather than fish for one |
| ambiguous | clear, evidence-backed trigger to move to the 70B regime — which is where the thesis was always headed anyway |

The expected total cost of running this is **1–2 weeks** once M3.5 is in:
half a day for the analytical model, a few days to port the prefetch loop
to the llama.cpp path, a day for the sweep, a day for analysis.

That is small compared to the cost of being wrong about the central
research bet.

---

## Open Questions

- Which draft model? `gpt2` (current HF path) is wrong for the Q4_K_M
  setting — it doesn't share a tokenizer with Llama-3.2-3B. Candidates:
  Llama-3.2-1B Q4_K_M as the draft, or TinyLlama. Acceptance rate is a
  free variable until this is pinned.
- Do we need to measure `α` separately per RTT cell? `α` shouldn't depend
  on RTT in theory; verify empirically that it doesn't drift across runs.
- Single-stream only, or also a batched-arrival sensitivity test? Single
  stream first — batching is a separate axis and dilutes the result.
