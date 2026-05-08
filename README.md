# Relay

> **Pipeline-parallel LLM inference across the open internet** — laptops, residential
> connections, and low-cost cloud VMs connected by WireGuard. Relay sits on top of
> [llama.cpp](https://github.com/ggml-org/llama.cpp) and focuses on the coordination
> problems specific to high-RTT, heterogeneous, multi-ISP deployments.

<p align="center">
  <img src="images/networking1.png" width="300"/>
</p>

<p align="center">
  <img src="images/diagram.png" width="700"/>
</p>

---

# Thesis

Distributed LLM inference has so far been a datacenter problem: NVLink, InfiniBand,
low-jitter LAN. Those assumptions don't hold across residential ISPs, behind NATs,
with asymmetric bandwidth and tens of milliseconds of RTT. Relay is a research
platform exploring how far pipeline-parallel transformer inference can go on that
hardware — and the answer is shaped almost entirely by coordination, not kernels.

The kernel question is settled. llama.cpp's k-quants and GGML execution graph are
purpose-built for CPU and Apple Silicon, and a project-level diagnostic
(2026-05-07) confirmed they outperform a hand-rolled HuggingFace path by ~11× on
identical hardware. Relay uses them. The open questions are at the layer above:

1. **Speculative pipelining for high-RTT links** — hiding draft compute inside
   the verifier round-trip. The value of this technique scales with the WAN's
   share of the round budget rather than absolute compute cost, so it becomes
   *more* relevant after kernel acceleration, not less.
2. **Empirical characterization of LLM inference across real multi-ISP WAN** —
   jitter, packet loss, asymmetric bandwidth, and how each shapes TPS as model
   size and pipeline depth vary. Most prior work runs intra-datacenter or on
   well-conditioned volunteer grids; this regime is underexplored.
3. **WAN-aware orchestration policy** — placing stages, tuning speculative
   `k`, and scheduling prefetch when machines have asymmetric compute and uplink
   budgets.

What Relay deliberately is *not*: a from-scratch inference engine, an alternative
to llama.cpp, or a high-throughput serving system. The kernel work is a solved
problem; Relay reuses it. The unsolved question is whether the open internet can
serve as the interconnect at all.

---

# Motivation

Most distributed inference systems assume:

* GPUs connected through high-bandwidth fabrics
* low-latency datacenter networking
* homogeneous hardware
* tightly coordinated clusters

Relay deliberately does not. The target environment is laptops, desktops, spare
home servers, low-cost cloud VMs, and eventually heterogeneous consumer hardware
over the open internet.

The empirical bottleneck has shifted twice during the project, and that movement
is what defines the current research direction:

* **Phase 0 (GPT-2 in PyTorch, ~0.3 TPS)** — per-token compute was so slow that
  WAN RTT was a rounding error.
* **Phase 1 (Llama-3.2-3B in HF bf16)** — CPU compute and memory bandwidth still
  dominated; RTT measured at **< 2% of round time**.
* **Phase 1.6 (Llama-3.2-3B Q4_K_M via llama.cpp)** — per-token compute drops
  ~12× and WAN RTT finally becomes a meaningful fraction of the round.

The third regime is the one where the research questions get interesting, and
where Relay is now focused.

---

# What's Borrowed, What's Built

Explicit honesty about the layering matters more than usual here, because the
project's framing changed mid-flight when the kernel question collapsed onto
llama.cpp.

| Layer | Provider | Notes |
|---|---|---|
| GGML kernels (NEON / AVX-512 / Metal) | llama.cpp | Reimplementing them produces slower kernels — verified empirically. |
| Quantization (Q4_K_M, Q5_K_M, etc.) | llama.cpp k-quants | Outlier-aware; both x86 and ARM paths. |
| KV cache | llama.cpp | Better-managed than the from-scratch version. |
| Layer-level pipeline parallelism | llama.cpp RPC backend | Splits graph across machines via `--rpc` / `-ts`. |
| Per-token transport (currently) | llama.cpp's TCP transport | Pluggable seam at `ggml/src/ggml-rpc/transport.{h,cpp}`. |
| **Speculative pipelining (next-round prefetch)** | **Relay** | Latency-hiding when the verifier round travels over WAN. Not in upstream. |
| **Multi-ISP WAN benchmark harness** | **Relay** | Reproducible deployments, cross-ISP measurement. |
| **WAN-aware orchestration policy** | **Relay** | Stage placement, `k` tuning, prefetch scheduling. |
| **Optional ZMQ-over-WireGuard transport** | **Relay** | Drop-in replacement for `transport.cpp`. Built only if measurement shows it's load-bearing. |

---

# Current State

Relay is mid-pivot from a from-scratch HuggingFace pipeline to a llama.cpp-backed
implementation. Both code paths currently exist in the tree:

* **HuggingFace path (legacy)** — full pipeline-parallel split across machines
  with ZMQ activation transport, KV cache, int8 wire codec, speculative
  decoding, cascade, and speculative pipelining. Captures the bf16 baseline
  numbers and serves as the empirical foundation for the kernel comparison.
* **llama.cpp path (in progress)** — orchestration layer above llama.cpp's
  RPC backend. Per-stage forward and KV management are delegated; Relay
  handles draft/verifier loop, speculative pipelining, and WAN measurement
  at the application level above `llama_decode()`.

Active work is on the llama.cpp side. The HF path is retained as the reference
implementation that produced the bf16 baseline.

---

# Current Findings

## Path A diagnostic + Path B kernel validation (Llama-3.2-3B, M4 MacBook Air)

| backend | precision | TPS | output | speedup vs bf16/CPU |
|---|---|---:|---|---:|
| HF / PyTorch CPU | bf16 | 3.37 | coherent | 1.0× *(baseline)* |
| HF / PyTorch MPS | bf16 | 7.74 | coherent | 2.3× |
| HF / PyTorch CPU + `torch.ao` int8 | int8 dyn | 1.31 | **gibberish** | 0.4× |
| **llama.cpp CPU (`-ngl 0`)** | **Q4_K_M** | **38.31** | coherent | **11.4×** |
| **llama.cpp Metal** | **Q4_K_M** | **43.45** | coherent | **12.9×** |

Two findings stack:

* Hand-rolled HF int8 dynamic quant produces a 2.6× *slowdown* AND broken output
  on QNNPACK (the `reduce_range` interaction with Llama's outlier weight
  distribution). Even the optimistic FBGEMM (x86) projection of 1.7× sits below
  llama.cpp Q4_K_M's floor. **Path A killed.**
* llama.cpp Q4_K_M kernels are ~11× faster than HF bf16 on the same hardware.
  Even llama.cpp's CPU-only path beats HF's MPS path by ~5×. **Kernel question
  closed; WAN question reopens.**

## HF baseline across two DigitalOcean droplets (May 2026)

| variant | TPS |
|---|---:|
| no-spec | 3.15 |
| spec k=2 | 2.77 |

These are the numbers M3.5b (two-machine RPC over stock TCP-over-WireGuard) has
to beat to justify the pivot.

## Earlier per-token speculative results (HF, single machine)

| variant | TPS | avg ms/token |
|---|---:|---:|
| no-spec | 7.27 | 137.6 |
| spec k=2 | 6.53 | 153.2 |
| spec k=6 | 5.54 | 180.4 |

Speculative decoding does not currently beat no-spec on either backend. It is
preserved at the application level for re-evaluation under the new compute regime,
where the verifier wait shrinks and the WAN share of the round grows.

---

# Architecture

Relay is now structured as an orchestration layer above llama.cpp:

```text
┌──────────────────────────────────────────────────┐
│ Relay orchestrator (Python)                      │
│   • draft / verifier loop                        │
│   • speculative pipelining (background draft)    │
│   • WAN measurement / instrumentation            │
└──────────────────────┬───────────────────────────┘
                       │ llama_decode()
                       ▼
┌──────────────────────────────────────────────────┐
│ llama.cpp                                        │
│   • GGML kernels (NEON / AVX-512 / Metal)        │
│   • Q4_K_M weights, fp16 activations             │
│   • KV cache, graph executor                     │
└──────────────────────┬───────────────────────────┘
                       │ RPC backend
                       ▼
        ┌──────────────┴──────────────┐
        ▼                             ▼
   Stage 0 (rpc-server)         Stage N (rpc-server)
   layers [0..N/2]              layers [N/2..end]
                       │
                       │ TCP-over-WireGuard
                       │ (or ZMQ-over-WireGuard, pending)
                       ▼
             real WAN, multi-ISP
```

The pipeline split itself is implicit in llama.cpp's tensor placement and graph
executor — Relay does not reimplement it. Relay's contributions live at the
top and bottom of this stack: orchestration above `llama_decode()`, and
optionally a swapped transport at the very bottom.

---

# Network Topology

```text
┌────────────────────────────────────────────┐
│ Stage 0  (orchestrator + first half)      │
│   draft model (small)                      │
│   speculative scheduler                    │
│   forward through local layers             │
└──────────────────────┬─────────────────────┘
                       │
                       │ WireGuard VPN
                       │ (TCP today, ZMQ optional)
                       ▼
┌────────────────────────────────────────────┐
│ Stage N  (last layers)                    │
│   complete forward                         │
│   logits / sample                          │
│   return to Stage 0                        │
└────────────────────────────────────────────┘
```

Relay intentionally uses real inter-machine networking rather than localhost
simulation to expose latency, serialization cost, bandwidth limits, packet
overhead, and synchronization behavior that disappear on a single box.

---

# Why DigitalOcean?

DigitalOcean droplets act as a reproducible WAN testbed. They are not the end
goal.

The real target is geographically separate machines, different ISPs, different
hardware, unreliable consumer networking. Cloud VMs simply make deployment,
benchmarking, iteration, and instrumentation easier during development.

WireGuard provides the encrypted overlay network. Relay runs on top using
TCP-over-WireGuard today, with ZMQ as an optional drop-in once measurement shows
it warrants the work.

---

# Project Structure

```text
relay/
├── src/
│   ├── launch.py
│   ├── worker.py
│   ├── model.py
│   ├── draft.py
│   ├── utils.py
│   └── dashboard.py
│
├── deploy/
│   ├── relayctl.py
│   ├── deploy.sh
│   ├── run.sh
│   ├── run_local.py
│   ├── benchmark_matrix.py
│   └── teardown.sh
│
├── docs/
│   └── PLAN.md           ← living research plan + decisions
│
├── bench_bf16_cpu.yaml   ← reproducible Path A diagnostic configs
├── bench_bf16_mps.yaml
├── bench_int8_cpu.yaml
│
├── config.yaml
├── config.smoke.yaml
├── config.nospec.yaml
└── requirements.txt
```

---

# Quick Start

## Prerequisites

```bash
brew install doctl wireguard-tools llama.cpp
doctl auth init
```

Create `deploy/.env`:

```bash
DO_TOKEN=your_token
GITHUB_REPO=youruser/relay
DO_SSH_KEY=your_ssh_key
HF_TOKEN=your_hf_token
REGION=sfo3
```

---

# Deploy WAN Nodes

```bash
./deploy/relayctl.py provision
```

This:

1. Creates droplets
2. Configures WireGuard
3. Clones the repo
4. Installs dependencies
5. Downloads model weights
6. Bootstraps the pipeline environment

Check deployment status:

```bash
./deploy/relayctl.py status
./deploy/relayctl.py doctor
```

---

# Run Relay

```bash
./deploy/relayctl.py run \
  --prompt "the future of computing is"
```

---

# Local Testing

Smoke test:

```bash
./deploy/run_local.py \
  --config config.smoke.yaml \
  --prompt "hello from localhost"
```

Baseline run:

```bash
./deploy/run_local.py \
  --config config.nospec.yaml \
  --prompt "the future of computing is"
```

Benchmark matrix:

```bash
./deploy/benchmark_matrix.py \
  --base-config config.yaml \
  --ks 1,2,4,6
```

Path A diagnostic (reproducible):

```bash
./deploy/run_local.py --config bench_bf16_cpu.yaml --prompt "..."
./deploy/run_local.py --config bench_bf16_mps.yaml --prompt "..."
./deploy/run_local.py --config bench_int8_cpu.yaml --prompt "..."   # broken output expected
```

llama.cpp local kernel reference:

```bash
llama-bench -hf unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M -ngl 0   # CPU
llama-bench -hf unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M          # Metal default
```

---

# Configuration

```yaml
pipeline:
  num_stages: 2

model:
  name: "unsloth/Llama-3.2-3B"
  arch: "llama"
  dtype: "bfloat16"

quantization:
  weight_mode: "bf16"   # bf16 | int8 (Path A, dead) | q4_k_m (Path B, in progress)

compression:
  mode: "int8"          # legacy HF activation codec; ignored on llama.cpp path

speculative:
  enabled: false
  k: 1
  pipeline_prefetch: false

cascade:
  enabled: false        # parked: incompatible with llama.cpp whole-graph decode
```

---

# Research Directions

**Active**

1. **llama.cpp port (M3.5)** — single-machine `llama_decode` smoke first; then
   `rpc-server` on a second WireGuard-connected machine, two-machine Q4_K_M
   over stock TCP. The result decides whether a ZMQ transport replacement is
   load-bearing.
2. **Speculative pipelining over WAN** — characterize when next-round draft
   prefetch wins as a function of RTT, draft acceptance rate, and verifier
   round time. Becomes more relevant in the post-Q4 regime where WAN dominates.
3. **Multi-ISP WAN measurement (M5)** — TPS vs RTT, jitter, packet loss,
   asymmetric bandwidth across genuinely different ISPs (not same-region cloud).

**Queued**

* Tree speculation (EAGLE-2 / Medusa) at the application level.
* Asymmetric-hardware orchestration (slow + fast machine, residential + cloud).
* MoE expert sharding for WAN, after dense Q4 saturates (M4).

**Parked**

* Cascade / early exit — blocked by llama.cpp's whole-graph decode model, and
  its value shrinks at 11× kernel speedup. Re-evaluate after M4.
* Custom int8 activation codec — RTT was already negligible in the bf16 regime,
  and llama.cpp's RPC handles fp16 activations with built-in framing.
* From-scratch KV cache — better-managed by llama.cpp.

---

# Non-Goals

Relay is **not**:

* a from-scratch inference engine — kernel work delegates to llama.cpp
* a production inference framework
* a decentralized AI marketplace
* a blockchain project
* a high-throughput serving system
* a replacement for datacenter inference

The current focus is systems research and empirical measurement of WAN-distributed
LLM inference at the coordination layer.

---

# Long-Term Vision

Relay explores a broader possibility:

> Large models may eventually run cooperatively across globally distributed
> commodity hardware instead of only centralized clusters.

Whether that becomes practical depends almost entirely on:

* communication efficiency
* scheduling
* transport overhead
* and distributed systems design.

Relay exists to experimentally investigate those limits, on top of a kernel
substrate that's already as fast as kernels get.

<p align="center">
  <b>Built on cheap hardware, open networks, and stubborn optimism ☕</b>
</p>
