# Relay

> **Experimental research platform for WAN-distributed LLM inference.**
>
> Relay explores whether large language models can be executed cooperatively across ordinary internet-connected machines — without shared memory, datacenter networking, or specialized hardware.
>
> Instead of assuming NVLink, InfiniBand, or a high-speed LAN, Relay treats the public internet itself as the interconnect. The project focuses on the systems problems that appear when transformer inference is pushed across real WAN conditions: latency, bandwidth limits, heterogeneous hardware, activation transfer costs, and pipeline coordination.

<p align="center">
  <img src="images/networking1.png" width="300"/>
</p>

<p align="center">
  <img src="images/diagram.png" width="700"/>
</p>

---

# Motivation

Most distributed inference systems assume:

* GPUs connected through high-bandwidth fabrics
* low-latency datacenter networking
* homogeneous hardware
* tightly coordinated clusters

Relay deliberately does not.

The target environment is:

* laptops
* desktops
* spare home servers
* low-cost cloud VMs
* eventually heterogeneous consumer hardware over the open internet

The central research question is:

> Can pipeline-parallel transformer inference remain practical when communication happens over WAN instead of a datacenter fabric?

Relay is currently focused on measuring and reducing the real bottleneck:

# Communication overhead

In WAN-distributed inference, moving activations between machines quickly becomes more expensive than the compute itself. Relay experiments with:

* pipeline parallelism
* KV-cached decoding
* activation compression
* asynchronous execution strategies
* speculative decoding (experimental)
* heterogeneous stage partitioning

The goal is not to beat single-machine inference.

The goal is to understand:

* where WAN inference breaks down
* what techniques help
* what techniques fail
* and how far commodity distributed inference can realistically scale.

---

# Current State

Relay currently supports:

* multi-stage transformer partitioning
* WAN-connected execution over WireGuard
* ZeroMQ-based activation transport
* KV-cached autoregressive decoding
* int8 activation compression
* reproducible DigitalOcean deployments
* local multi-process simulation
* experimental speculative decoding paths

The current implementation prioritizes:

* reproducibility
* instrumentation
* architecture experimentation
* benchmark iteration

over raw throughput.

---

# Current Findings

Early benchmarks suggest that:

* communication overhead dominates quickly
* speculative decoding is not currently beneficial on the CPU backend
* minimizing activation transfer matters more than draft-model complexity
* simpler synchronous pipelines outperform more elaborate speculative paths under current conditions

Example local benchmark results:

| variant  |  TPS | avg ms/token |
| -------- | ---: | -----------: |
| no-spec  | 7.27 |        137.6 |
| spec k=2 | 6.53 |        153.2 |
| spec k=6 | 5.54 |        180.4 |

DigitalOcean WAN benchmarks showed the same trend:

| variant  |  TPS |
| -------- | ---: |
| no-spec  | 3.15 |
| spec k=2 | 2.77 |

Current direction:

* focus on transport efficiency
* improve compression strategies
* explore asynchronous execution
* benchmark scaling across additional nodes

rather than increasing speculative complexity.

---

# Architecture

Relay splits transformer layers across multiple machines.

Example two-stage layout:

```text
Node 0 (Stage 0)
────────────────────────────
embed_tokens
layers [0 … N/2]

           │
           │ hidden states
           ▼

Node 1 (Stage 1)
────────────────────────────
layers [N/2 … N]
RMSNorm
LM head
```

The pipeline currently uses:

* KV-cached decoding
* int8 hidden-state compression
* ZeroMQ sockets over WireGuard
* token-by-token autoregressive generation

---

# Network Topology

```text
┌────────────────────────────────────────────┐
│ Stage 0                                   │
│                                            │
│ Forward pass through local layers          │
│ Compress hidden states (int8)              │
│ Send activations over WAN                  │
└──────────────────────┬─────────────────────┘
                       │
                       │ WireGuard VPN
                       │ ZeroMQ transport
                       ▼
┌────────────────────────────────────────────┐
│ Stage 1                                   │
│                                            │
│ Complete forward pass                      │
│ Generate logits                            │
│ Sample next token                          │
│ Return token to Stage 0                    │
└────────────────────────────────────────────┘
```

Relay intentionally uses real inter-machine networking rather than localhost simulation to expose:

* latency
* serialization cost
* bandwidth limitations
* packet overhead
* synchronization behavior

that disappear in single-machine experiments.

---

# Why DigitalOcean?

DigitalOcean droplets act as a reproducible WAN testbed.

They are not the end goal.

The real target is:

* geographically separate machines
* different ISPs
* different hardware
* unreliable consumer networking

Cloud VMs simply make:

* deployment
* benchmarking
* iteration
* instrumentation

easier during development.

WireGuard provides the encrypted overlay network.
Relay runs on top using ZeroMQ transport.

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
├── config.yaml
├── config.smoke.yaml
├── config.nospec.yaml
└── requirements.txt
```

---

# Quick Start

## Prerequisites

```bash
brew install doctl wireguard-tools

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

---

# Configuration

```yaml
pipeline:
  num_stages: 2

model:
  name: "unsloth/Llama-3.2-3B"
  arch: "llama"
  dtype: "bfloat16"

compression:
  mode: "int8"

speculative:
  enabled: false
  k: 1

cascade:
  enabled: false
```

---

# Research Directions

Current active areas:

* activation compression
* WAN-aware pipeline scheduling
* asynchronous execution
* heterogeneous stage balancing
* transport efficiency
* pipeline backpressure handling
* dynamic topology experimentation

Experimental areas:

* speculative decoding
* cascade verification
* speculative prefetch

---

# Non-Goals

Relay is currently **not**:

* a production inference framework
* a decentralized AI marketplace
* a blockchain project
* a high-throughput serving system
* a replacement for datacenter inference

The current focus is systems research and empirical measurement.

---

# Long-Term Vision

Relay explores a broader possibility:

> Large models may eventually run cooperatively across globally distributed commodity hardware instead of only centralized clusters.

Whether that becomes practical depends almost entirely on:

* communication efficiency
* scheduling
* transport overhead
* compression
* and distributed systems design.

Relay exists to experimentally investigate those limits.


<p align="center">
  Built on cheap hardware, open networks, and stubborn optimism.
</p>
