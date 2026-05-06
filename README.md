<p align="center"><img src="images/networking1.png" width="300"/></p>


# Relay

> **Research project.** The goal is to decentralize AI compute — running large models cooperatively across ordinary consumer machines connected over the open internet, with no shared memory, no datacenter fabric, and no specialized hardware. This is an attempt to push the limits of what WAN inference can actually do.

Most distributed ML research assumes you have a high-bandwidth, low-latency interconnect (NVLink, InfiniBand, a datacenter LAN). Relay deliberately doesn't. The bet is that with the right combination of pipeline parallelism, KV caching, compression, and fast CPU inference backends, you can make inference across geographically separate commodity machines practical — not just possible.

DigitalOcean droplets over WireGuard stand in for the real target: two people's laptops, a laptop and a spare desktop, a phone and a friend's computer. The cloud VMs are just a reproducible way to develop and benchmark the networking layer before the hardware is in hand.

<p align="center"><img src="images/diagram.png" width="700"/></p>

---

## What It Does

A single Llama 3.2-3B model is too large to run comfortably on one small machine. Relay solves this by splitting the transformer layers across machines — each node loads only its slice, and they chain together:

```
Node 0 (Stage 0)                 Node 1 (Stage 1)
──────────────────────           ──────────────────────
embed_tokens                     layers [N/2 … N]
layers [0 … N/2]                 RMSNorm
                                 LM head
```

The default path is now the simplest one: KV-cached pipeline-parallel decoding with int8 activation compression over the wire. Speculative decoding, cascade, and prefetch remain in the codebase as experimental modes, but local and DigitalOcean benchmarks both showed that the no-speculative baseline is faster for the current PyTorch CPU backend.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Stage 0                                                 │
│                                                          │
│  Target Stage 0 layers   ──► produce hidden states       │
│                                                          │
│  Compress hidden states (int8) and push over network ─┐ │
└──────────────────────────────────────────────────────│──┘
                                                        │
                  WireGuard VPN (10.99.0.0/24)          │
                                                        ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1                                                 │
│                                                          │
│  Target Stage 1 layers   ──► logits                      │
│  Sample next token                                       │
│  Send result back to Stage 0 ◄──────────────────────    │
└──────────────────────────────────────────────────────────┘
```

### Why DigitalOcean as a mock network

Real distributed inference runs across machines on different subnets with real network latency, firewall rules, and no shared memory. Running it locally on one machine skips all of that. Two DO droplets on the same region but different VMs give genuine inter-machine communication — bandwidth limits, real TCP round-trips, port routing — while staying cheap and reproducible. WireGuard creates the encrypted overlay network; Relay runs on top of it over ZeroMQ sockets.

---

## Project Structure

```
relay/
├── src/
│   ├── launch.py       # peer discovery + pipeline launcher (entry point)
│   ├── worker.py       # per-stage generation loop (speculative + cascade logic)
│   ├── model.py        # splits Llama/GPT-2 into stage slices
│   ├── draft.py        # draft model wrapper for speculative decoding
│   ├── utils.py        # ZMQ socket helpers, tensor send/recv, int8 codec
│   └── dashboard.py    # live Rich terminal display
├── deploy/
│   ├── relayctl.py     # status, doctor checks, logs, provision/run wrappers
│   ├── deploy.sh       # spin up N DO droplets, wire WireGuard, bootstrap Relay
│   ├── run.sh          # start an existing deployment in tmux
│   ├── run_local.py    # local subprocess runner, no tmux or cloud
│   ├── benchmark_matrix.py # local no-spec / k-value benchmark runner
│   └── teardown.sh     # delete deployed droplets
├── config.yaml         # real model, pipeline, network, speculative settings
├── config.smoke.yaml   # tiny local smoke-test config
├── config.nospec.yaml  # real model without speculative decoding
└── requirements.txt
```

---

## Quick Start — DigitalOcean Deploy

### Prerequisites

```bash
brew install doctl wireguard-tools
doctl auth init          # paste your DO token
```

### Configure

Create `deploy/.env`:

```bash
DO_TOKEN=your_digitalocean_token
GITHUB_REPO=youruser/relay
DO_SSH_KEY=your_ssh_key_fingerprint   # doctl compute ssh-key list
HF_TOKEN=your_hf_token                # only needed for gated models
REGION=sfo3
```

### Deploy

```bash
./deploy/relayctl.py provision
# or: ./deploy/deploy.sh
```

The droplet bootstrap clones the current Git branch from `GITHUB_REPO`. Push the branch before provisioning, or set `GIT_BRANCH=main` in `deploy/.env` when you want to deploy a known remote branch.

This will:
1. Generate WireGuard keypairs locally
2. Create `pipeline.num_stages` Ubuntu droplets unless `NUM_STAGES` is set
3. Each droplet clones the repo, creates a venv, and installs dependencies
4. WireGuard is configured and started across the full mesh
5. Model weights are downloaded in the background
6. Runtime state is written to ignored `deploy/.deploy-state` and `deploy/state.json`

Before running the model, check the deployment:

```bash
./deploy/relayctl.py status
./deploy/relayctl.py doctor
./deploy/relayctl.py logs --kind models
```

### Run

Once the model download finishes on all droplets, start Relay:

```bash
./deploy/relayctl.py run --prompt "the future of computing is"
# or: ./deploy/run.sh --prompt "the future of computing is"
```

`run.sh` starts worker stages first, then starts Stage 0 after a short delay. It builds the tmux layout dynamically from deploy state, so two-stage and four-stage deployments use the same entry point.

To check the remote commands without opening tmux or SSH sessions:

```bash
./deploy/relayctl.py run --prompt "the future of computing is" --dry-run
```

### Tear down

```bash
./deploy/relayctl.py teardown
# or: ./deploy/teardown.sh
```

---

## Local Testing (single machine)

Fast smoke test with the tiny GPT-2 config:

```bash
./deploy/run_local.py --config config.smoke.yaml --prompt "hello from localhost"
```

This starts all stages as subprocesses on `127.0.0.1`, waits for Stage 0 to finish, and writes per-stage logs under `logs/local-runs/`.

Baseline without speculative decoding:

```bash
./deploy/run_local.py --config config.nospec.yaml --prompt "the future of computing is" --timeout 900
```

Local benchmark matrix:

```bash
./deploy/benchmark_matrix.py --base-config config.yaml --prompt "the future of computing is" --ks 1,2,4,6 --max-new-tokens 30
```

By default the matrix disables cascade and prefetch for the speculative variants so `k` is the only moving part. Add `--cascade` or `--prefetch` when you explicitly want those included.

### Current benchmark findings

The latest local matrix on a MacBook Air showed the no-speculative baseline winning:

| variant | TPS | avg ms/token | note |
|---|---:|---:|---|
| no-spec | 7.27 | 137.6 | fastest local path |
| spec k=1 | 6.00 | 166.7 | slower |
| spec k=2 | 6.53 | 153.2 | best speculative local run, still slower |
| spec k=4 | 6.18 | 161.7 | slower |
| spec k=6 | 5.54 | 180.4 | much slower |

DigitalOcean showed the same direction:

| variant | TPS | result |
|---|---:|---|
| no-spec | 3.15 | fastest DO path |
| spec k=2 | 2.77 | ~12% slower |

Conclusion: `config.yaml` defaults to no-speculative decoding. The next useful performance work is quantized weights / a faster CPU backend, not more speculative tuning.

For manual terminal testing, run both stages on localhost:

```bash
# Terminal 2 — Stage 1 first
python src/launch.py --stage 1 --stages 2

# Terminal 1 — Stage 0, enter prompt when ready
python src/launch.py --stage 0 --stages 2
```

Stages find each other via UDP broadcast. For manual IP assignment (cross-subnet or VPN):

```bash
python src/launch.py --stage 1 --stages 2 --peer-ip <ip0> <ip1>
python src/launch.py --stage 0 --stages 2 --peer-ip <ip0> <ip1>
```

---

## Configuration

`config.yaml` controls everything:

```yaml
pipeline:
  num_stages: 2

model:
  name: "unsloth/Llama-3.2-3B"
  arch: "llama"
  dtype: "bfloat16"         # fp32 | fp16 | bf16
  max_new_tokens: 30
  temperature: 1.0

compression:
  mode: "int8"              # fp32 | fp16 | int8 — applied to hidden states in transit

speculative:
  enabled: false            # speculative decoding is experimental; no-spec is the default baseline
  k: 1                      # ignored when speculative.enabled is false
  draft_model: "unsloth/Llama-3.2-1B"
  pipeline_prefetch: false  # opt-in; usually misses unless the next seed is predictable

cascade:
  enabled: false
  confidence_threshold: 0.9 # accept draft locally if top-1 prob ≥ this
```

### Port reference

With `base_port: 5550` and `num_stages: 2`:

| Purpose | Port |
|---|---|
| Hidden states (Stage 0 → 1) | 5551 |
| Token return (Stage 1 → 0) | 5556 |
| Telemetry | 5554–5555 |

Open these if running with a firewall:
```bash
sudo ufw allow 5550:5560/tcp
```

---

## Experimental Speculative Decoding

Standard autoregressive decoding sends one token at a time through the full pipeline. Relay also has an experimental speculative path: Stage 0 drafts `k` tokens with a 1B model, sends all candidates to Stage 1, and Stage 1 verifies them in one forward pass. Stage 0 accepts the longest valid prefix, rejects when the target disagrees, and samples from the target distribution.

This path is useful research scaffolding, but it is not the current default. With the current PyTorch CPU backend, the draft model costs more than it earns back in accepted tokens. The best observed speculative setting was `k=2`, and it still lost to no-spec locally and on DigitalOcean.

The cascade layer can short-circuit high-confidence draft tokens locally. Speculative prefetch can try to draft the next round during verifier wait time. Both are disabled by default and should be re-enabled only when benchmark data shows a win.

End-of-run stats show draft acceptance, rejection positions, cascade hit rate, and prefetch status:

```
[Stage 0] Spec:     avg 2.42 output tokens/step (k=2, draft_accept=70.8%, full=6/12)
[Stage 0] Rejects:  pos0=3, pos1=3
[Stage 0] Cascade: 12/31 steps handled locally (39%)
[Stage 0] Pipeline: prefetch disabled
```
