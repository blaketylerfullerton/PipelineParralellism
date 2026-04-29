<p align="center"><img src="images/networking1.png" width="600"/></p>


# Relay

> **Research project.** The goal is to decentralize AI compute — running large models cooperatively across ordinary consumer machines connected over the open internet, with no shared memory, no datacenter fabric, and no specialized hardware. This is an attempt to push the limits of what WAN inference can actually do.

Most distributed ML research assumes you have a high-bandwidth, low-latency interconnect (NVLink, InfiniBand, a datacenter LAN). Relay deliberately doesn't. The bet is that with the right combination of pipeline parallelism, speculative decoding, aggressive compression, and smarter local caching, you can make inference across geographically separate commodity machines practical — not just possible.

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

Stage 0 also runs a smaller **draft model** (Llama 3.2-1B) for speculative decoding. The draft proposes `k` tokens ahead; Stage 1 verifies them in one forward pass. A **cascade** layer adds a local fast-path: if the draft's top-1 probability clears a threshold, Stage 0 accepts the token immediately without sending anything to Stage 1.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Stage 0                                                 │
│                                                          │
│  Draft model (Llama 1B) ──► speculate k tokens           │
│  Target Stage 0 layers   ──► produce hidden states       │
│                                                          │
│  Cascade check: if draft confidence ≥ 0.9                │
│    → accept token locally, skip network round-trip       │
│                                                          │
│  Otherwise → compress (int8) & push over WireGuard ──┐  │
└──────────────────────────────────────────────────────│──┘
                                                        │
                  WireGuard VPN (10.99.0.0/24)          │
                                                        ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1                                                 │
│                                                          │
│  Target Stage 1 layers   ──► logits                      │
│  Accept/reject draft tokens, sample bonus token          │
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
│   ├── deploy.sh       # spin up two DO droplets, wire WireGuard, bootstrap Relay
│   └── teardown.sh     # delete both droplets
├── config.yaml         # model, pipeline, network, and speculative settings
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
cd deploy
./deploy.sh
```

This will:
1. Generate WireGuard keypairs locally
2. Create two `s-8vcpu-16gb` Ubuntu droplets
3. Each droplet clones the repo, creates a venv, and installs dependencies
4. WireGuard is configured and started on both droplets
5. Model weights are downloaded in the background
6. A tmux session opens with both droplets side by side (requires tmux)

### Run

Once the model download finishes on both droplets (check with `tail -f /var/log/pipeline-models.log`), start Relay. **Stage 1 must start first.**

```bash
# tmux pane — Node 1 (Stage 1)
/opt/pipeline/start.sh

# tmux pane — Node 0 (Stage 0)
/opt/pipeline/start.sh --prompt "the future of computing is"
```

### Tear down

```bash
./deploy/teardown.sh
```

---

## Local Testing (single machine)

Run both stages on localhost in two terminals:

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
  enabled: true
  k: 6                      # draft tokens per step
  draft_model: "unsloth/Llama-3.2-1B"

cascade:
  enabled: true
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

## How Speculative Decoding Works Here

Standard autoregressive decoding sends one token at a time through the full pipeline — slow over WAN, because each token is a full network round-trip. Relay uses speculative decoding to draft `k` tokens with the cheap 1B model, then sends them all to Stage 1 at once. Stage 1 runs one forward pass over all `k+1` positions and either accepts each draft token (if the target agrees) or rejects and resamples from that point. On a typical good-confidence step you get 3–5 tokens accepted per round-trip instead of 1.

The cascade layer short-circuits even further: if the draft's confidence is high enough (≥ 0.9 top-1 probability), Stage 0 accepts the token locally and buffers the hidden states. These get flushed in a batch on the next network send, so Stage 1 stays in sync without a per-token round-trip.

End-of-run stats show acceptance rate and cascade hit rate:

```
[Stage 0] Spec:    avg 3.8 tokens/step  (k=6, acceptance=54%)
[Stage 0] Cascade: 12/31 steps handled locally (39%)
```
