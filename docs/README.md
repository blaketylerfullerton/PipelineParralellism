# Relay Runtime Notes

Relay is a secure cooperative inference runtime for explicitly trusted stages.
It splits an LLM into deterministic runtime operations across authenticated
machines, then studies how scheduling, KV-cache control, speculative decoding,
and topology-aware routing behave over WAN links.

> Current state: the Python reference path runs Llama 3.2-3B slices with KV cache,
> int8 activation compression, authenticated message envelopes, speculative
> decoding, and optional cascade experiments. May 2026 benchmarks showed the
> no-speculative path winning locally and on DigitalOcean, so `config.yaml`
> defaults to no-spec.

---

## What Is Cooperative Pipeline Inference?

GPT-2 is a stack of 12 identical **transformer blocks** (layers). Normally one machine runs all 12 in sequence. Pipeline parallelism splits those layers across machines:

```
Mac (Stage 0)          Server (Stage 1)
─────────────────      ─────────────────
Embedding layer        Blocks 6-11
Blocks 0-5             Final LayerNorm
                       LM Head (vocab)
```

Each trusted stage only loads its assigned weight slice. A stage receives hidden
states, executes its local forward pass, updates its KV cache, and forwards the
next hidden state. It does not receive arbitrary code or peer-supplied tasks.

The real reason this matters in production is **model size**. GPT-2 is tiny (117M parameters, ~500MB). But:

- GPT-3: 175B parameters → ~350GB to store in float16
- GPT-4: estimated 1.8T parameters

No single GPU can hold those. Cooperative pipeline inference lets you spread
layers across trusted machines so the model fits in memory. Each device only
holds its slice of layers.

---

## How a Generation Step Works

When you type a prompt like `"ai is the beginning of"` and ask for 5 new tokens, here is what happens for one generation step:

```
Mac (Stage 0)                          Server (Stage 1)
──────────────────────────────────     ──────────────────────────────────
1. Tokenize prompt
   → [id, id, id, id, id]

2. Embed tokens
   → tensor (1, 5, 768)

3. Run transformer blocks 0–5

4. Send hidden states ─────────────▶  5. Receive tensor (1, 5, 768)

                                       6. Run transformer blocks 6–11

                                       7. LM head → logits (1, 5, 50257)

                                       8. Sample next token → "everything"

9. Receive token ◀─────────────────   9. Send token back

10. Append to sequence
    → [id, id, id, id, id, NEW]

11. Repeat for next token...
```

The `(1, 5, 768)` tensor is the **hidden state** — a 768-dimensional learned representation of every token in the sequence. It is what travels across the network between machines.

---

## Why the First Token Is Slower

The first step processes the full prompt (e.g. 5 tokens). Every step after that adds one new token, so the sequence grows by 1 each time. Steps 1–N are much faster than step 0 because they do less work. In a production system you would use a **KV cache** to avoid recomputing all previous positions — this implementation re-runs the full sequence from scratch each step to keep the code simple and readable.

---

## The Network Layer

Hidden states travel over **ZeroMQ PUSH/PULL sockets**:

- Stage 0 has a **PUSH socket** → connects to `stage1-ip:5551`
- Stage 1 has a **PULL socket** → bound to port 5551, waits for data

For the token return path (Stage 1 → Stage 0), a separate socket pair runs on port 5556.

Runtime messages use a constrained JSON envelope with binary tensor fields and an
HMAC signature from the configured cluster token. Tensors are still stored as raw
numpy bytes with shape and dtype metadata, but the wire format is not a Python
object deserialization channel.

---

## Project Structure

```
PipeLineParralel/
├── src/
│   ├── launch.py       # trusted-peer launcher
│   ├── worker.py       # generation loop logic (one instance per machine)
│   ├── model.py        # splits GPT-2 into stage slices (Stage0Module, LastModule)
│   ├── utils.py        # ZMQ socket factories + tensor send/recv
│   └── dashboard.py    # live Rich terminal display
├── config.yaml         # pipeline config (stages, ports, model name)
└── requirements.txt
```

### How the pieces connect

```
src/launch.py
  └── DiscoveryManager   optional authenticated lab discovery
  └── src/worker.py      runs the generation loop for this stage
       └── src/model.py      loads only this stage's slice of GPT-2
       └── src/utils.py      handles all network communication
       └── src/dashboard.py  shows live status in the terminal
```

---

## Setup

Requires Python 3.10+ on each machine.

```bash
git clone https://github.com/blaketylerfullerton/PipelineParralellism
cd PipeLineParralel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running It

### Same machine (subprocess smoke test)

```bash
./deploy/run_local.py --config config.smoke.yaml --prompt "hello from localhost"
```

This starts every stage on `127.0.0.1`, writes logs under `logs/local-runs/`, and avoids tmux and DigitalOcean entirely.

For a local performance matrix:

```bash
./deploy/benchmark_matrix.py --base-config config.yaml --prompt "hello from localhost" --ks 1,2,4,6
```

Use `config.nospec.yaml` for a direct no-speculative baseline. The most recent local matrix:

| variant | TPS | avg ms/token |
|---|---:|---:|
| no-spec | 7.27 | 137.6 |
| spec k=1 | 6.00 | 166.7 |
| spec k=2 | 6.53 | 153.2 |
| spec k=4 | 6.18 | 161.7 |
| spec k=6 | 5.54 | 180.4 |

DigitalOcean showed the same conclusion: no-spec reached 3.15 TPS; spec `k=2` reached 2.77 TPS.

### Same machine (manual terminals)

```bash
# Terminal 2
python src/launch.py --stage 1 --stages 2 --peer-ip 127.0.0.1 127.0.0.1

# Terminal 1 (start last — this one asks for the prompt)
python src/launch.py --stage 0 --stages 2 --peer-ip 127.0.0.1 127.0.0.1
```

### Optional lab discovery

Manual `--peer-ip` is the default for trusted-peer runs. Authenticated UDP
discovery is retained for lab testing when all stages already share the cluster
token in config.

```bash
# Machine B (server — start first)
python src/launch.py --stage 1 --stages 2 --allow-discovery

# Machine A (driver — start second, will ask for prompt)
python src/launch.py --stage 0 --stages 2 --allow-discovery
```

The discovery panel lights up green when each authenticated stage comes online.

### Two machines on different networks (manual IPs)

If the machines are on different subnets or connected via a VPN like Tailscale,
use `--peer-ip` and provide IPs directly.

Get each machine's IP first:
```bash
# On each machine
tailscale ip -4        # if using Tailscale
# or
hostname -I            # local LAN IP
```

Then on both machines, provide all IPs in stage order:
```bash
# Server (Stage 1) — run first
python src/launch.py --stage 1 --stages 2 --peer-ip <stage0-ip> <stage1-ip>

# Mac (Stage 0) — run second
python src/launch.py --stage 0 --stages 2 --peer-ip <stage0-ip> <stage1-ip>
```

Example with Tailscale:
```bash
python src/launch.py --stage 1 --stages 2 --peer-ip 100.76.11.94 100.76.230.67
python src/launch.py --stage 0 --stages 2 --peer-ip 100.76.11.94 100.76.230.67
```

### 3 machines

```bash
# Machine C (stage 2)
python src/launch.py --stage 2 --stages 3 --peer-ip <ip0> <ip1> <ip2>

# Machine B (stage 1)
python src/launch.py --stage 1 --stages 3 --peer-ip <ip0> <ip1> <ip2>

# Machine A (stage 0, driver)
python src/launch.py --stage 0 --stages 3 --peer-ip <ip0> <ip1> <ip2>
```

---

## Configuration

Edit `config.yaml` to change model or generation settings:

```yaml
pipeline:
  num_stages: 2         # how many machines

model:
  name: "unsloth/Llama-3.2-3B"
  arch: "llama"
  max_new_tokens: 30    # tokens to generate
  temperature: 1.0      # sampling temperature

network:
  base_port: 5550       # all ports derived from this

speculative:
  enabled: false        # no-spec is the measured default
```

### Port reference

With `base_port: 5550` and `num_stages: 2`:

| Purpose | Ports |
|---|---|
| Forward (hidden states) | 5550, 5551 |
| Telemetry (dashboard) | 5554, 5555 |
| Token return | 5556 |

Open these on any machine with a firewall:
```bash
sudo ufw allow 5550:5560/tcp
```

---

## What Each Log Line Means

```
[Stage 1] Step   0  shape=(1, 5, 768)  624.6ms
```

- `Step 0` — first token being generated
- `shape=(1, 5, 768)` — batch of 1, sequence length 5, hidden dim 768
- `624.6ms` — time this stage spent on its forward pass

```
[Stage 0] Generating:  the future of
```

Stage 0 is printing each token as it comes back from Stage 1, one by one.
