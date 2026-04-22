# Pipeline Parallelism

A from-scratch implementation of pipeline parallelism for LLM inference across multiple machines. Split GPT-2's layers across two or more computers, send a prompt, and watch each machine process its slice of the model in real time.

---

## What Is Pipeline Parallelism?

GPT-2 is a stack of 12 identical **transformer blocks** (layers). Normally one machine runs all 12 in sequence. Pipeline parallelism splits those layers across machines:

```
Mac (Stage 0)          Server (Stage 1)
─────────────────      ─────────────────
Embedding layer        Blocks 6-11
Blocks 0-5             Final LayerNorm
                       LM Head (vocab)
```

Each machine only ever loads **half the weights**. Neither machine has the full model in memory at once.

The real reason this matters in production is **model size**. GPT-2 is tiny (117M parameters, ~500MB). But:

- GPT-3: 175B parameters → ~350GB to store in float16
- GPT-4: estimated 1.8T parameters

No single GPU can hold those. Pipeline parallelism lets you spread layers across many GPUs or machines so the model fits in memory. Each device only holds its slice of layers.

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

Tensors are serialized as raw numpy bytes — `tensor.numpy().tobytes()` with shape and dtype stored alongside in a pickle dict. No heavy serialization library, so you can inspect every message clearly in the logs.

---

## Project Structure

```
PipeLineParralel/
├── launch.py       # auto-discovery + pipeline launcher (start here)
├── worker.py       # generation loop logic (one instance per machine)
├── model.py        # splits GPT-2 into stage slices (Stage0Module, LastModule)
├── utils.py        # ZMQ socket factories + tensor send/recv
├── dashboard.py    # live Rich terminal display
├── config.yaml     # pipeline config (stages, ports, model name)
└── requirements.txt
```

### How the pieces connect

```
launch.py
  └── DiscoveryManager   discovers peers via UDP broadcast
  └── worker.py          runs the generation loop for this stage
       └── model.py      loads only this stage's slice of GPT-2
       └── utils.py      handles all network communication
       └── dashboard.py  shows live status in the terminal
```

---

## Setup

Requires Python 3.10+ on each machine.

```bash
git clone <repo-url>
cd PipeLineParralel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running It

### Same machine (3 terminals, for testing)

```bash
# Terminal 2
python launch.py --stage 1 --stages 2

# Terminal 1 (start last — this one asks for the prompt)
python launch.py --stage 0 --stages 2
```

### Two machines on the same network (auto-discovery)

Both machines must be on the same subnet for UDP broadcast to work.

```bash
# Machine B (server — start first)
python launch.py --stage 1 --stages 2

# Machine A (driver — start second, will ask for prompt)
python launch.py --stage 0 --stages 2
```

The discovery panel lights up green when each machine comes online.

### Two machines on different networks (manual IPs)

If the machines are on different subnets or connected via a VPN like Tailscale, use `--peer-ip` to skip auto-discovery and provide IPs directly.

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
python launch.py --stage 1 --stages 2 --peer-ip <stage0-ip> <stage1-ip>

# Mac (Stage 0) — run second
python launch.py --stage 0 --stages 2 --peer-ip <stage0-ip> <stage1-ip>
```

Example with Tailscale:
```bash
python launch.py --stage 1 --stages 2 --peer-ip 100.76.11.94 100.76.230.67
python launch.py --stage 0 --stages 2 --peer-ip 100.76.11.94 100.76.230.67
```

### 3 machines

```bash
# Machine C (stage 2)
python launch.py --stage 2 --stages 3 --peer-ip <ip0> <ip1> <ip2>

# Machine B (stage 1)
python launch.py --stage 1 --stages 3 --peer-ip <ip0> <ip1> <ip2>

# Machine A (stage 0, driver)
python launch.py --stage 0 --stages 3 --peer-ip <ip0> <ip1> <ip2>
```

---

## Configuration

Edit `config.yaml` to change model or generation settings:

```yaml
pipeline:
  num_stages: 2         # how many machines

model:
  name: "gpt2"          # gpt2 / gpt2-medium / gpt2-large
  max_new_tokens: 30    # tokens to generate
  temperature: 1.0      # sampling temperature

network:
  base_port: 5550       # all ports derived from this
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
