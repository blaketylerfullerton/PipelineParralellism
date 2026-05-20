# Architecture Audit: Trusted WAN Inference Runtime

Relay is being realigned around this research position:

> A WAN-native runtime architecture for cooperative distributed LLM inference across trusted peers.

## Current Architecture

The repository contains a Python reference runtime for pipeline-parallel LLM
inference:

- `src/launch.py` starts one assigned stage per machine, loads config, resolves
  peers, loads the stage model slice, and enters the worker loop.
- `src/worker.py` implements the deterministic runtime loop: prefill, stage-local
  forward pass, activation forwarding, token return, KV-cache trim/reset, optional
  speculative decoding, and optional cascade.
- `src/model.py` slices GPT-2 or Llama into fixed stage modules.
- `src/draft.py` runs the local draft model used by speculative decoding.
- `src/utils.py` owns tensor encoding, authenticated message serialization, ZMQ
  sockets, and config loading.
- `deploy/` provides local and cloud benchmark harnesses.

The hot path is inference-specific rather than arbitrary compute: peers exchange
hidden states, sampled-token metadata, KV-cache controls, and telemetry.

```text
Trusted control plane
  config.yaml / explicit peer list / cluster auth token
        │
        ▼
Stage 0 ── activation + KV control ──▶ Stage 1 ── ... ──▶ Stage N
  ▲                                                        │
  └──────── speculative result / sampled token ────────────┘
```

## Conflicts With The New Direction

The original language over-emphasized open networking and under-emphasized trust.
Documentation now needs to consistently frame Relay as a trusted cooperative
cluster, not an open peer fabric.

Architectural conflicts found:

- Peer discovery previously ran by default and accepted unsigned UDP beacons.
- Runtime messages previously used `pickle`, which is unsafe for untrusted input.
- Dashboard telemetry also used `pickle`.
- Config did not explicitly describe a cluster trust boundary.
- Documentation described auto-discovery as the normal multi-machine flow.
- Some text centered broadly reachable networking rather than authenticated
  private WAN participation.

No direct remote shell execution, dynamic Python execution, peer-provided plugin
execution, dynamic container execution, or `eval`/`exec` runtime path was found in
`src/`. Deployment scripts do use SSH/subprocess locally to provision owned test
machines; that is an operator deployment surface, not a peer execution protocol.

## Security Review

Primary risks:

- **Unsafe serialization:** `pickle.loads` on ZMQ traffic could execute attacker
  controlled Python objects if an attacker reached a runtime socket.
- **Unauthenticated peer admission:** unsigned discovery could allow a host on the
  same network or tailnet to claim a stage.
- **Unrestricted bind addresses:** ZMQ sockets bind to `tcp://*`, so network
  controls and message authentication matter.
- **Weak membership model:** peers were identified by stage index and IP but not
  authenticated at the runtime protocol layer.
- **Telemetry trust:** dashboard messages were accepted without authentication.

Implemented hardening:

- Replaced pickle runtime messages with a constrained JSON envelope in
  `src/utils.py`.
- Added HMAC-SHA256 authentication when `trust.auth_token` is configured.
- Added explicit `trust.cluster_id` and `trust.auth_token` config sections.
- Reused the authenticated envelope for telemetry and lab discovery.
- Made launch require explicit `--peer-ip` by default.
- Moved UDP discovery behind `--allow-discovery`.

Remaining hardening priorities:

1. Replace shared-token HMAC with per-node identities: mTLS, WireGuard public keys,
   or signed node certificates mapped to stage assignments.
2. Bind sockets to configured private interfaces instead of `tcp://*`.
3. Add replay protection with monotonic message sequence numbers per stage link.
4. Add payload size limits before base64 decode and tensor allocation.
5. Enforce config-level stage identity: expected node id, allowed IP/CIDR, and
   model slice checksum.
6. Encrypt at the transport layer for all WAN runs, even when HMAC is enabled.

## Target Runtime Model

Relay should expose a fixed operation vocabulary:

- `activation.forward`: send encoded hidden state to the next stage.
- `kv.trim`: trim stage-local KV cache to an accepted sequence length.
- `kv.reset_prefill`: reset cache at prompt prefill boundaries.
- `spec.verify`: return target probabilities and sampled fallback tokens.
- `route.update`: update stage placement/topology observations.
- `telemetry.stage`: report latency, queueing, step, and failure state.

Relay should not expose:

- remote shell commands
- peer-supplied scripts
- arbitrary Python functions
- dynamic containers
- unbounded plugin hooks
- general compute jobs

## Refactor Plan

Priority 0, completed in this audit:

- Remove unsafe pickle deserialization from runtime and dashboard paths.
- Add authenticated message envelopes.
- Make peer discovery opt-in and authenticated.
- Add trust configuration to checked-in configs.
- Rewrite README/docs language toward secure cooperative inference.

Priority 1:

- Introduce a typed protocol module with message dataclasses and schema tests.
- Add bounded tensor shape validation per operation and per stage.
- Add per-link sequence numbers and reject stale/replayed control messages.
- Split discovery from launch into a trusted membership module.
- Add a topology table: stage id, node id, private endpoint, RTT estimate,
  bandwidth estimate, device class, and availability.

Priority 2:

- Add a topology-aware scheduler for stage placement and speculative `k`.
- Add WAN fault policies: timeout, retry, rejoin, cache invalidation, and stage
  drain.
- Move deployment credentials and model tokens entirely out of generated scripts.
- Define an enterprise-compatible node bootstrap flow.

Priority 3:

- Move hot kernels to llama.cpp while preserving the fixed Relay operation model.
- Evaluate tree speculation, Medusa/EAGLE-style branches, and WAN-aware verifier
  batching.
- Explore expert/layer placement under asymmetric bandwidth and compute.

## Updated Terminology

Use:

- trusted cooperative peers
- secure distributed inference fabric
- WAN-native inference runtime
- topology-aware execution engine
- authenticated stage membership
- deterministic runtime operations
- cooperative AI cluster

Avoid:

- open admission peer network
- public admission compute network
- arbitrary peer execution
- peer-defined compute tasks
- token economics or governance framing

## Future Research Directions

- WAN-aware speculative decoding where draft depth adapts to measured RTT.
- Topology-aware pipeline routing with asymmetric uplinks and heterogeneous CPUs.
- KV-cache consistency protocols under intermittent node availability.
- Stage placement heuristics for mixed cloud, edge, and personal workstation
  clusters.
- Failure-mode benchmarks: packet loss, jitter bursts, slow receivers, node
  pause/resume, and route changes.
- Secure membership bootstrapping with per-node attestable identities.
