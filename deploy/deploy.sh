#!/usr/bin/env bash
# deploy.sh — spin up two DO droplets, wire them with WireGuard, bootstrap the pipeline
#
# Required env vars:
#   DO_TOKEN       DigitalOcean API token
#   GITHUB_REPO    owner/repo  (e.g. youruser/PipeLineParralel)
#
# Optional env vars:
#   HF_TOKEN       HuggingFace token (needed if models are gated)
#   DO_SSH_KEY     Fingerprint or name of your DO SSH key
#                  (run: doctl compute ssh-key list)
#   REGION         Defaults to nyc3
#   SIZE           Defaults to s-4vcpu-8gb
set -euo pipefail

# ── output helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${BOLD}▶ $*${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
err()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

# ── load .env if present ─────────────────────────────────────────────────────
ENV_FILE="$(dirname "$0")/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
fi

# ── prerequisites ────────────────────────────────────────────────────────────
for cmd in doctl wg; do
    command -v "$cmd" &>/dev/null || err "$cmd not found. Install it first."
done

: "${DO_TOKEN:?Set DO_TOKEN}"
: "${GITHUB_REPO:?Set GITHUB_REPO (e.g. youruser/PipeLineParralel)}"
HF_TOKEN="${HF_TOKEN:-}"
DO_SSH_KEY="${DO_SSH_KEY:-}"
REGION="${REGION:-sfo3}"
SIZE="${SIZE:-s-8vcpu-16gb}"

export DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN"
IMAGE="ubuntu-24-04-x64"
GIT_BRANCH=$(git -C "$(dirname "$0")/.." rev-parse --abbrev-ref HEAD)
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10"

# ── WireGuard key generation ──────────────────────────────────────────────────
log "Generating WireGuard keypairs..."
PRIV0=$(wg genkey);  PUB0=$(echo "$PRIV0" | wg pubkey)
PRIV1=$(wg genkey);  PUB1=$(echo "$PRIV1" | wg pubkey)

WG0_IP="10.99.0.1"
WG1_IP="10.99.0.2"
WG_PORT=51820

ok "Stage 0 pubkey: $PUB0"
ok "Stage 1 pubkey: $PUB1"

# ── resolve SSH key flag ──────────────────────────────────────────────────────
SSH_FLAG=""
if [[ -n "$DO_SSH_KEY" ]]; then
    SSH_FLAG="--ssh-keys $DO_SSH_KEY"
fi

# ── cloud-init builder ────────────────────────────────────────────────────────
# Args: STAGE  MY_WG_IP  MY_WG_PRIV  PEER_WG_IP  PEER_WG_PUB  PEER_PUBLIC_IP_PLACEHOLDER
make_cloud_init() {
    local stage="$1"
    local my_wg_ip="$2"
    local my_wg_priv="$3"
    local peer_wg_ip="$4"
    local peer_wg_pub="$5"
    # peer public IP is filled in after droplets are created — placeholder token
    # replaced by sed in a second pass (see below)
    local peer_pub_ip_token="PEER_PUBLIC_IP"

    # Base64-encode start.sh content so "$@" survives the heredoc expansion chain
    local _start_b64
    _start_b64=$(printf '#!/bin/bash\ncd /opt/pipeline\nsource .venv/bin/activate\nexec python src/launch.py --stage %s --peer-ip %s %s "$@"\n' \
        "${stage}" "${WG0_IP}" "${WG1_IP}" | base64)

    cat <<CLOUDINIT
#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/pipeline-init.log) 2>&1

echo "[init] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git curl wireguard wireguard-tools build-essential

# Write the WireGuard config FIRST (before the slow pip install) so the deploy
# script's patch step doesn't race the multi-minute torch download. The peer
# public IP is filled in over SSH once both droplets are up.
echo "[init] Writing WireGuard config..."
install -d -m 0700 /etc/wireguard
cat > /etc/wireguard/wg0.conf <<WGCONF
[Interface]
PrivateKey = ${my_wg_priv}
Address = ${my_wg_ip}/24
ListenPort = ${WG_PORT}

[Peer]
PublicKey = ${peer_wg_pub}
AllowedIPs = ${peer_wg_ip}/32
Endpoint = ${peer_pub_ip_token}:${WG_PORT}
PersistentKeepalive = 25
WGCONF
chmod 0600 /etc/wireguard/wg0.conf
echo "[init] WireGuard config written (peer IP will be patched after both droplets are up)"

echo "[init] Cloning repo (branch: ${GIT_BRANCH})..."
git clone --branch "${GIT_BRANCH}" "https://github.com/${GITHUB_REPO}.git" /opt/pipeline
cd /opt/pipeline

echo "[init] Installing Python deps (CPU torch)..."
python3 -m venv /opt/pipeline/.venv
/opt/pipeline/.venv/bin/pip install --quiet --upgrade pip
/opt/pipeline/.venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
/opt/pipeline/.venv/bin/pip install --quiet -r requirements.txt

# Write start.sh from base64 to preserve "$@" through heredoc expansion
printf '%s' "${_start_b64}" | base64 -d > /opt/pipeline/start.sh
chmod +x /opt/pipeline/start.sh

echo "[init] Downloading HuggingFace models (background)..."
cat > /opt/pipeline/download_models.sh <<DLSH
#!/bin/bash
source /opt/pipeline/.venv/bin/activate
export HF_TOKEN="${HF_TOKEN}"
echo "[dl] Downloading draft model unsloth/Llama-3.2-1B..."
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B', token='${HF_TOKEN}' or None); \
    AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-1B', token='${HF_TOKEN}' or None)"
echo "[dl] Downloading target model unsloth/Llama-3.2-3B..."
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('unsloth/Llama-3.2-3B', token='${HF_TOKEN}' or None); \
    AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-3B', token='${HF_TOKEN}' or None)"
echo "[dl] Done."
DLSH
chmod +x /opt/pipeline/download_models.sh
nohup /opt/pipeline/download_models.sh > /var/log/pipeline-models.log 2>&1 &

echo "[init] Bootstrap complete. Model download running in background."
echo "[init] Check progress: tail -f /var/log/pipeline-models.log"
echo "[init] After WG is up, run: /opt/pipeline/start.sh --prompt 'your prompt'"
CLOUDINIT
}

# ── create droplets (in parallel) ─────────────────────────────────────────────
echo ""
log "Creating both droplets in parallel..."
INIT0=$(make_cloud_init 0 "$WG0_IP" "$PRIV0" "$WG1_IP" "$PUB1")
INIT1=$(make_cloud_init 1 "$WG1_IP" "$PRIV1" "$WG0_IP" "$PUB0")

OUT0=$(mktemp); OUT1=$(mktemp)

doctl compute droplet create "pipeline-stage-0" \
    --image "$IMAGE" --size "$SIZE" --region "$REGION" \
    $SSH_FLAG --user-data "$INIT0" --wait --no-header --format ID \
    >"$OUT0" 2>/dev/null &
PID0=$!

doctl compute droplet create "pipeline-stage-1" \
    --image "$IMAGE" --size "$SIZE" --region "$REGION" \
    $SSH_FLAG --user-data "$INIT1" --wait --no-header --format ID \
    >"$OUT1" 2>/dev/null &
PID1=$!

wait $PID0 || err "Stage 0 droplet creation failed"
wait $PID1 || err "Stage 1 droplet creation failed"

DROPLET0_ID=$(tail -1 "$OUT0")
DROPLET1_ID=$(tail -1 "$OUT1")
rm -f "$OUT0" "$OUT1"

ok "Stage 0 droplet ID: $DROPLET0_ID"
ok "Stage 1 droplet ID: $DROPLET1_ID"

# ── fetch public IPs ──────────────────────────────────────────────────────────
echo ""
log "Fetching public IPs..."
IP0=$(doctl compute droplet get "$DROPLET0_ID" --no-header --format PublicIPv4)
IP1=$(doctl compute droplet get "$DROPLET1_ID" --no-header --format PublicIPv4)

ok "Stage 0: $IP0"
ok "Stage 1: $IP1"

# ── patch WireGuard peer IP on each droplet ───────────────────────────────────
# The cloud-init wrote PEER_PUBLIC_IP as a placeholder — patch it over SSH
# once both droplets are up. We poll TCP/22 directly (much faster than full
# ssh attempts) and only spend a real SSH session once the port is open.

# Poll with a real SSH auth probe. On Ubuntu 24.04 the sshd is socket-activated,
# so a TCP/22 probe returns success the instant the kernel is up — long before
# cloud-init has written /root/.ssh/authorized_keys or finished apt-get install.
# A `ssh ... true` round-trip only succeeds once auth and sshd are truly ready.
wait_for_ssh() {
    local host="$1"
    local timeout="${2:-300}"
    local elapsed=0
    while (( elapsed < timeout )); do
        if ssh $SSH_OPTS -o BatchMode=yes -o ConnectTimeout=8 \
              root@"$host" true 2>/dev/null; then
            return 0
        fi
        sleep 5
        elapsed=$((elapsed+5))
    done
    return 1
}

patch_wg() {
    local host="$1"
    local peer_ip="$2"
    if ! wait_for_ssh "$host" 300; then
        warn "SSH never came up on $host within 5 min. Patch manually:"
        echo "    ssh root@$host \"sed -i 's/PEER_PUBLIC_IP/${peer_ip}/' /etc/wireguard/wg0.conf && systemctl enable --now wg-quick@wg0\""
        return 1
    fi

    # cloud-init writes /etc/wireguard/wg0.conf early, but apt-get install of
    # wireguard-tools can still be in flight when sshd accepts logins. Wait up
    # to 5 min for the file to appear before trying to sed it.
    local file_ready=0
    for i in $(seq 1 60); do
        if ssh $SSH_OPTS root@"$host" "test -f /etc/wireguard/wg0.conf" 2>/dev/null; then
            file_ready=1
            break
        fi
        sleep 5
    done
    if (( file_ready == 0 )); then
        warn "wg0.conf never appeared on $host. cloud-init may have failed; check:"
        echo "    ssh root@$host 'tail -200 /var/log/pipeline-init.log'"
        return 1
    fi

    local err_log
    err_log=$(mktemp)
    for attempt in 1 2 3; do
        if ssh $SSH_OPTS root@"$host" \
            "sed -i 's/PEER_PUBLIC_IP/${peer_ip}/' /etc/wireguard/wg0.conf && \
             systemctl enable wg-quick@wg0 && \
             (systemctl start wg-quick@wg0 || systemctl restart wg-quick@wg0)" 2>"$err_log"; then
            ok "WireGuard patched on $host"
            rm -f "$err_log"
            return 0
        fi
        warn "SSH attempt $attempt/3 to $host failed: $(tr -d '\n' <"$err_log" | head -c 200)"
        sleep 5
    done
    warn "Could not patch WireGuard on $host after 3 SSH attempts"
    rm -f "$err_log"
    return 1
}

echo ""
log "Waiting for SSH and patching WireGuard on both nodes (in parallel)..."
patch_wg "$IP0" "$IP1" &
PATCH_PID0=$!
patch_wg "$IP1" "$IP0" &
PATCH_PID1=$!
wait $PATCH_PID0 $PATCH_PID1

# ── save deployment state ─────────────────────────────────────────────────────
cat > "$(dirname "$0")/.deploy-state" <<STATE
DROPLET0_ID=$DROPLET0_ID
DROPLET1_ID=$DROPLET1_ID
IP0=$IP0
IP1=$IP1
WG0_IP=$WG0_IP
WG1_IP=$WG1_IP
STATE

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Deployment complete"
echo ""
echo "  Stage 0  public: $IP0   WireGuard: $WG0_IP"
echo "  Stage 1  public: $IP1   WireGuard: $WG1_IP"
echo ""
echo "  SSH in:"
echo "    ssh root@$IP0   # stage 0"
echo "    ssh root@$IP1   # stage 1"
echo ""
echo "  Check model download progress:"
echo "    ssh root@$IP0 tail -f /var/log/pipeline-models.log"
echo "    ssh root@$IP1 tail -f /var/log/pipeline-models.log"
echo ""
# ── auto-start pipeline ───────────────────────────────────────────────────────
echo ""
log "Starting pipeline automatically on both nodes..."

# Stage 1 (must start first, no prompt)
ssh $SSH_OPTS root@$IP1 \
  "nohup /opt/pipeline/start.sh > /var/log/pipeline.log 2>&1 &"

# Small delay to ensure stage 1 is listening
sleep 5

# Stage 0 (entry point with prompt)
ssh $SSH_OPTS root@$IP0 \
  "nohup /opt/pipeline/start.sh --prompt 'hello world' > /var/log/pipeline.log 2>&1 &"

ok "Pipeline started on both nodes"

# ── show how to monitor (non-blocking) ────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Pipeline is running 🚀"
echo ""
echo "  View logs anytime:"
echo "    ssh root@$IP0 'tail -f /var/log/pipeline.log'"
echo "    ssh root@$IP1 'tail -f /var/log/pipeline.log'"
echo ""
echo "  Quick status check:"
echo "    ssh root@$IP0 'ps aux | grep launch.py'"
echo "    ssh root@$IP1 'ps aux | grep launch.py'"
echo "════════════════════════════════════════════════════════"