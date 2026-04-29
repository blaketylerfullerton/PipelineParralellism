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

# ── load .env if present ─────────────────────────────────────────────────────
ENV_FILE="$(dirname "$0")/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
fi

# ── prerequisites ────────────────────────────────────────────────────────────
for cmd in doctl wg; do
    command -v "$cmd" &>/dev/null || { echo "ERROR: $cmd not found. Install it first."; exit 1; }
done

: "${DO_TOKEN:?Set DO_TOKEN}"
: "${GITHUB_REPO:?Set GITHUB_REPO (e.g. youruser/PipeLineParralel)}"
HF_TOKEN="${HF_TOKEN:-}"
DO_SSH_KEY="${DO_SSH_KEY:-}"
REGION="${REGION:-sfo3}"
SIZE="${SIZE:-s-4vcpu-8gb}"

export DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN"
IMAGE="ubuntu-24-04-x64"
GIT_BRANCH=$(git -C "$(dirname "$0")/.." rev-parse --abbrev-ref HEAD)

# ── WireGuard key generation ──────────────────────────────────────────────────
echo "Generating WireGuard keypairs..."
PRIV0=$(wg genkey);  PUB0=$(echo "$PRIV0" | wg pubkey)
PRIV1=$(wg genkey);  PUB1=$(echo "$PRIV1" | wg pubkey)

WG0_IP="10.99.0.1"
WG1_IP="10.99.0.2"
WG_PORT=51820

echo "  Stage 0 pubkey: $PUB0"
echo "  Stage 1 pubkey: $PUB1"

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

echo "[init] Cloning repo (branch: ${GIT_BRANCH})..."
git clone --branch "${GIT_BRANCH}" "https://github.com/${GITHUB_REPO}.git" /opt/pipeline
cd /opt/pipeline

echo "[init] Installing Python deps (CPU torch)..."
python3 -m venv /opt/pipeline/.venv
/opt/pipeline/.venv/bin/pip install --quiet --upgrade pip
/opt/pipeline/.venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
/opt/pipeline/.venv/bin/pip install --quiet -r requirements.txt

echo "[init] Writing WireGuard config..."
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

echo "[init] WireGuard config written (peer IP will be patched after both droplets are up)"

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

# ── create droplets ───────────────────────────────────────────────────────────
echo ""
echo "Creating Stage 0 droplet..."
INIT0=$(make_cloud_init 0 "$WG0_IP" "$PRIV0" "$WG1_IP" "$PUB1")

DROPLET0_ID=$(doctl compute droplet create "pipeline-stage-0" \
    --image "$IMAGE" \
    --size "$SIZE" \
    --region "$REGION" \
    $SSH_FLAG \
    --user-data "$INIT0" \
    --wait \
    --no-header \
    --format ID 2>/dev/null | tail -1)

echo "  Droplet ID: $DROPLET0_ID"

echo "Creating Stage 1 droplet..."
INIT1=$(make_cloud_init 1 "$WG1_IP" "$PRIV1" "$WG0_IP" "$PUB0")

DROPLET1_ID=$(doctl compute droplet create "pipeline-stage-1" \
    --image "$IMAGE" \
    --size "$SIZE" \
    --region "$REGION" \
    $SSH_FLAG \
    --user-data "$INIT1" \
    --wait \
    --no-header \
    --format ID 2>/dev/null | tail -1)

echo "  Droplet ID: $DROPLET1_ID"

# ── fetch public IPs ──────────────────────────────────────────────────────────
echo ""
echo "Fetching public IPs..."
IP0=$(doctl compute droplet get "$DROPLET0_ID" --no-header --format PublicIPv4)
IP1=$(doctl compute droplet get "$DROPLET1_ID" --no-header --format PublicIPv4)

echo "  Stage 0: $IP0"
echo "  Stage 1: $IP1"

# ── patch WireGuard peer IP on each droplet ───────────────────────────────────
# The cloud-init wrote PEER_PUBLIC_IP as a placeholder — patch it over SSH
# once both droplets are up. Retry a few times for SSH to come up.
patch_wg() {
    local host="$1"
    local peer_ip="$2"
    for attempt in 1 2 3 4 5 6 7 8; do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$host" \
            "sed -i 's/PEER_PUBLIC_IP/${peer_ip}/' /etc/wireguard/wg0.conf && \
             systemctl enable wg-quick@wg0 && \
             systemctl start wg-quick@wg0 || systemctl restart wg-quick@wg0" 2>/dev/null; then
            echo "  WireGuard patched on $host"
            return 0
        fi
        echo "  SSH not ready on $host (attempt $attempt/5), waiting 30s..."
        sleep 30
    done
    echo "  WARNING: Could not patch WireGuard on $host. SSH in manually and run:"
    echo "    sed -i 's/PEER_PUBLIC_IP/${peer_ip}/' /etc/wireguard/wg0.conf"
    echo "    systemctl enable wg-quick@wg0 && systemctl start wg-quick@wg0"
}

echo ""
echo "Patching WireGuard peer IPs and starting wg0..."
patch_wg "$IP0" "$IP1"
patch_wg "$IP1" "$IP0"

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
echo "  Once models are downloaded, run in TWO terminals:"
echo "    ssh root@$IP1 '/opt/pipeline/start.sh'            # stage 1 first"
echo "    ssh root@$IP0 '/opt/pipeline/start.sh --prompt \"hello world\"'  # stage 0"
echo ""
echo "  Verify WireGuard tunnel (from stage 0):"
echo "    ssh root@$IP0 ping -c 3 $WG1_IP"
echo ""
echo "  Tear down:  ./teardown.sh"
echo "════════════════════════════════════════════════════════"

