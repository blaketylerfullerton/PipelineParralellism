#!/usr/bin/env bash
# deploy.sh — spin up DO droplets, full-mesh WireGuard, bootstrap the pipeline
#
# Required (in deploy/.env):
#   DO_TOKEN       DigitalOcean API token
#   GITHUB_REPO    owner/repo  (e.g. blaketylerfullerton/PipeLineParralel)
#
# Optional:
#   HF_TOKEN       HuggingFace token (needed if models are gated)
#   DO_SSH_KEY     Fingerprint of your DO SSH key  (doctl compute ssh-key list)
#   REGION         Defaults to sfo3
#   SIZE           Defaults to s-8vcpu-16gb
#   NUM_STAGES     Defaults to pipeline.num_stages in config.yaml
#   CONFIG_FILE    Defaults to ../config.yaml
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${BOLD}▶ $*${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
err()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

# ── load .env ────────────────────────────────────────────────────────────────
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$DIR/.." && pwd)"
ENV_FILE="$DIR/.env"
[[ -f "$ENV_FILE" ]] && { set -o allexport; source "$ENV_FILE"; set +o allexport; }

# ── prerequisites ────────────────────────────────────────────────────────────
for cmd in doctl wg python3; do
    command -v "$cmd" &>/dev/null || err "$cmd not found. Install it first."
done

: "${DO_TOKEN:?Set DO_TOKEN in deploy/.env}"
: "${GITHUB_REPO:?Set GITHUB_REPO in deploy/.env}"
HF_TOKEN="${HF_TOKEN:-}"
DO_SSH_KEY="${DO_SSH_KEY:-}"
REGION="${REGION:-sfo3}"
SIZE="${SIZE:-s-8vcpu-16gb}"
CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/config.yaml}"
WG_SUBNET_PREFIX="${WG_SUBNET_PREFIX:-10.99.0}"

export DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN"
IMAGE="ubuntu-24-04-x64"
GIT_BRANCH="${GIT_BRANCH:-$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)}"
GIT_COMMIT="${GIT_COMMIT:-$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10"

read_config_stages() {
    python3 - "$CONFIG_FILE" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
match = re.search(r"(?m)^\s*num_stages\s*:\s*(\d+)", text)
print(match.group(1) if match else "")
PY
}

NUM_STAGES="${NUM_STAGES:-$(read_config_stages)}"
[[ "$NUM_STAGES" =~ ^[0-9]+$ ]] || err "NUM_STAGES must be a number; set NUM_STAGES or fix $CONFIG_FILE"
(( NUM_STAGES >= 2 )) || err "NUM_STAGES must be at least 2"

declare -a WG_IPS
for i in $(seq 0 $((NUM_STAGES-1))); do
    WG_IPS[$i]="${WG_SUBNET_PREFIX}.$((i+1))"
done
WG_PORT=51820

# ── generate WireGuard keypairs ───────────────────────────────────────────────
log "Generating WireGuard keypairs for $NUM_STAGES nodes..."
declare -a PRIV PUB
for i in $(seq 0 $((NUM_STAGES-1))); do
    PRIV[$i]=$(wg genkey)
    PUB[$i]=$(echo "${PRIV[$i]}" | wg pubkey)
    ok "Stage $i pubkey: ${PUB[$i]}"
done

SSH_FLAG=""
[[ -n "$DO_SSH_KEY" ]] && SSH_FLAG="--ssh-keys $DO_SSH_KEY"

# ── cloud-init builder ────────────────────────────────────────────────────────
# All scripts that need to land in /opt/pipeline are base64-encoded here and
# decoded on the droplet — this sidesteps heredoc-within-heredoc quoting issues.
make_cloud_init() {
    local stage="$1"
    local my_wg_ip="${WG_IPS[$stage]}"
    local my_priv="${PRIV[$stage]}"

    # Build full-mesh WireGuard config with placeholder IPs (patched over SSH later)
    local wg_conf
    wg_conf="[Interface]
PrivateKey = ${my_priv}
Address = ${my_wg_ip}/24
ListenPort = ${WG_PORT}"
    for i in $(seq 0 $((NUM_STAGES-1))); do
        [[ $i -eq $stage ]] && continue
        wg_conf+="

[Peer]
PublicKey = ${PUB[$i]}
AllowedIPs = ${WG_IPS[$i]}/32
Endpoint = PEER_PUB_IP_${i}:${WG_PORT}
PersistentKeepalive = 25"
    done
    local wg_b64; wg_b64=$(printf '%s' "$wg_conf" | base64 | tr -d '\n')

    # start.sh: WireGuard IPs are baked in — no discovery needed at runtime
    local all_wg_ips="${WG_IPS[*]}"
    local start_b64; start_b64=$(printf '#!/bin/bash\ncd /opt/pipeline\nsource .venv/bin/activate\nexec python src/launch.py --stage %d --peer-ip %s --stages %d "$@"\n' \
        "$stage" "$all_wg_ips" "$NUM_STAGES" | base64 | tr -d '\n')

    # download_models.sh: stage 0 gets draft + target; all others get target only
    local dl_script
    dl_script="#!/bin/bash
source /opt/pipeline/.venv/bin/activate
export HF_TOKEN='${HF_TOKEN}'"

    if [[ $stage -eq 0 ]]; then
        dl_script+="
echo '[dl] Downloading draft model unsloth/Llama-3.2-1B...'
python3 -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B', token='${HF_TOKEN}' or None); AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-1B', token='${HF_TOKEN}' or None)\""
    fi
    dl_script+="
echo '[dl] Downloading target model unsloth/Llama-3.2-3B...'
python3 -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('unsloth/Llama-3.2-3B', token='${HF_TOKEN}' or None); AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-3B', token='${HF_TOKEN}' or None)\"
echo '[dl] Done.'"
    local dl_b64; dl_b64=$(printf '%s' "$dl_script" | base64 | tr -d '\n')

    cat <<CLOUDINIT
#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/pipeline-init.log) 2>&1

echo "[init] Stage ${stage}/${NUM_STAGES} — installing packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git curl wireguard wireguard-tools build-essential

echo "[init] Writing WireGuard config (peer IPs patched after all droplets are up)..."
install -d -m 0700 /etc/wireguard
printf '%s' "${wg_b64}" | base64 -d > /etc/wireguard/wg0.conf
chmod 0600 /etc/wireguard/wg0.conf
# Allow WireGuard through ufw if it happens to be active
ufw allow ${WG_PORT}/udp 2>/dev/null || true

echo "[init] Cloning repo (branch: ${GIT_BRANCH})..."
git clone --branch "${GIT_BRANCH}" "https://github.com/${GITHUB_REPO}.git" /opt/pipeline
cd /opt/pipeline

echo "[init] Installing Python deps (CPU torch)..."
python3 -m venv /opt/pipeline/.venv
/opt/pipeline/.venv/bin/pip install --quiet --upgrade pip
/opt/pipeline/.venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
/opt/pipeline/.venv/bin/pip install --quiet -r requirements.txt

echo "[init] Writing helper scripts..."
printf '%s' "${start_b64}" | base64 -d > /opt/pipeline/start.sh
chmod +x /opt/pipeline/start.sh
printf '%s' "${dl_b64}" | base64 -d > /opt/pipeline/download_models.sh
chmod +x /opt/pipeline/download_models.sh

echo "[init] Starting model download in background..."
nohup /opt/pipeline/download_models.sh > /var/log/pipeline-models.log 2>&1 &

echo "[init] Bootstrap complete. Peer IPs will be patched and WireGuard started shortly."
echo "[init] Monitor: tail -f /var/log/pipeline-models.log"
CLOUDINIT
}

# ── create all droplets in parallel ──────────────────────────────────────────
echo ""
log "Creating $NUM_STAGES droplets in parallel..."
declare -a TMPFILES DROPLET_PIDS
for i in $(seq 0 $((NUM_STAGES-1))); do
    TMPFILES[$i]=$(mktemp)
    INIT=$(make_cloud_init $i)
    doctl compute droplet create "pipeline-stage-${i}" \
        --image "$IMAGE" --size "$SIZE" --region "$REGION" \
        $SSH_FLAG --user-data "$INIT" --wait --no-header --format ID \
        >"${TMPFILES[$i]}" 2>/dev/null &
    DROPLET_PIDS[$i]=$!
done

for i in $(seq 0 $((NUM_STAGES-1))); do
    wait "${DROPLET_PIDS[$i]}" || err "Stage $i droplet creation failed"
done

declare -a DROPLET_IDS PUBLIC_IPS
for i in $(seq 0 $((NUM_STAGES-1))); do
    DROPLET_IDS[$i]=$(tail -1 "${TMPFILES[$i]}")
    rm -f "${TMPFILES[$i]}"
    ok "Stage $i droplet ID: ${DROPLET_IDS[$i]}"
done

# ── fetch public IPs ──────────────────────────────────────────────────────────
echo ""
log "Fetching public IPs..."
for i in $(seq 0 $((NUM_STAGES-1))); do
    PUBLIC_IPS[$i]=$(doctl compute droplet get "${DROPLET_IDS[$i]}" --no-header --format PublicIPv4)
    ok "Stage $i  public: ${PUBLIC_IPS[$i]}   WireGuard: ${WG_IPS[$i]}"
done

# ── patch WireGuard on each node ──────────────────────────────────────────────
# Waits for SSH, then replaces PEER_PUB_IP_N placeholders and enables wg-quick.

wait_for_ssh() {
    local host="$1" timeout="${2:-300}" elapsed=0
    while (( elapsed < timeout )); do
        ssh $SSH_OPTS -o BatchMode=yes -o ConnectTimeout=8 root@"$host" true 2>/dev/null && return 0
        sleep 5; elapsed=$((elapsed+5))
    done
    return 1
}

patch_wg_node() {
    local idx="$1"
    local host="${PUBLIC_IPS[$idx]}"

    if ! wait_for_ssh "$host" 300; then
        warn "SSH never came up on Stage $idx ($host). Patch manually:"
        local sed_expr=""
        for i in $(seq 0 $((NUM_STAGES-1))); do
            [[ $i -eq $idx ]] && continue
            sed_expr+="s|PEER_PUB_IP_${i}|${PUBLIC_IPS[$i]}|g;"
        done
        echo "  ssh root@$host \"sed -i '${sed_expr}' /etc/wireguard/wg0.conf && systemctl enable --now wg-quick@wg0\""
        return 1
    fi

    # Wait for wg0.conf to appear (wireguard-tools install may still be running)
    local file_ready=0
    for _ in $(seq 1 60); do
        ssh $SSH_OPTS root@"$host" "test -f /etc/wireguard/wg0.conf" 2>/dev/null && { file_ready=1; break; }
        sleep 5
    done
    (( file_ready )) || { warn "wg0.conf never appeared on Stage $idx"; return 1; }

    # Build sed expression for all peer placeholders
    local sed_expr=""
    for i in $(seq 0 $((NUM_STAGES-1))); do
        [[ $i -eq $idx ]] && continue
        sed_expr+="s|PEER_PUB_IP_${i}|${PUBLIC_IPS[$i]}|g;"
    done

    local attempt
    for attempt in 1 2 3; do
        if ssh $SSH_OPTS root@"$host" \
            "sed -i '${sed_expr}' /etc/wireguard/wg0.conf && \
             systemctl enable wg-quick@wg0 && \
             (systemctl start wg-quick@wg0 || systemctl restart wg-quick@wg0)" 2>/dev/null; then
            ok "WireGuard up on Stage $idx ($host)"
            return 0
        fi
        sleep 5
    done
    warn "Could not start WireGuard on Stage $idx after 3 attempts"
    return 1
}

echo ""
log "Waiting for SSH and bringing up WireGuard on all nodes (in parallel)..."
declare -a PATCH_PIDS
for i in $(seq 0 $((NUM_STAGES-1))); do
    patch_wg_node $i &
    PATCH_PIDS[$i]=$!
done
for i in $(seq 0 $((NUM_STAGES-1))); do
    wait "${PATCH_PIDS[$i]}" || warn "WireGuard setup on Stage $i may have failed — check manually"
done

# ── save deployment state ─────────────────────────────────────────────────────
STATE_FILE="$DIR/.deploy-state"
STATE_JSON_FILE="${STATE_JSON_FILE:-$DIR/state.json}"
{
    echo "NUM_STAGES=$NUM_STAGES"
    echo "REGION=$REGION"
    echo "SIZE=$SIZE"
    echo "GIT_BRANCH=$GIT_BRANCH"
    echo "GIT_COMMIT=$GIT_COMMIT"
    for i in $(seq 0 $((NUM_STAGES-1))); do
        echo "DROPLET_ID_${i}=${DROPLET_IDS[$i]}"
        echo "IP${i}=${PUBLIC_IPS[$i]}"
        echo "WG${i}=${WG_IPS[$i]}"
    done
} > "$STATE_FILE"

python3 - "$STATE_FILE" "$STATE_JSON_FILE" <<'PY'
import json
import time
import sys
from pathlib import Path

legacy = Path(sys.argv[1])
target = Path(sys.argv[2])
values = {}
for line in legacy.read_text().splitlines():
    if "=" not in line:
        continue
    key, value = line.split("=", 1)
    values[key] = value

num_stages = int(values["NUM_STAGES"])
state = {
    "schema_version": 1,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "num_stages": num_stages,
    "region": values.get("REGION"),
    "size": values.get("SIZE"),
    "git_branch": values.get("GIT_BRANCH"),
    "git_commit": values.get("GIT_COMMIT"),
    "stages": [
        {
            "stage": i,
            "droplet_id": values.get(f"DROPLET_ID_{i}"),
            "public_ip": values.get(f"IP{i}"),
            "wireguard_ip": values.get(f"WG{i}"),
        }
        for i in range(num_stages)
    ],
}
target.write_text(json.dumps(state, indent=2) + "\n")
PY

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Deployment complete  ($NUM_STAGES stages, $SIZE, $REGION)"
echo "  Repo ref: ${GIT_BRANCH}@${GIT_COMMIT}"
echo ""
for i in $(seq 0 $((NUM_STAGES-1))); do
    printf "  Stage %d  public: %-16s  WireGuard: %s\n" "$i" "${PUBLIC_IPS[$i]}" "${WG_IPS[$i]}"
done
echo ""
echo "  SSH in:  ssh root@${PUBLIC_IPS[0]}   (and so on)"
echo ""
echo "  Watch model downloads:"
for i in $(seq 0 $((NUM_STAGES-1))); do
    echo "    ssh root@${PUBLIC_IPS[$i]} 'tail -f /var/log/pipeline-models.log'"
done
echo ""
echo "  When models are ready, run the pipeline:"
echo "    ./deploy/run.sh --prompt 'your prompt here'"
echo "  Or inspect first:"
echo "    ./deploy/relayctl.py doctor"
echo "════════════════════════════════════════════════════════"
