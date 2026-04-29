#!/usr/bin/env bash
# run.sh — start the pipeline on already-deployed droplets
#
# Usage:
#   ./run.sh                          # will ask for prompt interactively
#   ./run.sh --prompt "hello world"   # pass prompt directly
set -euo pipefail

DIR="$(dirname "$0")"
STATE="$DIR/.deploy-state"

if [[ ! -f "$STATE" ]]; then
    echo "ERROR: No .deploy-state found. Run deploy.sh first."
    exit 1
fi

source "$STATE"

# ── parse args ────────────────────────────────────────────────────────────────
PROMPT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt) PROMPT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$PROMPT" ]]; then
    printf "Prompt: "
    read -r PROMPT
fi

if [[ -z "$PROMPT" ]]; then
    echo "ERROR: prompt cannot be empty."
    exit 1
fi

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3"

# ── wait for model downloads ──────────────────────────────────────────────────
wait_for_models() {
    local host="$1"
    local label="$2"
    local attempts=0
    echo "  Waiting for models on $label ($host)..."
    while true; do
        local last
        last=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
            -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
            root@"$host" "tail -1 /var/log/pipeline-models.log 2>/dev/null || echo NOLOG" 2>&1 || echo "SSH_ERR")

        if [[ "$last" == "SSH_ERR"* ]] || [[ "$last" == *"Permission denied"* ]]; then
            attempts=$((attempts + 1))
            echo "  [$label] SSH not ready yet (attempt $attempts) — retrying in 15s..."
            sleep 15
            continue
        fi

        # grep avoids the [dl] glob-character-class pitfall
        if echo "$last" | grep -q '\[dl\] Done'; then
            echo "  Models ready on $label"
            return 0
        fi

        if [[ "$last" != "NOLOG" ]] && [[ -n "$last" ]]; then
            printf "\r  %-70s" "$label: $last"
        fi
        sleep 10
    done
}

echo ""
echo "Checking model download status..."
wait_for_models "$IP0" "Stage 0"
wait_for_models "$IP1" "Stage 1"
echo ""

# ── launch in tmux ───────────────────────────────────────────────────────────
SESSION="relay"

# Kill old session if exists
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -x "$(tput cols)" -y "$(tput lines)"

# Left pane — Stage 1 (starts first, goes into waiting mode)
tmux send-keys -t "$SESSION:0.0" \
    "${SSH} root@${IP1} '/opt/pipeline/start.sh'" Enter

# Right pane — Stage 0 with the prompt (starts after a short delay)
tmux split-window -h -t "$SESSION"
tmux send-keys -t "$SESSION:0.1" \
    "sleep 4 && ${SSH} root@${IP0} '/opt/pipeline/start.sh --prompt \"${PROMPT}\"'" Enter

tmux select-pane -t "$SESSION:0.0"

echo "════════════════════════════════════════════════════════"
echo "  Stage 1  left pane   →  $IP1"
echo "  Stage 0  right pane  →  $IP0"
echo "  Prompt: \"$PROMPT\""
echo "════════════════════════════════════════════════════════"
echo ""

exec tmux attach-session -t "$SESSION"
