#!/usr/bin/env bash
# run.sh — start the pipeline on already-deployed droplets
#
# Usage:
#   ./deploy/run.sh --prompt "hello world"
#   ./deploy/run.sh --prompt "hello world" --no-wait
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$DIR/.deploy-state"

[[ -f "$STATE" ]] || { echo "ERROR: No .deploy-state found. Run deploy.sh first."; exit 1; }
source "$STATE"

NUM_STAGES="${NUM_STAGES:-2}"
(( NUM_STAGES >= 2 )) || { echo "ERROR: NUM_STAGES must be at least 2."; exit 1; }

PROMPT=""
WAIT_FOR_MODELS=1
SESSION="${SESSION:-relay}"
START_DELAY="${START_DELAY:-6}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt) PROMPT="$2"; shift 2 ;;
        --no-wait) WAIT_FOR_MODELS=0; shift ;;
        --session) SESSION="$2"; shift 2 ;;
        --start-delay) START_DELAY="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; WAIT_FOR_MODELS=0; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "$PROMPT" ]] && { printf "Prompt: "; read -r PROMPT; }
[[ -z "$PROMPT" ]] && { echo "ERROR: prompt cannot be empty."; exit 1; }

for i in $(seq 0 $((NUM_STAGES-1))); do
    ip_var="IP${i}"
    [[ -n "${!ip_var:-}" ]] || { echo "ERROR: $ip_var missing from $STATE"; exit 1; }
done

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3"

if (( DRY_RUN )); then
    prompt_b64=$(printf '%s' "$PROMPT" | base64 | tr -d '\n')
    remote_driver="PROMPT=\$(printf '%s' '$prompt_b64' | base64 -d); /opt/pipeline/start.sh --prompt \"\$PROMPT\""
    remote_driver_q=$(printf '%q' "$remote_driver")
    echo "Dry run: would start $NUM_STAGES stages from $STATE"
    for stage in $(seq 1 $((NUM_STAGES-1))); do
        ip_var="IP${stage}"
        echo "  stage $stage: $SSH root@${!ip_var} /opt/pipeline/start.sh"
    done
    echo "  stage 0: sleep $START_DELAY && $SSH root@${IP0} $remote_driver_q"
    exit 0
fi

wait_for_models() {
    local host="$1" label="$2" attempts=0
    echo "  Waiting for models on $label ($host)..."
    while true; do
        local last
        last=$($SSH root@"$host" "tail -1 /var/log/pipeline-models.log 2>/dev/null || echo NOLOG" 2>&1 || echo "SSH_ERR")

        if [[ "$last" == "SSH_ERR"* ]] || [[ "$last" == *"Permission denied"* ]] || [[ "$last" == *"Connection refused"* ]]; then
            attempts=$((attempts+1))
            echo "  [$label] SSH unreachable (attempt $attempts)"
            [[ $attempts -eq 3 ]] && echo "  HINT: check DO_SSH_KEY in deploy/.env and try: ssh root@$host"
            sleep 15; continue
        fi

        if echo "$last" | grep -q '\[dl\] Done'; then
            echo "  Models ready on $label"
            return 0
        fi

        [[ "$last" != "NOLOG" ]] && [[ -n "$last" ]] && printf "\r  %-70s" "$label: $last"
        sleep 10
    done
}

if (( WAIT_FOR_MODELS )); then
    echo ""
    echo "Checking model download status on all $NUM_STAGES nodes..."
    declare -a WAIT_PIDS
    for i in $(seq 0 $((NUM_STAGES-1))); do
        ip_var="IP${i}"
        wait_for_models "${!ip_var}" "Stage ${i}" &
        WAIT_PIDS[$i]=$!
    done
    for i in $(seq 0 $((NUM_STAGES-1))); do
        wait "${WAIT_PIDS[$i]}"
    done
    echo ""
fi

cols=$(tput cols 2>/dev/null || echo 120)
lines=$(tput lines 2>/dev/null || echo 40)

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x "$cols" -y "$lines"

send_stage() {
    local pane="$1" stage="$2"
    local ip_var="IP${stage}"
    tmux send-keys -t "$pane" "$SSH root@${!ip_var} /opt/pipeline/start.sh" Enter
}

first_pane=$(tmux display-message -p -t "$SESSION:0.0" '#{pane_id}')
send_stage "$first_pane" 1

if (( NUM_STAGES > 2 )); then
    for stage in $(seq 2 $((NUM_STAGES-1))); do
        pane=$(tmux split-window -d -P -F '#{pane_id}' -t "$SESSION:0")
        send_stage "$pane" "$stage"
    done
fi

prompt_b64=$(printf '%s' "$PROMPT" | base64 | tr -d '\n')
remote_driver="PROMPT=\$(printf '%s' '$prompt_b64' | base64 -d); /opt/pipeline/start.sh --prompt \"\$PROMPT\""
remote_driver_q=$(printf '%q' "$remote_driver")
driver_pane=$(tmux split-window -d -P -F '#{pane_id}' -t "$SESSION:0")
tmux send-keys -t "$driver_pane" "sleep $START_DELAY && $SSH root@${IP0} $remote_driver_q" Enter

tmux select-layout -t "$SESSION:0" tiled >/dev/null
tmux select-pane -t "$driver_pane"

echo "════════════════════════════════════════════════════════"
for stage in $(seq 1 $((NUM_STAGES-1))); do
    ip_var="IP${stage}"
    printf "  Stage %-2d worker → %s\n" "$stage" "${!ip_var}"
done
echo "  Stage 0  driver → $IP0  (starts in ${START_DELAY}s)"
echo "  Prompt: \"$PROMPT\""
echo "════════════════════════════════════════════════════════"
echo ""

exec tmux attach-session -t "$SESSION"
