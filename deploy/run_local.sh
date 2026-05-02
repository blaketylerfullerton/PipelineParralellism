#!/usr/bin/env bash
# run_local.sh — run both pipeline stages locally (no deploy needed)
#
# Usage:
#   ./run_local.sh                          # will ask for prompt interactively
#   ./run_local.sh --prompt "hello world"   # pass prompt directly
#
# Both stages run on 127.0.0.1 in a split tmux session.
# Models are downloaded from HuggingFace on first run and cached automatically.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/venv"
ACTIVATE="$VENV/bin/activate"

if [[ ! -f "$ACTIVATE" ]]; then
    echo "ERROR: venv not found at $VENV"
    echo "Set one up with: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

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

# ── tmux session ──────────────────────────────────────────────────────────────
SESSION="pipeline-local"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x "$(tput cols)" -y "$(tput lines)"

# Left pane — Stage 1 (binds sockets first, waits for hidden states from Stage 0)
tmux send-keys -t "$SESSION:0.0" \
    "cd '$REPO_ROOT' && source '$ACTIVATE' && python src/launch.py --stage 1 --peer-ip 127.0.0.1 127.0.0.1" Enter

# Right pane — Stage 0 (connects after a brief delay so Stage 1 is bound)
tmux split-window -h -t "$SESSION"
tmux send-keys -t "$SESSION:0.1" \
    "cd '$REPO_ROOT' && source '$ACTIVATE' && sleep 5 && python src/launch.py --stage 0 --peer-ip 127.0.0.1 127.0.0.1 --prompt '${PROMPT}'" Enter

tmux select-pane -t "$SESSION:0.0"

echo "════════════════════════════════════════════════════════"
echo "  Stage 1  left pane   →  127.0.0.1 (waiting for hiddens)"
echo "  Stage 0  right pane  →  127.0.0.1 (starts in 5 s)"
echo "  Prompt: \"$PROMPT\""
echo "  Models cached at: ~/.cache/huggingface"
echo "════════════════════════════════════════════════════════"
echo ""

exec tmux attach-session -t "$SESSION"
