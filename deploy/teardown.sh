#!/usr/bin/env bash
# teardown.sh — destroy both pipeline droplets
set -euo pipefail

STATE="$(dirname "$0")/.deploy-state"
if [[ ! -f "$STATE" ]]; then
    echo "No .deploy-state found. Run deploy.sh first."
    exit 1
fi

source "$STATE"

: "${DO_TOKEN:?Set DO_TOKEN}"
export DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN"

echo "This will DESTROY:"
echo "  pipeline-stage-0  (ID: $DROPLET0_ID, IP: $IP0)"
echo "  pipeline-stage-1  (ID: $DROPLET1_ID, IP: $IP1)"
echo ""
read -rp "Type 'yes' to confirm: " confirm
[[ "$confirm" == "yes" ]] || { echo "Aborted."; exit 0; }

doctl compute droplet delete "$DROPLET0_ID" --force
echo "  Deleted stage 0 ($DROPLET0_ID)"
doctl compute droplet delete "$DROPLET1_ID" --force
echo "  Deleted stage 1 ($DROPLET1_ID)"

rm "$STATE"
echo "Done."
