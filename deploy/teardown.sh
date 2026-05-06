#!/usr/bin/env bash
# teardown.sh — destroy all pipeline droplets and remove local state
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$DIR/.deploy-state"
STATE_JSON="$DIR/state.json"
[[ -f "$STATE" ]] || { echo "No .deploy-state found. Run deploy.sh first."; exit 1; }
source "$STATE"
NUM_STAGES="${NUM_STAGES:-2}"

ENV_FILE="$DIR/.env"
[[ -f "$ENV_FILE" ]] && { set -o allexport; source "$ENV_FILE"; set +o allexport; }
: "${DO_TOKEN:?Set DO_TOKEN in deploy/.env}"
export DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN"

echo "This will DESTROY $NUM_STAGES droplets:"
for i in $(seq 0 $((NUM_STAGES-1))); do
    id_var="DROPLET_ID_${i}"
    ip_var="IP${i}"
    echo "  pipeline-stage-${i}  (ID: ${!id_var}, IP: ${!ip_var})"
done
echo ""
read -rp "Type 'yes' to confirm: " confirm
[[ "$confirm" == "yes" ]] || { echo "Aborted."; exit 0; }

for i in $(seq 0 $((NUM_STAGES-1))); do
    id_var="DROPLET_ID_${i}"
    doctl compute droplet delete "${!id_var}" --force
    echo "  Deleted stage $i (${!id_var})"
done

rm -f "$STATE" "$STATE_JSON"
echo "Done."
