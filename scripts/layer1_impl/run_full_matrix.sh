#!/usr/bin/env bash
# Full Layer 1 implementation experiment: every model x every harness, one consistent code version.
#
#   Stage A  pinned wording   G0 only            7 models x 42 tasks (canonical20 + qcb22)
#   Stage B  original wording 10 harnesses       7 models x 22 tasks (qcb22)
#
# Each (stage, harness) writes to its own directory so stages can run in parallel without
# sharing a file. Resumable: re-running skips finished (harness, variant, model, case) rows and
# retries rows that ended in an infrastructure error.
#
# Usage:  bash scripts/layer1_impl/run_full_matrix.sh [OUT_DIR]
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

OUT="${1:-results/layer1_impl/full_matrix_20261023}"
MODELS="claude-sonnet-5.5 claude-opus-5.5 claude-fable-5.1 gpt-6.1-sol gpt-6-astra deepseek-v4.1-flash deepseek-v4-pro"
WORKERS=3
mkdir -p "$OUT/logs"

run() {  # name harness variant task_set
  local name=$1 harness=$2 variant=$3 tset=$4
  .venv/bin/python -W ignore scripts/layer1_impl/run_harness_matrix.py \
      --harness "$harness" --variant "$variant" --task-set "$tset" \
      --models $MODELS --workers $WORKERS --n-bars 1500 \
      --out "$OUT/$name" > "$OUT/logs/$name.log" 2>&1
  echo "$(date '+%H:%M:%S') done $name ($(grep -c '' "$OUT/$name/runs.jsonl" 2>/dev/null || echo 0) rows)"
}

run A_pinned_G0 G0 pinned all &
for h in G0 G1 G1c G2 G3w G4_v2 G3 G3c_q3 G3c_q5 G3c; do
  run "B_original_$h" "$h" original qcb22 &
done
wait
echo "all stages finished"
