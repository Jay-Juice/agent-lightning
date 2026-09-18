#!/usr/bin/env bash
# Revalidate the changed grader before starting one fresh, matched A/B pair.
set -Eeuo pipefail
PROJECT=/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917
RUNTIME=/media/ubuntu/D1/zsj/agent-lightning-runtime
ROOT=$RUNTIME/logs
OUT=$ROOT/recovery-v2-exit120-20260918-01
[[ ! -e "$OUT" ]]
mkdir "$OUT"
exec >"$OUT/run.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$OUT/run.exit"' EXIT
source "$RUNTIME/admin/activate-agent-lightning-d1.sh"
cd "$PROJECT"
ENV_AUDIT=$ROOT/swe-full-python-envs-reliable-v2-20260918-03
LIVE_AUDIT=$ROOT/audit-v2-grading-live-20260918-02
bash research/run_v2_env_audit.sh "$ENV_AUDIT" &
env_pid=$!
bash research/run_verify_v2_grading_live.sh "$LIVE_AUDIT" \
  --extra-original "$ROOT/training-capo-swe-v2-mini128-zero-4b-gpu47-20260918-02" &
live_pid=$!
env_status=0
live_status=0
wait "$env_pid" || env_status=$?
wait "$live_pid" || live_status=$?
printf 'AUDITS_FINISHED environment=%s live=%s\n' "$env_status" "$live_status"
[[ "$env_status" == 0 && "$live_status" == 0 ]]
echo STARTING_FRESH_PAIR_20260918_03
bash examples/multiturn_ppo/run_reliable_ab_pair.sh \
  20260918-03 \
  "$ROOT/audit-checkpoint-roundtrip-v2-20260918-03" \
  "$ENV_AUDIT" "$LIVE_AUDIT" \
  "$ROOT/audit-v2-episode-replay-all-20260918-01"
