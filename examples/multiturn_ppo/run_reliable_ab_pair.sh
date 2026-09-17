#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh
PAIR="${1:?Set a unique pair identifier}"
CHECKPOINT_AUDIT="${2:?Pass the completed four-rank checkpoint audit directory}"
export AGL_FULL_ENV_AUDIT="${3:?Pass the new environment audit directory}"
[[ "$PAIR" =~ ^[a-zA-Z0-9_-]+$ ]]
ROOT=/media/ubuntu/D1/zsj/agent-lightning-runtime/logs
OUT="$ROOT/ab-pair-$PAIR"
[[ ! -e "$OUT" ]]
mkdir "$OUT"
exec >"$OUT/run.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$OUT/run.exit"' EXIT
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python - "$CHECKPOINT_AUDIT" "$AGL_FULL_ENV_AUDIT" <<'PY'
import json, sys
from pathlib import Path
checkpoint, environment = map(Path, sys.argv[1:])
proof = json.loads((checkpoint / 'completed.json').read_text())
assert proof['passed'] and set(proof['roles']) == {'actor', 'critic'}
assert (checkpoint / 'run.exit').read_text().strip() == '0'
for role in ('actor', 'critic'):
    assert proof['roles'][role]['world_size'] == 4
    evidence = Path(proof['roles'][role].get('evidence_directory', str(checkpoint / role)))
    for rank in range(4):
        record = json.loads((evidence / f'rank-{rank}.json').read_text())
        assert all(record[k] for k in ('restored_state_exact', 'restored_prediction_exact',
                                       'replay_state_exact', 'replay_prediction_exact'))
summary = json.loads((environment / 'summary.json').read_text())
assert summary['ready'] and summary['passed'] == summary['total_images']
print('WORKER_RECOVERY_AND_ENVIRONMENT_GATES_PASSED', flush=True)
PY
unset AGL_CONFIG_ONLY FLASH_ATTENTION_DETERMINISTIC CUBLAS_WORKSPACE_CONFIG
# Prioritize the P1-supported candidate. Both arms start fresh and stop at 20;
# a failure/behavior stop in B exits this pair instead of silently continuing.
for arm in B A; do
  if [[ "$arm" == B ]]; then mode=zero; else mode=default; fi
  export AGL_TRAIN_TAG="capo-swe-v2-mini128-$mode-4b-gpu47-$PAIR"
  export AGL_TRAIN_PORT=18521
  printf '%s\n' "START_ARM=$arm TAG=$AGL_TRAIN_TAG"
  bash "$TOOLS/run_reliable_ab_trial.sh" "$arm"
  printf '%s\n' "FINISHED_ARM=$arm TAG=$AGL_TRAIN_TAG"
done
echo AB_PAIR_20_STEPS_COMPLETE
