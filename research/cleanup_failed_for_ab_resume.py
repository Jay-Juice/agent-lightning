"""Scoped cleanup authorized for parallel A/B continuation on 2026-09-19."""
import argparse
import json
import shutil
import time

import cleanup_old_baseline_step80 as audit

ROOT = audit.ROOT
TARGETS = [
    ROOT / 'training-capo-swe-pythonfull-v7-4b-gpu47-20260914-01/checkpoints/global_step_40',
    ROOT / 'training-capo-swe-pythonfull-v7-p2fixed-4b-gpu03-20260916-01/checkpoints/global_step_40',
]
KEEP = [
    ROOT / 'training-capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01/checkpoints/global_step_120',
    ROOT / 'training-capo-swe-pythonfull-v7-p2fixed-4b-gpu03-20260916-01/checkpoints/global_step_80',
    *[ROOT / f'training-capo-swe-v2-mini128-{head}-4b-gpu47-20260918-03/checkpoints/global_step_{step}'
      for head in ('default', 'zero') for step in (15, 20)],
]
OUT = ROOT / 'cleanup-failed-for-ab-resume-20260919-01'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    targets = {}
    for target in TARGETS:
        assert target.parent.parent.parent == ROOT and target.name == 'global_step_40'
        assert (target.parent.parent / 'run.exit').read_text().strip() == '1'
        audit.TARGET = target
        assert not audit.references(), 'Checkpoint is still referenced by a live process'
        targets[str(target)] = audit.inventory(target)
    preserved = {str(p): audit.inventory(p) for p in KEEP}
    plan = {'targets': targets, 'preserved': preserved,
            'bytes': sum(v[0] for files in targets.values() for v in files.values())}
    OUT.mkdir(exist_ok=True)
    if not args.apply:
        assert not (OUT / 'plan.json').exists()
        (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
        print(json.dumps({'targets': list(targets), 'gib': plan['bytes'] / 2**30,
                          'preserved': list(preserved), 'mode': 'dry-run'}))
        return
    assert not (OUT / 'result.json').exists()
    assert json.loads((OUT / 'plan.json').read_text()) == plan, 'State changed after dry-run'
    free_before = shutil.disk_usage(ROOT).free
    for target in TARGETS:
        shutil.rmtree(target)
        assert not target.exists()
    assert {str(p): audit.inventory(p) for p in KEEP} == preserved
    result = {'deleted': list(targets), 'preserved_unchanged': True, 'bytes': plan['bytes'],
              'free_before': free_before, 'free_after': shutil.disk_usage(ROOT).free, 'time': time.time()}
    (OUT / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
