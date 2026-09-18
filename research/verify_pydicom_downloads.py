"""Replay the two held candidates and their reference without model resampling."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'examples/multiturn_ppo'), str(REPO)]
from full_python_agent import grade, grading_test_nodes
from swe_reliability import grade_fixed_patch

ROOT = Path('/media/ubuntu/D1/zsj/agent-lightning-runtime/logs')
OUT = ROOT / 'audit-pydicom-downloads-20260919-01'


def main():
    os.environ.update(SMITH_EVAL_TIMEOUT='600', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    assert not OUT.exists(), 'Never overwrite prior audit evidence'
    OUT.mkdir()
    hashes = {name: hashlib.sha256((REPO / 'examples/multiturn_ppo' / name).read_bytes()).hexdigest()
              for name in ['full_python_agent.py', 'pydicom_http_fixture.py',
                           'swe_reliability.py', 'swe_grading_evidence.py']}
    summary = {'passed': False, 'source_hashes': hashes, 'cases': []}
    cases = [('A', 'default', '0123', '86292d3089ea45afb2f5c2595a550508'),
             ('B', 'zero', '4567', 'add0ee6523f14507b54bc1c958951d7d')]
    task_hashes = set()
    for arm, head, gpus, rid in cases:
        original = ROOT / f'training-capo-swe-v2-full-{head}-4b-gpu{gpus}-resume20-20260919-01'
        row = json.loads((original / 'traces' / (rid + '.json')).read_text())['rollout']['input']
        patch = (original / 'agent' / rid / 'model.patch').read_text()
        old = json.loads((original / 'agent' / rid / 'grade.json').read_text())
        assert hashlib.sha256(patch.encode()).hexdigest() == old['patch_sha256']
        task_hashes.add(old['task_sha256'])
        path = OUT / arm
        path.mkdir()
        (path / 'task.json').write_text(json.dumps(row, indent=2))
        (path / 'candidate.patch').write_text(patch)
        started = time.monotonic()
        report = grade_fixed_patch(grade, row, patch, path, retry_errors=(), deadline=started + 2000)
        nodes = set(sum(grading_test_nodes(row), []))
        passed = (report['pytest_exit'] == 1 and report['reward'] == 0
                  and report['grading_status'] == 'completed' and set(report['test_statuses']) == nodes
                  and report['patch_sha256'] == old['patch_sha256']
                  and report['recorded_pydicom_fixture']['files'] == 79)
        summary['cases'].append({'arm': arm, 'passed': passed, 'exit': report['pytest_exit'],
                                 'reward': report['reward'], 'tests': len(report['test_statuses']),
                                 'elapsed': time.monotonic() - started,
                                 'f2p_passed': report['f2p_passed'], 'p2p_passed': report['p2p_passed']})
        print(json.dumps(summary['cases'][-1]), flush=True)
        assert passed, 'Candidate did not produce a complete, ordinary failure'
        if arm == 'A':
            ref_dir = OUT / 'reference'
            ref_dir.mkdir()
            ref = grade(row, patch, ref_dir, reference=True)
            passed = ref['pytest_exit'] == 0 and ref['reward'] == 1 and set(ref['test_statuses']) == nodes
            summary['reference'] = {'passed': passed, 'exit': ref['pytest_exit'], 'reward': ref['reward'],
                                    'tests': len(ref['test_statuses']), 'elapsed': ref['test_elapsed_seconds']}
            print(json.dumps(summary['reference']), flush=True)
            assert passed, 'Reference must pass every original test'
    assert len(task_hashes) == 1
    summary['passed'] = True
    summary['finished_at'] = time.time()
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
