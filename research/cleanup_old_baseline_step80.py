"""Remove only the historical step80 approved when the user continued A/B."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/media/ubuntu/D1/zsj/agent-lightning-runtime/logs')
OLD = ROOT / 'training-capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01/checkpoints'
TARGET = OLD / 'global_step_80'
KEEP = [OLD / 'global_step_120',
        ROOT / 'training-capo-swe-pythonfull-v7-4b-gpu47-20260914-01/checkpoints/global_step_40',
        ROOT / 'training-capo-swe-v2-mini128-zero-4b-gpu47-20260918-03/checkpoints/global_step_15',
        ROOT / 'training-capo-swe-v2-mini128-zero-4b-gpu47-20260918-03/checkpoints/global_step_20',
        ROOT / 'training-capo-swe-pythonfull-v7-p2fixed-4b-gpu03-20260916-01/checkpoints/global_step_40',
        ROOT / 'training-capo-swe-pythonfull-v7-p2fixed-4b-gpu03-20260916-01/checkpoints/global_step_80']
AUDIT = ROOT / 'cleanup-old-baseline-step80-20260918-01'


def inventory(path):
    assert path.is_dir() and path.resolve() == path
    assert all(not p.is_symlink() for p in (path, *path.parents))
    entries = list(path.rglob('*'))
    assert not any(p.is_symlink() for p in entries)
    files = {str(p.relative_to(path)): [p.stat().st_size, p.stat().st_mtime_ns]
             for p in entries if p.is_file()}
    for role in ('actor', 'critic'):
        for kind in ('model', 'optim', 'extra_state'):
            for rank in range(4):
                assert files[f'{role}/{kind}_world_size_4_rank_{rank}.pt'][0] > 0
    assert files['data.pt'][0] > 0
    return files


def references():
    hits = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            if (proc / 'stat').read_text().rpartition(')')[2].split()[0] == 'Z':
                continue
            comm = (proc / 'comm').read_text().strip()
            if comm in {'(sd-pam)', 'sshd'}:
                continue
            hit = (str(TARGET).encode() in (proc / 'cmdline').read_bytes()
                   or str(TARGET).encode() in (proc / 'environ').read_bytes())
            for fd in (proc / 'fd').iterdir():
                try:
                    hit = hit or os.readlink(fd).startswith(str(TARGET) + '/')
                except FileNotFoundError:
                    pass
            if hit:
                hits.append({'pid': int(proc.name), 'comm': comm})
        except (FileNotFoundError, ProcessLookupError):
            pass
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    assert TARGET.parent.parent.parent == ROOT and TARGET.name == 'global_step_80'
    target_files = inventory(TARGET)
    keep_files = {str(p): inventory(p) for p in KEEP}
    refs = references()
    assert not refs, refs
    plan = {'target': str(TARGET), 'files': target_files, 'preserved': keep_files,
            'bytes': sum(v[0] for v in target_files.values()), 'live_references': refs}
    AUDIT.mkdir(exist_ok=True)
    if not args.apply:
        assert not (AUDIT / 'plan.json').exists(), 'Keep the original dry-run evidence'
        (AUDIT / 'plan.json').write_text(json.dumps(plan, indent=2))
        print(json.dumps({'mode': 'dry-run', 'target': str(TARGET), 'gib': plan['bytes'] / 2**30,
                          'preserved_checkpoints': len(KEEP), 'live_references': refs}))
        return
    assert not (AUDIT / 'result.json').exists()
    assert json.loads((AUDIT / 'plan.json').read_text()) == plan, 'State changed since dry-run'
    before = shutil.disk_usage(ROOT).free
    shutil.rmtree(TARGET)
    assert not TARGET.exists()
    assert {str(p): inventory(p) for p in KEEP} == keep_files
    result = {'deleted': str(TARGET), 'bytes': plan['bytes'], 'preserved_unchanged': True,
              'free_before': before, 'free_after': shutil.disk_usage(ROOT).free, 'time': time.time()}
    (AUDIT / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
