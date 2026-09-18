"""Retain complete checkpoint pairs across trainer process restarts."""

import json
import shutil
from pathlib import Path


def complete_checkpoint(path: Path, world_size: int) -> bool:
    if not path.is_dir() or path.is_symlink():
        return False
    if any(p.is_symlink() for p in path.rglob("*")):
        return False
    try:
        step = int(path.name.removeprefix("global_step_"))
        state = json.loads((path / "reliability-state.json").read_text())
        if state["version"] != 1 or state["step"] != step or (path / "data.pt").stat().st_size == 0:
            return False
        return all(
            (path / role / f"{kind}_world_size_{world_size}_rank_{rank}.pt").stat().st_size > 0
            for role in ("actor", "critic")
            for kind in ("model", "optim", "extra_state")
            for rank in range(world_size)
        )
    except (OSError, ValueError, KeyError):
        return False


def retain_complete_checkpoints(root: Path, current_step: int, world_size: int, keep: int = 2):
    """Prune only role directories after the new complete pair is durable.

    veRL tracks saves in memory, so an otherwise correct resume accumulates the
    previous process's checkpoints. Metadata and audit evidence remain on disk.
    Each independent trainer must own its checkpoint root exclusively.
    """
    root = Path(root).absolute()
    if keep < 2 or world_size < 1 or root.resolve() != root:
        raise ValueError("Retention requires a canonical root and at least two recovery points")
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Checkpoint root must not traverse symlinks")
    current = root / f"global_step_{current_step}"
    if not complete_checkpoint(current, world_size):
        raise RuntimeError("New checkpoint pair is incomplete; no historical checkpoint removed")
    complete = sorted(
        (p for p in root.glob("global_step_*") if complete_checkpoint(p, world_size)),
        key=lambda p: int(p.name.removeprefix("global_step_")),
    )
    if complete[-1] != current:
        raise RuntimeError("Refusing retention from a stale training step")
    removed = []
    for old in complete[:-keep]:
        if old.parent != root or old.resolve() != old:
            raise RuntimeError("Checkpoint escaped its owned root")
        # Recheck before the destructive operation; never follow symlinks.
        if not complete_checkpoint(old, world_size):
            raise RuntimeError("Historical checkpoint changed during retention")
        for role in ("actor", "critic"):
            shutil.rmtree(old / role)
        removed.append(str(old))
    report = {"step": current_step, "keep": keep, "removed_role_directories_from": removed,
              "retained": [str(p) for p in complete[-keep:]]}
    (root / f"retention-step-{current_step}.json").write_text(json.dumps(report, indent=2))
    return report
