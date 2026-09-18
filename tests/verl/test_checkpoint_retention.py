import json

import pytest

from agentlightning.verl.checkpoint_retention import complete_checkpoint, retain_complete_checkpoints


def checkpoint(root, step):
    p = root / f"global_step_{step}"
    p.mkdir()
    (p / "data.pt").write_bytes(b"dataloader")
    (p / "reliability-state.json").write_text(json.dumps({"version": 1, "step": step}))
    for role in ("actor", "critic"):
        (p / role).mkdir()
        for kind in ("model", "optim", "extra_state"):
            for rank in range(4):
                (p / role / f"{kind}_world_size_4_rank_{rank}.pt").write_bytes(b"state")
    return p


def test_resumed_save_rotates_previous_process_checkpoint_and_preserves_evidence(tmp_path):
    old = checkpoint(tmp_path, 15)
    seed = checkpoint(tmp_path, 20)
    current = checkpoint(tmp_path, 25)
    report = retain_complete_checkpoints(tmp_path, 25, 4)
    assert report["removed_role_directories_from"] == [str(old)]
    assert (old / "data.pt").exists() and not (old / "actor").exists()
    assert complete_checkpoint(seed, 4) and complete_checkpoint(current, 4)
    checkpoint(tmp_path, 30)
    retain_complete_checkpoints(tmp_path, 30, 4)
    assert not (seed / "critic").exists()


@pytest.mark.parametrize("damage", ["rank", "marker", "empty"])
def test_incomplete_new_pair_does_not_prune(damage, tmp_path):
    old = checkpoint(tmp_path, 15)
    checkpoint(tmp_path, 20)
    new = checkpoint(tmp_path, 25)
    f = new / "critic/optim_world_size_4_rank_3.pt"
    if damage == "rank":
        f.unlink()
    elif damage == "marker":
        (new / "reliability-state.json").unlink()
    else:
        f.write_bytes(b"")
    with pytest.raises(RuntimeError, match="incomplete"):
        retain_complete_checkpoints(tmp_path, 25, 4)
    assert complete_checkpoint(old, 4)


def test_symlink_is_not_a_retention_candidate(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    real = checkpoint(outside, 1)
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "global_step_1").symlink_to(real, target_is_directory=True)
    checkpoint(owned, 20)
    checkpoint(owned, 25)
    retain_complete_checkpoints(owned, 25, 4)
    assert complete_checkpoint(real, 4)


def test_stale_writer_never_prunes_newer_checkpoint(tmp_path):
    checkpoint(tmp_path, 20)
    checkpoint(tmp_path, 25)
    with pytest.raises(RuntimeError, match="stale"):
        retain_complete_checkpoints(tmp_path, 20, 4)
