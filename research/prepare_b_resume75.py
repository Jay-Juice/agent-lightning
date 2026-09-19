"""Prepare an exact B-arm resume from the last complete step-75 checkpoint."""
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.main_ppo import create_rl_sampler
from verl.utils.dataset.rl_dataset import collate_fn

from agentlightning.verl.checkpoint_retention import complete_checkpoint
from agentlightning.verl.dataset import LoadedDataset


ROOT = Path("/media/ubuntu/D1/zsj/agent-lightning-runtime/logs")
RUN = ROOT / "training-capo-swe-v2-mini128-zero-4b-gpu47-20260918-03"
FAILED = ROOT / "training-capo-swe-v2-full-zero-4b-gpu4567-resume20-20260919-02"
OUT = ROOT / "b-resume75-preflight-20260920-01"


def main():
    assert not OUT.exists(), "Do not overwrite resume preflight evidence"
    assert (FAILED / "run.exit").read_text().strip() == "1"
    checkpoint = RUN / "checkpoints/global_step_75"
    assert complete_checkpoint(checkpoint, 4)
    cfg = OmegaConf.create(json.loads((RUN / "resolved-config.json").read_text()))
    state = torch.load(checkpoint / "data.pt", map_location="cpu", weights_only=False)
    data = json.loads((RUN / "datasets.json").read_text())
    dataset = LoadedDataset(data["train"])
    sampler = create_rl_sampler(cfg.data, dataset)
    loader = StatefulDataLoader(
        dataset=dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )
    loader.load_state_dict(state)
    expected = [str(value) for value in next(iter(loader))["data_id"]]
    assert len(expected) == len(set(expected)) == 32
    assert len(loader) * 4 == 784
    OUT.mkdir()
    spec = {
        "arm": "B",
        "head": "zero",
        "gpus": "4,5,6,7",
        "port": 18541,
        "tag": "capo-swe-v2-full-zero-4b-gpu4567-resume75-20260920-01",
        "checkpoint": str(checkpoint),
        "checkpoint_root": str(checkpoint.parent),
        "expected_first_step": 76,
        "expected_data_ids": expected,
        "total_training_steps": 784,
        "train_count": len(dataset),
        "validation_count": len(data["validation"]),
        "state_steps_yielded": state.get("_num_yielded"),
    }
    (OUT / "B.json").write_text(json.dumps(spec, indent=2))
    (OUT / "completed.json").write_text(json.dumps({"passed": True, "step": 75}))
    print(json.dumps(spec))


if __name__ == "__main__":
    main()
