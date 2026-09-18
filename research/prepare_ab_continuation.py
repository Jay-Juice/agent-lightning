"""CPU-only validation of both immutable seeds and their next dataloader batches."""
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.main_ppo import create_rl_sampler
from verl.utils.dataset.rl_dataset import collate_fn

from agentlightning.verl.checkpoint_retention import complete_checkpoint
from agentlightning.verl.dataset import LoadedDataset

ROOT = Path('/media/ubuntu/D1/zsj/agent-lightning-runtime/logs')
OUT = ROOT / 'ab-continuation-preflight-20260919-01'


def main():
    assert not OUT.exists(), 'Do not overwrite preflight evidence'
    OUT.mkdir()
    for arm, head, gpus, port in [('A', 'default', '0,1,2,3', 18531), ('B', 'zero', '4,5,6,7', 18541)]:
        run = ROOT / f'training-capo-swe-v2-mini128-{head}-4b-gpu47-20260918-03'
        checkpoint = run / 'checkpoints/global_step_20'
        assert (run / 'run.exit').read_text().strip() == '0'
        assert complete_checkpoint(checkpoint, 4)
        cfg = OmegaConf.create(json.loads((run / 'resolved-config.json').read_text()))
        state = torch.load(checkpoint / 'data.pt', map_location='cpu', weights_only=False)
        data = json.loads((run / 'datasets.json').read_text())
        dataset = LoadedDataset(data['train'])
        sampler = create_rl_sampler(cfg.data, dataset)
        loader = StatefulDataLoader(dataset=dataset, batch_size=32, sampler=sampler,
                                    num_workers=0, drop_last=False, collate_fn=collate_fn)
        loader.load_state_dict(state)
        expected = [str(x) for x in next(iter(loader))['data_id']]
        assert len(expected) == len(set(expected)) == 32
        assert len(loader) * 4 == 784
        # All learning settings are inherited unchanged through the same entrypoint.
        tag = f'capo-swe-v2-full-{head}-4b-gpu{gpus.replace(",", "")}-resume20-20260919-01'
        spec = {'arm': arm, 'head': head, 'gpus': gpus, 'port': port, 'tag': tag,
                'seed': str(checkpoint), 'checkpoint_root': str(checkpoint.parent),
                'expected_first_step': 21, 'expected_data_ids': expected, 'total_training_steps': 784,
                'train_count': len(dataset), 'validation_count': len(data['validation']),
                'dataloader_state_keys': list(state), 'state_steps_yielded': state.get('_num_yielded')}
        (OUT / f'{arm}.json').write_text(json.dumps(spec, indent=2))
        print(json.dumps(spec))
    (OUT / 'completed.json').write_text(json.dumps({'passed': True, 'arms': ['A', 'B']}))


if __name__ == '__main__':
    main()
