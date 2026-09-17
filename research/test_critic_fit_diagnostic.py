"""CPU-only unittest coverage for the paired critic diagnostic.

Run from the checkout: python -m unittest discover -s research -p test_critic_fit_diagnostic.py -v
No model download, distributed initialization, or GPU allocation is performed.
"""

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f"CPU tensor dependencies unavailable: {exc}") from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from critic_fit_data import local_batch, prediction_sums, reduce_prediction_sums, split_episodes
    from critic_fit_diagnostic import find_value_head, initialize_head
finally:
    sys.path.pop(0)


def episode_snapshot():
    rows, ids, turns, lengths, terminals = [], [], [], [], []
    for reward in (0, 1):
        for episode in range(4):
            n_calls = 1 + episode % 3
            for turn in range(n_calls):
                rows.append([float(reward)] * 3)
                ids.append(f"label-{reward}-episode-{episode}")
                turns.append(turn)
                lengths.append(n_calls)
                terminals.append(turn == n_calls - 1)
    # A duplicate padding row must neither create a new episode nor make its
    # source episode appear to contain a repeated terminal call.
    rows.append(list(rows[-1]))
    ids.append(ids[-1])
    turns.append(turns[-1])
    lengths.append(lengths[-1])
    terminals.append(True)
    n = len(rows)
    tensors = {
        "response_mask": torch.tensor([[1, 0, 1]] * n),
        "returns": torch.tensor(rows),
        "responses": torch.tensor([[2, 0, 3]] * n),
        "input_ids": torch.arange(n * 7).reshape(n, 7),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0, 1]] * n),
        "position_ids": torch.tensor([[0, 0, 1, 2, 3, 3, 4]] * n),
    }
    metadata = {
        "rollout_id_list": ids,
        "turn_index_list": turns,
        "episode_turn_count": lengths,
        "episode_terminal": terminals,
        "is_pad": [False] * (n - 1) + [True],
    }
    return tensors, metadata


class EpisodeSplitTests(unittest.TestCase):
    def test_complete_episodes_do_not_leak_across_splits(self):
        tensors, metadata = episode_snapshot()
        original_meta = copy.deepcopy(metadata)
        original_tensors = {k: v.clone() for k, v in tensors.items()}
        split = split_episodes(tensors, metadata, seed=1729)
        fit, check = set(split["fit_episodes"]), set(split["check_episodes"])
        self.assertFalse(fit & check)
        self.assertEqual(fit | check, set(metadata["rollout_id_list"]))
        fit_rows, check_rows = set(split["fit_rows"]), set(split["check_rows"])
        self.assertFalse(fit_rows & check_rows)
        self.assertEqual(fit_rows | check_rows, set(range(len(metadata["is_pad"]) - 1)))
        for uid in fit | check:
            all_calls = {
                i for i, current in enumerate(metadata["rollout_id_list"])
                if current == uid and not metadata["is_pad"][i]
            }
            self.assertTrue(all_calls <= (fit_rows if uid in fit else check_rows))
        self.assertEqual(split["fit_positive_episodes"], 3)
        self.assertEqual(split["check_positive_episodes"], 1)
        self.assertEqual(split, split_episodes(tensors, metadata, seed=1729))
        self.assertEqual(metadata, original_meta)
        for key, value in tensors.items():
            torch.testing.assert_close(value, original_tensors[key], rtol=0, atol=0)

    def test_train_constant_uses_fit_tokens_only(self):
        tensors, metadata = episode_snapshot()
        # Different action lengths expose accidental episode/call weighting.
        for row, uid in enumerate(metadata["rollout_id_list"]):
            if uid.startswith("label-1"):
                tensors["response_mask"][row] = torch.tensor([1, 1, 1])
        split = split_episodes(tensors, metadata)
        fit_rows = split["fit_rows"]
        selected = tensors["returns"][fit_rows][tensors["response_mask"][fit_rows].bool()]
        self.assertEqual(split["train_target_mean"], selected.double().mean().item())
        self.assertNotEqual(split["train_target_mean"], .5)

    def test_invalid_or_incomplete_episodes_are_rejected(self):
        for corruption in ("turn_index_list", "episode_turn_count", "episode_terminal"):
            tensors, metadata = episode_snapshot()
            if corruption == "episode_terminal":
                metadata[corruption][0] = False
            else:
                metadata[corruption][0] = 99
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                split_episodes(tensors, metadata)
        tensors, metadata = episode_snapshot()
        tensors["response_mask"][0].zero_()
        with self.assertRaisesRegex(ValueError, "Empty action"):
            split_episodes(tensors, metadata)
        tensors, metadata = episode_snapshot()
        tensors["returns"][0, 2] = .25
        with self.assertRaisesRegex(ValueError, "binary targets"):
            split_episodes(tensors, metadata)

    def test_distributed_local_batches_pad_without_duplicating_real_rows(self):
        tensors, _ = episode_snapshot()
        rows = [0, 2, 4, 6, 8]
        observed, padded = [], []
        for rank in range(4):
            batch = local_batch(tensors, rows, rank, 4)
            self.assertEqual(len(batch.batch["responses"]), 2)
            self.assertEqual(batch.meta_info["global_token_num"], tensors["attention_mask"][rows].sum(-1).tolist())
            for record, is_pad in zip(batch.batch["input_ids"], batch.non_tensor_batch["is_pad"], strict=True):
                (padded if is_pad else observed).append(int(record[0]) // 7)
            batch.batch["input_ids"].zero_()
        self.assertEqual(observed, rows)
        self.assertEqual(padded, [rows[-1]] * 3)
        self.assertNotEqual(tensors["input_ids"][2, 0].item(), 0)


class PredictionStatisticsTests(unittest.TestCase):
    @staticmethod
    def sample():
        # 3 real tokens over 2 calls plus an intentionally poisonous pad row.
        values = torch.tensor([[0., float("nan"), 2.], [.5, float("nan"), float("nan")], [99., 99., 99.]])
        batch = SimpleNamespace(
            batch={
                "response_mask": torch.tensor([[1, 0, 1], [1, 0, 0], [1, 1, 1]]),
                "returns": torch.tensor([[0., 99., 1.], [1., 99., 99.], [99., 99., 99.]]),
            },
            non_tensor_batch={"is_pad": np.array([False, False, True])},
        )
        return values, batch

    def test_statistics_are_exact_and_padding_is_excluded(self):
        values, batch = self.sample()
        original_mask = batch.batch["response_mask"].clone()
        original_values = values.clone()
        result = reduce_prediction_sums([prediction_sums(values, batch, train_mean=.25)])
        all_stats = result["all"]
        self.assertEqual(all_stats["count"], 3)
        self.assertAlmostEqual(all_stats["mean"], 5 / 6)
        self.assertAlmostEqual(all_stats["mse"], 5 / 12)
        self.assertAlmostEqual(all_stats["return_variance"], 2 / 9)
        self.assertAlmostEqual(all_stats["zero_constant_mse"], 2 / 3)
        self.assertAlmostEqual(all_stats["train_constant_mse"], 19 / 48)
        self.assertAlmostEqual(all_stats["explained_variance"], -.75)
        self.assertEqual(result["success"]["count"], 2)
        self.assertEqual(result["failure"]["count"], 1)
        self.assertEqual(result["first_call_token"]["count"], 2)
        self.assertAlmostEqual(result["first_call_token"]["mse"], .125)
        self.assertAlmostEqual(result["last_call_token"]["mse"], .625)
        self.assertAlmostEqual(result["call_mse"]["mse"], .375)
        self.assertNotIn("explained_variance", result["success"])
        torch.testing.assert_close(batch.batch["response_mask"], original_mask, rtol=0, atol=0)
        torch.testing.assert_close(values, original_values, rtol=0, atol=0, equal_nan=True)

    def test_sufficient_statistics_merge_matches_single_batch(self):
        values, batch = self.sample()
        parts = []
        for start, end in ((0, 1), (1, 3)):
            shard = SimpleNamespace(
                batch={key: value[start:end] for key, value in batch.batch.items()},
                non_tensor_batch={"is_pad": batch.non_tensor_batch["is_pad"][start:end]},
            )
            parts.append(prediction_sums(values[start:end], shard, .25))
        self.assertEqual(
            reduce_prediction_sums(parts),
            reduce_prediction_sums([prediction_sums(values, batch, .25)]),
        )

    def test_empty_groups_omit_metrics_and_real_nonfinite_prediction_fails(self):
        values, batch = self.sample()
        batch.non_tensor_batch["is_pad"][:] = True
        result = reduce_prediction_sums([prediction_sums(values, batch, .25)])
        self.assertTrue(all(stats == {"count": 0} for stats in result.values()))
        values, batch = self.sample()
        values[0, 0] = math.inf
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            prediction_sums(values, batch, .25)


class HeadInitializationTests(unittest.TestCase):
    @staticmethod
    def model():
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(3, 4)
        model.score = torch.nn.Linear(4, 1)
        return model

    def test_zero_changes_only_head_and_preserves_trainability(self):
        model = self.model()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        name, head = initialize_head(model, "zero")
        self.assertEqual(name, "score")
        self.assertIs(head, model.score)
        for parameter_name, parameter in model.named_parameters():
            if parameter_name.startswith("score."):
                self.assertTrue(torch.equal(parameter, torch.zeros_like(parameter)))
            else:
                torch.testing.assert_close(parameter, before[parameter_name], rtol=0, atol=0)
            self.assertEqual(parameter.requires_grad, requires_grad[parameter_name])

    def test_default_changes_nothing_and_invalid_heads_are_rejected(self):
        model = self.model()
        before = copy.deepcopy(model.state_dict())
        initialize_head(model, "default")
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            initialize_head(model, "unsupported")
        model.classifier = torch.nn.Linear(4, 1)
        with self.assertRaisesRegex(ValueError, "one identifiable"):
            find_value_head(model)
        with self.assertRaisesRegex(ValueError, "one identifiable"):
            find_value_head(torch.nn.Linear(2, 1))

    def test_zero_head_blocks_first_backward_to_backbone_but_head_learns(self):
        model = self.model()
        initialize_head(model, "zero")
        flow = {"input": [], "output": []}

        def observe(module, inputs, outputs):
            flow["input"].extend(value.detach().clone() for value in inputs if isinstance(value, torch.Tensor))
            flow["output"].extend(value.detach().clone() for value in outputs if isinstance(value, torch.Tensor))

        hook = model.score.register_full_backward_hook(observe)
        try:
            prediction = model.score(model.backbone(torch.ones(2, 3)))
            (prediction - 1).square().mean().backward()
        finally:
            hook.remove()
        self.assertTrue(flow["input"])
        self.assertTrue(all(torch.count_nonzero(value).item() == 0 for value in flow["input"]))
        self.assertTrue(any(torch.count_nonzero(value).item() > 0 for value in flow["output"]))
        self.assertTrue(
            all(torch.count_nonzero(parameter.grad).item() == 0 for parameter in model.backbone.parameters())
        )
        self.assertGreater(torch.count_nonzero(model.score.bias.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()
