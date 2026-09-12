# Copyright (c) Microsoft. All rights reserved.
"""Check the installed chunked output layer before using it with copied PPO."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl")

from verl.utils.experimental.torch_functional import FusedLinearForPPO  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
@pytest.mark.parametrize("entropy_weight", [0.0, 0.01])
def test_chunked_output_probabilities_and_gradients(dtype, temperature, entropy_weight):
    torch.manual_seed(90212)
    hidden = (torch.randn(1031, 67) * 0.1).to(dtype).requires_grad_()
    vocab = (torch.randn(127, 67) * 0.1).to(dtype).requires_grad_()
    labels = torch.randint(127, (1031,))
    credit = torch.randn(1031).to(dtype)
    logits = ((hidden @ vocab.T) / temperature).float()
    log_probs = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1).to(dtype)
    entropy = (logits.logsumexp(-1) - (logits.softmax(-1) * logits).sum(-1)).to(dtype)
    objective = (log_probs * credit + entropy_weight * entropy).float().mean()
    expected = torch.autograd.grad(objective, (hidden, vocab))

    actual_log_probs, actual_entropy = FusedLinearForPPO(chunk_size=512)(hidden, vocab, labels, temperature)
    fused_objective = (actual_log_probs * credit + entropy_weight * actual_entropy).float().mean()
    actual = torch.autograd.grad(fused_objective, (hidden, vocab))
    torch.testing.assert_close(actual_log_probs, log_probs)
    torch.testing.assert_close(actual_entropy, entropy)
    for reference, observed in zip(expected, actual, strict=True):
        relative_error = (observed.float() - reference.float()).norm() / reference.float().norm()
        assert relative_error < (0.02 if dtype == torch.bfloat16 else 2e-5)
