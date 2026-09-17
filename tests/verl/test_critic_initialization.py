"""Check initialization changes only fresh heads and never the resume path."""
# ruff: noqa: E402

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from agentlightning.verl.critic_initialization import initialize_value_head, register_value_head_initialization


class TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.score = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.score(self.backbone(x))


def test_zero_changes_only_head_and_preserves_trainability():
    model = TinyCritic()
    before = deepcopy(model.state_dict())
    evidence = initialize_value_head(model, "zero")
    assert evidence["head"] == "score"
    for name, tensor in model.state_dict().items():
        if name.startswith("score."):
            assert not tensor.count_nonzero()
        else:
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert all(p.requires_grad for p in model.parameters())


def test_default_and_invalid_modes():
    model = TinyCritic()
    before = deepcopy(model.state_dict())
    assert initialize_value_head(model, "default") is None
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    with pytest.raises(ValueError):
        initialize_value_head(model, "unknown")
    model.v_head = torch.nn.Linear(4, 1)
    with pytest.raises(ValueError, match="exactly one"):
        initialize_value_head(model, "zero")


def test_factory_only_initializes_fresh_model_and_load_preserves_trained_head(monkeypatch):
    pytest.importorskip("verl")
    import verl.utils.model as model_api

    monkeypatch.setattr(model_api, "load_valuehead_model", lambda: TinyCritic())
    monkeypatch.setenv("AGL_CRITIC_HEAD_INIT", "zero")
    register_value_head_initialization()
    factory = model_api.load_valuehead_model
    register_value_head_initialization()
    assert model_api.load_valuehead_model is factory
    model = factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    inputs = torch.randn(8, 4)
    loss = (model(inputs) - 1).square().mean()
    loss.backward()
    optimizer.step()
    saved = deepcopy(model.state_dict())
    assert saved["score.weight"].count_nonzero()
    restored = factory()
    assert not restored.score.weight.count_nonzero()
    restored.load_state_dict(saved)
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, saved[name], rtol=0, atol=0)
    # Registering the same factory again cannot mutate existing/restored models.
    register_value_head_initialization()
    torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    monkeypatch.setenv("AGL_CRITIC_HEAD_INIT", "default")
    with pytest.raises(RuntimeError, match="cannot switch"):
        register_value_head_initialization()
