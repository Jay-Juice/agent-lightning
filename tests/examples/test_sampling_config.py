# Copyright (c) Microsoft. All rights reserved.

import pytest

from examples.multiturn_ppo.sampling_config import proxy_overrides


def test_training_and_validation_overrides_are_not_silent_defaults():
    result = proxy_overrides(
        {"temperature": 0.85, "top_k": -1, "top_p": 1, "val_kwargs": {"temperature": 0.4, "top_k": 20, "top_p": 0.9}}
    )
    assert "default_proxy.train.temperature=0.85" in result
    assert "default_proxy.val.temperature=0.4" in result
    assert "++default_proxy.val.top_k=20" in result
    assert "++default_proxy.val.top_p=0.9" in result


def test_greedy_validation_and_invalid_greedy_ppo():
    cfg = {"temperature": 1, "val_kwargs": {"temperature": 0.7, "do_sample": False}}
    assert "default_proxy.val.temperature=0.0" in proxy_overrides(cfg)
    with pytest.raises(ValueError, match="positive sampling"):
        proxy_overrides(dict(cfg, do_sample=False))
