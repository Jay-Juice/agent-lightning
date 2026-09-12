# Copyright (c) Microsoft. All rights reserved.
from agentlightning.server.proxy import ProxyRouter


def test_explicit_sampling_config_reaches_backend():
    proxy = ProxyRouter(
        {
            "model_name": "test",
            "train": {"temperature": 0.85, "top_p": 1, "top_k": -1},
            "val": {"temperature": 0, "top_p": 0.9, "top_k": 20},
        }
    )
    train = proxy.prepare_body({"temperature": 1, "top_p": 0.5}, "train")
    val = proxy.prepare_body({"temperature": 1}, "val")
    assert (train["temperature"], train["top_p"], train["top_k"]) == (0.85, 1, -1)
    assert (val["temperature"], val["top_p"], val["top_k"]) == (0, 0.9, 20)


def test_legacy_config_preserves_request_top_p():
    proxy = ProxyRouter({"model_name": "test", "train": {"temperature": 1}, "val": {"temperature": 0.7}})
    assert proxy.prepare_body({"top_p": 0.9}, "train")["top_p"] == 0.9
