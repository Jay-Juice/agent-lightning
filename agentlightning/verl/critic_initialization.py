# Copyright (c) Microsoft. All rights reserved.

"""Optional fresh-model value-head initialization, before FSDP and checkpoint load.

This factory runs only while constructing a new model. veRL subsequently loads
saved model/optimizer/RNG state on resume; no load or update method is patched.
"""

import functools
import json
import os

import torch


def initialize_value_head(model, mode):
    if mode not in {"default", "zero"}:
        raise ValueError(f"Unknown critic_head_init: {mode}")
    if mode == "default":
        return None
    candidates = [(name, getattr(model, name)) for name in ("score", "classifier", "v_head")
                  if isinstance(getattr(model, name, None), torch.nn.Module)]
    if len(candidates) != 1:
        raise ValueError("Zero initialization requires exactly one identifiable value head")
    name, head = candidates[0]
    parameters = list(head.parameters())
    if not parameters or sum(p.numel() for p in parameters) > 1_000_000:
        raise ValueError("Unexpected value-head structure")
    with torch.no_grad():
        for parameter in parameters:
            parameter.zero_()
    return {"head": name, "parameters": sum(p.numel() for p in parameters),
            "meta_tensor": any(p.is_meta for p in parameters)}


def register_value_head_initialization():
    """Idempotently wrap only veRL's value-model factory in each Ray worker."""
    import verl.utils.model as model_api

    mode = os.environ.get("AGL_CRITIC_HEAD_INIT", "default")
    if mode not in {"default", "zero"}:
        raise ValueError(f"Unknown AGL_CRITIC_HEAD_INIT: {mode}")
    original = model_api.load_valuehead_model
    installed = getattr(original, "_agl_head_init_mode", None)
    if installed is not None:
        if installed != mode:
            raise RuntimeError("A worker cannot switch critic initialization modes")
        return
    if mode == "default":
        return

    @functools.wraps(original)
    def factory(*args, **kwargs):
        model = original(*args, **kwargs)
        evidence = initialize_value_head(model, mode)
        print("CRITIC_HEAD_INITIALIZATION " + json.dumps({
            "mode": mode, "phase": "fresh_model_before_fsdp_and_checkpoint_load",
            "rank": os.environ.get("RANK"), **evidence,
        }), flush=True)
        return model

    factory._agl_head_init_mode = mode
    model_api.load_valuehead_model = factory
