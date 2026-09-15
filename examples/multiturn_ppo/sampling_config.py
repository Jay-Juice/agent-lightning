# Copyright (c) Microsoft. All rights reserved.

"""Resolve the same sampling distribution for the external gateway and PPO."""


def proxy_overrides(rollout):
    result = []
    for mode, config in (("train", rollout), ("val", rollout["val_kwargs"])):
        temperature = float(config["temperature"]) if config.get("do_sample", True) else 0.0
        if mode == "train" and temperature <= 0:
            raise ValueError("PPO training requires a positive sampling temperature")
        result.append(f"default_proxy.{mode}.temperature={temperature}")
        for name, fallback in (("top_p", 1.0), ("top_k", -1)):
            value = config.get(name, fallback)
            result.append(f"++default_proxy.{mode}.{name}={value}")
    return result
