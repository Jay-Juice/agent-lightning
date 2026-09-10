# Copyright (c) Microsoft. All rights reserved.

"""Replay a failed final-turn history through original Transformers weights, without veRL/vLLM/AGL."""

import argparse
import json
import os
from contextlib import suppress
from pathlib import Path

import torch
from sandbox import load_smith
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rollout", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "1", "2", "3"}
    assert not args.output.exists()
    config = json.loads((args.run / "resolved-config.json").read_text())
    model_path = config["actor_rollout_ref"]["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    smith = load_smith()
    results = []
    for rid in args.rollout:
        trace = json.loads((args.run / "traces" / f"{rid}.json").read_text())
        requests = [
            e["data"] for events in trace["events"].values() for e in events if e["event_type"] == "model_request"
        ]
        last = requests[-1]
        body = last["request"]
        ids = tokenizer.apply_chat_template(
            body["messages"], tokenize=True, add_generation_prompt=True, **body["chat_template_kwargs"]
        )
        assert ids == last["response"]["prompt_token_ids"]
        inputs = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                do_sample=False,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
            )
        content = tokenizer.decode(output[0, len(ids) :], skip_special_tokens=True)
        original = last["response"]["choices"][0]["message"]["content"]
        action = None
        with suppress(smith.FormatError):
            action = smith.parse_action(content)
        result = {
            "instance_id": trace["rollout"]["input"]["instance_id"],
            "rollout_id": rid,
            "history_messages": len(body["messages"]),
            "prompt_tokens": len(ids),
            "prompt_token_ids_match": True,
            "original_response": original,
            "hf_response": content,
            "hf_action": action,
            "same_action_as_failed_original": action is not None and action == smith.parse_action(original),
            "scope": "Failure-history diagnosis: HF greedy vs original sampled turn; not a new SWE score",
        }
        results.append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k not in {"original_response", "hf_response"}}), flush=True
        )
    args.output.write_text(json.dumps({"model": model_path, "probes": results}, indent=2))


if __name__ == "__main__":
    main()
