# Copyright (c) Microsoft. All rights reserved.
"""Compare fixed SWE token probabilities against Transformers, without vLLM or PPO."""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-run", type=Path)
    parser.add_argument("--failure-rollout")
    parser.add_argument("--attention", choices=["sdpa", "flash_attention_2"], default="sdpa")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "1", "2", "3"}
    assert not args.output.exists()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation=args.attention
        )
        .to("cuda")
        .eval()
    )
    results = []
    for record in json.loads(args.baseline.read_text())["records"]:
        start = record["prompt_length"]
        count = len(record["token_ids"]) - start
        ids = torch.tensor([record["token_ids"]], device="cuda")
        with torch.inference_mode():
            logits = model(input_ids=ids[:, :-1], use_cache=False, logits_to_keep=count).logits[0].float()
            actual = logits.log_softmax(-1).gather(-1, ids[0, start:, None]).squeeze(-1).cpu()
        expected = torch.tensor(record["logprobs"][start:])
        delta = (actual - expected).abs()
        assert torch.isfinite(delta).all()
        result = dict(
            rid=record["rid"],
            turn=record["turn"],
            tokens=count,
            mae=delta.mean().item(),
            p99=delta.quantile(0.99).item(),
            maximum=delta.max().item(),
            hf_logprobs=actual.tolist(),
            vllm_logprobs=expected.tolist(),
        )
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if not k.endswith("logprobs")}), flush=True)
        del logits, ids
    failures = []
    if args.failure_run:
        calls = [
            json.loads(line)
            for line in (args.failure_run / "agent" / args.failure_rollout / "model-calls.jsonl")
            .read_text()
            .splitlines()
        ]
        candidates = [
            i
            for i, call in enumerate(calls)
            if call["response"]["choices"][0]["message"]["content"].count("if matching_exceptions:") >= 5
        ]
        assert candidates, "The selected history does not contain the diagnosed repeated edit"
        for index in sorted({candidates[0], candidates[-1]}):
            call = calls[index]
            prompt = tokenizer.apply_chat_template(
                call["request"]["messages"],
                tokenize=True,
                add_generation_prompt=True,
                **call["request"].get("chat_template_kwargs", {}),
            )
            assert prompt == call["response"]["prompt_token_ids"]
            ids = torch.tensor([prompt], device="cuda")
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(generated[0, len(prompt) :], skip_special_tokens=True)
            failures.append(
                dict(
                    turn=index + 1,
                    prompt_tokens=len(prompt),
                    hf_response=text,
                    scope="Greedy fixed-history diagnosis, not a SWE score or a same-sampler comparison",
                )
            )
            print(json.dumps(failures[-1]), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                model=args.model,
                backend="Transformers BF16 " + args.attention,
                records=results,
                failure_history_probes=failures,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
