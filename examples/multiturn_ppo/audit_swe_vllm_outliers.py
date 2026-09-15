# Copyright (c) Microsoft. All rights reserved.

"""Recheck fixed logprob outliers on fresh vLLM weights with prefix caching off."""

import argparse
import json
import os
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-check", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "1", "2", "3"}
    assert not args.output.exists()
    records = json.loads(args.baseline.read_text())["records"]
    assert all(len(row["token_ids"]) < 16384 for row in records)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=16384,
        max_num_batched_tokens=16384,
        max_num_seqs=1,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        generation_config="vllm",
        seed=42,
    )
    result = []
    for row in records:
        ids = row["token_ids"]
        target = ids[-1]
        forced = llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1),
            use_tqdm=False,
        )[0]
        next_token = llm.generate(
            [{"prompt_token_ids": ids[:-1]}],
            SamplingParams(max_tokens=1, temperature=0, logprobs=20),
            use_tqdm=False,
        )[0].outputs[0]
        candidate = next_token.logprobs[0].get(target)
        record = {
            "rid": row["rid"],
            "turn": row["turn"],
            "target": target,
            "token_text": row["token_text"],
            "original_sampling_logprob": row["logprobs"][-1],
            "training_logprob": row["training_logprob"],
            "fresh_prefill_logprob": forced.prompt_logprobs[-1][target].logprob,
            "fresh_next_token_logprob": candidate.logprob if candidate else None,
            "fresh_greedy_token": next_token.token_ids[0],
        }
        if args.generation_check:
            prompt = ids[: row["prompt_length"] - row["source_position"]]
            sampled = llm.generate(
                [{"prompt_token_ids": prompt}],
                SamplingParams(max_tokens=512, temperature=1, top_p=1, top_k=-1, seed=42, logprobs=1),
                use_tqdm=False,
            )[0].outputs[0]
            combined = prompt + list(sampled.token_ids)
            replay = llm.generate(
                [{"prompt_token_ids": combined}],
                SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1),
                use_tqdm=False,
            )[0]
            sampling = [p[t].logprob for p, t in zip(sampled.logprobs, sampled.token_ids, strict=True)]
            replayed = [replay.prompt_logprobs[len(prompt) + i][t].logprob for i, t in enumerate(sampled.token_ids)]
            difference = [abs(a - b) for a, b in zip(sampling, replayed, strict=True)]
            record["independent_generation_check"] = {
                "tokens": len(difference),
                "mae": sum(difference) / len(difference),
                "maximum": max(difference),
                "over_one": sum(d > 1 for d in difference),
                "token_ids": list(sampled.token_ids),
                "sampling_logprobs": sampling,
                "replayed_logprobs": replayed,
            }
        result.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "independent_generation_check"}), flush=True)
        if args.generation_check:
            summary = {
                k: v
                for k, v in record["independent_generation_check"].items()
                if k not in {"token_ids", "sampling_logprobs", "replayed_logprobs"}
            }
            print(json.dumps(summary), flush=True)
    with args.output.open("x") as file:
        json.dump(
            {
                "model": args.model,
                "prefix_caching": False,
                "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                "records": result,
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
