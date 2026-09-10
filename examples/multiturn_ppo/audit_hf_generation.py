# Copyright (c) Microsoft. All rights reserved.

"""Compare captured greedy vLLM probes with original Hugging Face weights, without AGL/veRL."""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def without_eos(ids, eos):
    return ids[:-1] if ids and ids[-1] in eos else ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"0", "1", "2", "3"}
    assert not args.output.exists()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    results = []
    for probe in json.loads(args.probes.read_text())["probes"]:
        body = probe["request"]
        assert body["model"] == args.model
        prompt = tokenizer.apply_chat_template(
            body["messages"], tokenize=True, add_generation_prompt=True, **body["chat_template_kwargs"]
        )
        assert prompt == probe["direct"]["prompt_token_ids"]
        ids = torch.tensor([prompt], device="cuda")
        with torch.inference_mode():
            output = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=body["max_tokens"],
                pad_token_id=tokenizer.pad_token_id,
            )
        actual = output[0, len(prompt) :].tolist()
        expected = probe["direct"]["choices"][0]["token_ids"]
        hf_ids, vllm_ids = without_eos(actual, eos), without_eos(expected, eos)
        common = 0
        for a, b in zip(hf_ids, vllm_ids, strict=False):
            if a != b:
                break
            common += 1
        result = {
            "messages_file": probe["messages_file"],
            "prompt_tokens_match": True,
            "generated_tokens_match": hf_ids == vllm_ids,
            "common_prefix_tokens": common,
            "hf_token_ids": actual,
            "vllm_token_ids": expected,
            "hf_text": tokenizer.decode(actual, skip_special_tokens=True),
            "vllm_text": probe["direct"]["choices"][0]["message"]["content"],
        }
        if common < min(len(hf_ids), len(vllm_ids)):
            prefix = torch.tensor([prompt + hf_ids[:common]], device="cuda")
            with torch.inference_mode():
                logits = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=False).logits
                scores = logits[0, -1].float()
            chosen = vllm_ids[common]
            top_scores, top_ids = scores.topk(5)
            result["first_divergence"] = {
                "vllm_token": chosen,
                "vllm_token_rank_under_hf": int((scores > scores[chosen]).sum().item()) + 1,
                "logit_gap_from_hf_best": float((top_scores[0] - scores[chosen]).item()),
                "hf_top5_ids": top_ids.tolist(),
                "hf_top5_logits": top_scores.tolist(),
                "hf_top5_text": [tokenizer.decode([token]) for token in top_ids.tolist()],
            }
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if "match" in k or k == "common_prefix_tokens"}), flush=True)
    args.output.write_text(
        json.dumps({"model": args.model, "dtype": "bfloat16", "attention": "sdpa", "probes": results})
    )
    # Different kernels can cause greedy divergence at near-tied logits.
    # Preserve any mismatch for investigation instead of labelling it an automatic framework defect.


if __name__ == "__main__":
    main()
