# Copyright (c) Microsoft. All rights reserved.

"""Compare teacher-forced log probabilities of fixed SWE token sequences.

Unlike repeated generation, this keeps the entire evaluated token sequence
fixed even when BF16 rounding changes an argmax. Baseline must come from a
separately loaded model; comparison may target a VERL weight-synchronized server.
"""

import argparse
import json
from pathlib import Path

import httpx
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", default="auto")
    p.add_argument("--run", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    assert bool(a.run) != bool(a.baseline)
    if a.baseline:
        records = json.loads(a.baseline.read_text())["records"]
    else:
        records = []
        for rid in ["val-04-r0-original", "val-01-r0-original", "train-05-r0-original"]:
            rows = [json.loads(line) for line in (a.run / "agent" / rid / "model-calls.jsonl").read_text().splitlines()]
            for index in sorted({0, len(rows) - 1}):
                response = rows[index]["response"]
                prompt = response["prompt_token_ids"]
                action = response["choices"][0]["token_ids"]
                records.append(
                    {"rid": rid, "turn": index + 1, "prompt_length": len(prompt), "token_ids": prompt + action}
                )
    metrics = []
    with httpx.Client(timeout=300, trust_env=False) as client:
        for record in records:
            body = dict(
                model=a.model, prompt=record["token_ids"], max_tokens=1, temperature=0, echo=True, logprobs=1, seed=42
            )
            result = client.post(a.endpoint.rstrip("/") + "/completions", json=body)
            result.raise_for_status()
            logprobs = result.json()["choices"][0]["logprobs"]["token_logprobs"][: len(record["token_ids"])]
            assert len(logprobs) == len(record["token_ids"])
            start = record["prompt_length"]
            current = np.array(logprobs[start:], dtype=float)
            assert np.isfinite(current).all()
            if a.baseline:
                expected = np.array(record["logprobs"][start:], dtype=float)
                delta = np.abs(current - expected)
                m = dict(
                    rid=record["rid"],
                    turn=record["turn"],
                    tokens=len(current),
                    mae=float(delta.mean()),
                    p99=float(np.quantile(delta, 0.99)),
                    maximum=float(delta.max()),
                )
                metrics.append(m)
                record["comparison_logprobs"] = logprobs
                print(json.dumps(m), flush=True)
            else:
                record["logprobs"] = logprobs
                print(json.dumps({"rid": record["rid"], "turn": record["turn"], "tokens": len(current)}), flush=True)
    a.output.write_text(json.dumps({"records": records, "metrics": metrics, "endpoint": a.endpoint}))
    # Numeric measurements are evidence; no arbitrary tolerance declares SWE solved.


if __name__ == "__main__":
    main()
