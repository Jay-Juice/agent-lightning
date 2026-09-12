# Copyright (c) Microsoft. All rights reserved.

"""Check saved SWE request history/token identity and summarize observable failure modes on CPU."""

import argparse
import json
from collections import Counter
from pathlib import Path

from sandbox import load_smith
from transformers import AutoTokenizer


def read_json(path):
    return json.loads(path.read_text())


def audit(run, tokenizer):
    counts = Counter()
    categories = Counter()
    records = []
    datasets = read_json(run / "datasets.json")
    expected = {row["instance_id"]: row for rows in datasets.values() for row in rows}
    smith = load_smith()
    config = read_json(run / "resolved-config.json")
    full_python = config["agentlightning"]["local"]["agent_class"] == "full_python_agent.FullPythonAgent"
    if full_python:
        from full_python_agent import FullPythonAgent, parse_statuses
    else:
        parse_statuses = smith.parse_test_statuses
    budgets = read_json(run / "provenance.json").get("runtime_options", {}).get("smith_budgets", {})
    obs_cap = int(budgets.get("SMITH_OBS_CHAR_CAP") or 4000)
    for file in sorted((run / "traces").glob("*.json")):
        trace = read_json(file)
        rid = trace["rollout"]["rollout_id"]
        row = trace["rollout"]["input"]
        assert trace["rollout"]["status"]["state"] == "succeeded"
        directory = run / "agent" / rid
        initial = read_json(directory / "initial-messages.json")
        expected_prompt = smith.INSTANCE_PROMPT.format(
            problem_statement=expected[row["instance_id"]]["problem_statement"]
        )
        if budgets.get("SMITH_VERIFY_SUBMISSION") == "1":
            from smith_submission import WORKFLOW_HINT

            expected_prompt += WORKFLOW_HINT
        if budgets.get("SMITH_CHECKED_EDITOR") == "1":
            from smith_submission import EDITOR_HINT

            expected_prompt += EDITOR_HINT
        if full_python:
            expected_prompt += FullPythonAgent.extra_workflow_hint
        assert initial[1]["content"] == expected_prompt
        messages = list(initial)
        turns = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
        turns = [turn for turn in turns if "response" in turn]
        requests = [
            e["data"] for events in trace["events"].values() for e in events if e["event_type"] == "model_request"
        ]
        rewards = [e["data"] for events in trace["events"].values() for e in events if e["event_type"] == "reward"]
        versions = {request["model_version"] for request in requests}
        assert len(versions) == 1, "Model version changed within an episode"
        grade = read_json(directory / "grade.json")
        assert not grade.get("reference_control", False)
        assert len(rewards) == 1 and rewards[0]["value"] == grade["reward"] == float(grade["resolved"])
        assert rewards[0]["source"] == "isolated_pytest"
        assert len(turns) == len(requests)
        local = Counter()
        for request, turn in zip(requests, turns, strict=True):
            body = request["request"]
            response = request["response"]
            local["calls"] += 1
            local["history_matches"] += body["messages"] == messages
            ids = tokenizer.apply_chat_template(
                body["messages"], tokenize=True, add_generation_prompt=True, **body.get("chat_template_kwargs", {})
            )
            local["prompt_token_matches"] += ids == response["prompt_token_ids"]
            local["response_matches"] += response["choices"][0]["message"]["content"] == turn["response"]
            finish = response["choices"][0]["finish_reason"]
            local["finish_" + finish] += 1
            if "action" not in turn:
                local["parse_failed"] += 1
                local["parse_failed_" + finish] += 1
                local["parse_block_count_" + str(len(smith._ACTION_RE.findall(turn["response"])))] += 1
            local["truncated_observations"] += len(turn.get("output", "")) > obs_cap
            messages += [
                {"role": "assistant", "content": turn["response"]},
                {"role": "user", "content": turn["observation"]},
            ]
        test_file = directory / "test-output.txt"
        statuses = parse_statuses(test_file.read_text()) if test_file.exists() else {}
        rejection = read_json(directory / "sandbox.json")["rejection"]
        if not rejection:
            for field, passed, total in (
                ("FAIL_TO_PASS", "f2p_passed", "f2p_total"),
                ("PASS_TO_PASS", "p2p_passed", "p2p_total"),
            ):
                assert grade[passed] == sum(statuses.get(node) in ("PASSED", "XFAIL") for node in row[field])
                assert grade[total] == len(row[field])
            assert grade["resolved"] == (
                grade["pytest_exit"] == 0
                and grade["f2p_passed"] == grade["f2p_total"]
                and grade["p2p_passed"] == grade["p2p_total"]
            )
        if grade["resolved"]:
            category = "resolved"
        elif rejection:
            category = "patch_rejected"
        elif not (directory / "model.patch").stat().st_size:
            category = "no_patch"
        elif not statuses:
            category = "patch_but_no_test_statuses"
        elif grade.get("f2p_passed", 0):
            category = "partial_f2p"
        else:
            category = "patch_but_zero_f2p"
        categories[category] += 1
        counts.update(local)
        records.append(
            {
                "instance_id": row["instance_id"],
                "rollout_id": rid,
                "is_train": trace["rollout"]["is_train"],
                "model_version": versions.pop(),
                "reward": grade["reward"],
                "category": category,
                "counts": dict(local),
            }
        )
    return {"run": str(run), "counts": dict(counts), "categories": dict(categories), "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tokenizers = {}
    results = []
    for run in args.run:
        model = read_json(run / "resolved-config.json")["actor_rollout_ref"]["model"]["path"]
        if model not in tokenizers:
            tokenizers[model] = AutoTokenizer.from_pretrained(model, local_files_only=True)
        result = audit(run, tokenizers[model])
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)
    with args.output.open("x") as file:
        json.dump(results, file, indent=2)
    for result in results:
        counts = result["counts"]
        assert (
            counts["calls"] == counts["history_matches"] == counts["prompt_token_matches"] == counts["response_matches"]
        )


if __name__ == "__main__":
    main()
