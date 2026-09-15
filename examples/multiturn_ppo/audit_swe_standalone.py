# Copyright (c) Microsoft. All rights reserved.

"""Full SWE episodes on independently loaded vLLM weights, without Lightning or PPO.

Two optional agent profiles use the same sandbox, data, grader and seeds. The
worker calls the production agent, intercepting only the model client sampling
arguments and terminal event delivery. No reference patch is passed to a model.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def worker(job_path):
    import httpx
    import smith_docker_agent as agent

    job = json.loads(Path(job_path).read_text())
    run = Path(job["run"])
    rid = job["rid"]
    os.environ.update(
        AGL_TASK=json.dumps(job["row"]),
        AGL_KEY="standalone-local-no-auth",
        AGL_EVENT_URL=f"http://127.0.0.1:1/api/rollouts/{rid}/events",
        AGL_RUN_DIR=str(run),
        AGL_TRAIN_MODEL=job["model"],
        AGL_OPENAI_BASE_URL=job["endpoint"],
        SMITH_MAX_TURNS=str(job["turns"]),
        SMITH_MAX_TOKENS=str(job["response"]),
        SMITH_CONTEXT=str(job["context"]),
        SMITH_OBS_CHAR_CAP=str(job["obs"]),
        SMITH_MODEL_TIMEOUT="600",
        SMITH_VERIFY_SUBMISSION=str(int(job["profile"] in ("verified", "verified_filtered"))),
        SMITH_ALLOW_REPRO_FILES=str(int(job["profile"] in ("filtered", "verified_filtered"))),
        SMITH_CHECK_SYNTAX=str(int(job.get("check_syntax", False))),
        SMITH_CHECKED_EDITOR=str(int(job.get("checked_editor", False))),
    )
    original_client = agent.OpenAI

    def client(**kwargs):
        llm = original_client(**kwargs)
        create = llm.chat.completions.create
        count = 0

        def completion(**body):
            nonlocal count
            body.update(temperature=job["temperature"], seed=job["seed"] + count, top_p=job.get("top_p", 1.0))
            body.setdefault("extra_body", {}).update(top_k=job.get("top_k", -1), return_token_ids=True)
            response = create(**body)
            with (run / "agent" / rid / "model-calls.jsonl").open("a") as f:
                f.write(json.dumps({"request": body, "response": response.model_dump()}) + "\n")
            count += 1
            return response

        llm.chat.completions.create = completion
        return llm

    agent.OpenAI = client
    original_post = agent.httpx.post

    def event(url, **kwargs):
        if url != os.environ["AGL_EVENT_URL"]:
            return original_post(url, **kwargs)
        assert kwargs["json"]["event_type"] == "reward"
        (run / "agent" / rid / "event.json").write_text(json.dumps(kwargs["json"]))
        return httpx.Response(200, request=httpx.Request("POST", url))

    agent.httpx.post = event
    agent.SmithDockerAgent().run()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--model")
    p.add_argument("--data", type=Path)
    p.add_argument(
        "--profiles",
        nargs="+",
        choices=["original", "verified", "filtered", "verified_filtered"],
        default=["original", "verified"],
    )
    p.add_argument("--splits", nargs="+", choices=["train", "val"], default=["val", "train"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--check-syntax", action="store_true")
    p.add_argument("--checked-editor", action="store_true")
    p.add_argument("--turns", type=int, default=24)
    p.add_argument("--response", type=int, default=4096)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--obs", type=int, default=6000)
    p.add_argument("--observation-budgets", nargs="+", type=int)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--tensor-parallel", type=int, choices=[1, 4], default=1)
    p.add_argument("--port", type=int, default=18410)
    a = p.parse_args()
    if a.worker:
        worker(a.worker)
        return
    assert a.output and a.model and a.data and not a.output.exists()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3", "Only physical GPUs 0-3 are authorized"
    used = subprocess.check_output(
        ["nvidia-smi", "-i", "0,1,2,3", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    assert all(int(x) < 1000 for x in used.split()), "A selected GPU is busy"
    server_count = 4 // a.tensor_parallel
    for port in range(a.port, a.port + server_count):
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))
    a.output.mkdir(parents=True)
    for name in ["agent", "jobs"]:
        (a.output / name).mkdir()
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    manifest.update(scope="zero-shot standalone vLLM, no PPO/Lightning/Ray", gpus=[0, 1, 2, 3])
    manifest["source_sha256"] = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    servers, logs = [], []
    exit_code = 1
    try:
        import httpx

        for gpu in range(server_count):
            log = (a.output / f"vllm-{gpu}.log").open("w")
            logs.append(log)
            cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                a.model,
                "--served-model-name",
                "auto",
                "--host",
                "127.0.0.1",
                "--port",
                str(a.port + gpu),
                "--dtype",
                "bfloat16",
                "--tensor-parallel-size",
                str(a.tensor_parallel),
                "--max-model-len",
                str(a.context),
                "--gpu-memory-utilization",
                "0.45",
                "--max-num-seqs",
                str(2 * a.tensor_parallel),
                "--max-num-batched-tokens",
                "4096",
                "--generation-config",
                "vllm",
                "--enforce-eager",
                "--seed",
                "42",
                "--disable-log-requests",
            ]
            devices = ",".join(str(x) for x in range(gpu * a.tensor_parallel, (gpu + 1) * a.tensor_parallel))
            servers.append(
                subprocess.Popen(
                    cmd,
                    env=dict(env, CUDA_VISIBLE_DEVICES=devices),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + 600
        pending = set(range(server_count))
        while pending:
            for gpu in list(pending):
                assert servers[gpu].poll() is None, f"vLLM {gpu} exited; inspect its log"
                try:
                    if httpx.get(f"http://127.0.0.1:{a.port + gpu}/health", timeout=2).status_code == 200:
                        pending.remove(gpu)
                except httpx.HTTPError:
                    pass
            assert time.monotonic() < deadline, "vLLM startup timeout"
            time.sleep(1)
        print(f"STANDALONE_SERVERS_READY count={server_count} physical_gpus=0,1,2,3", flush=True)
        jobs = []
        for split in a.splits:
            rows = [json.loads(line) for line in (a.data / f"{split}.jsonl").read_text().splitlines() if line.strip()]
            for repeat in range(a.repeats):
                for i, row in enumerate(rows):
                    for profile in a.profiles:
                        for obs in a.observation_budgets or [a.obs]:
                            assert obs > 0
                            rid = f"{split}-{i:02d}-r{repeat}-{profile}"
                            if a.observation_budgets:
                                rid += f"-o{obs}"
                            job = dict(
                                manifest,
                                row=row,
                                run=str(a.output),
                                rid=rid,
                                split=split,
                                obs=obs,
                                profile=profile,
                                seed=1000 + repeat * 10000 + i * 100,
                                endpoint=f"http://127.0.0.1:{a.port + len(jobs) % server_count}/v1",
                            )
                            path = a.output / "jobs" / f"{rid}.json"
                            path.write_text(json.dumps(job))
                            jobs.append(path)

        def run_job(path):
            started = time.monotonic()
            job = json.loads(path.read_text())
            with (a.output / "jobs" / f"{path.stem}.log").open("w") as f:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
                    env=dict(env, CUDA_VISIBLE_DEVICES=""),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=7200,
                )
            grade = a.output / "agent" / job["rid"] / "grade.json"
            return dict(
                rid=job["rid"],
                instance_id=job["row"]["instance_id"],
                split=job["split"],
                profile=job["profile"],
                observation_budget=job["obs"],
                exit=result.returncode,
                seconds=time.monotonic() - started,
                grade=json.loads(grade.read_text()) if grade.exists() else None,
            )

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                (a.output / "results.json").write_text(json.dumps(results, indent=2))
                print(json.dumps(result), flush=True)
        assert all(r["exit"] == 0 and r["grade"] is not None for r in results), (
            "Infrastructure failures are not zero rewards"
        )
        exit_code = 0
    finally:
        # Only stop process groups created by this invocation, never existing jobs.
        for process in servers:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in servers:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()
        (a.output / "run.exit").write_text(str(exit_code))


if __name__ == "__main__":
    main()
