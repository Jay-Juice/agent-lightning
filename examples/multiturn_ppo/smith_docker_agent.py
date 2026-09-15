# Copyright (c) Microsoft. All rights reserved.

"""Local AGL rollout worker: host model client, unprivileged offline Docker actions.

SWE-smith branches are prepared by trusted code. Grading takes place in a second,
fresh container, after agent patch export. Neither container has host mounts.
"""

import io
import json
import os
import re
import tarfile
import time
from pathlib import Path, PurePosixPath

import docker
import httpx
from openai import OpenAI
from sandbox import Sandbox, load_smith, validate_task


def agent_task(row):
    result = {k: row[k] for k in ("instance_id", "problem_statement", "image")}
    validate_task(result)
    if "@sha256:" not in result["image"]:
        raise ValueError("Training images must be pinned by digest")
    return result


class SmithSandbox(Sandbox):
    def install_editor(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            content = Path(__file__).with_name("smith_edit.py").read_bytes()
            info = tarfile.TarInfo("agl-edit")
            info.size, info.mode = len(content), 0o755
            tar.addfile(info, io.BytesIO(content))
        if not self.container.put_archive("/usr/local/bin", buf.getvalue()):
            raise RuntimeError("Could not install the checked source editor")

    def prepare(self):
        iid = self.container.labels["agl.instance_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", iid):
            raise ValueError("Invalid branch name")
        self.root(["git", "checkout", iid])
        return super().prepare()

    def copy_bytes(self, content, name="candidate.patch"):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(content))
        if not self.container.put_archive("/root", buf.getvalue()):
            raise RuntimeError("Could not stage patch")


def test_nodes(row, max_tests=200):
    f2p, p2p = row["FAIL_TO_PASS"], row["PASS_TO_PASS"]
    if not f2p or not isinstance(f2p, list) or not isinstance(p2p, list):
        raise ValueError("Missing test metadata")
    if max_tests is not None and len(f2p) + len(p2p) > max_tests:
        raise ValueError("Single-machine pilot limits grading to 200 tests")
    for node in f2p + p2p:
        path = node.split("::", 1)[0]
        if path.startswith(("/", "-")) or ".." in PurePosixPath(path).parts:
            raise ValueError("Invalid test path")
    return f2p, p2p


def grade(row, patch, output_dir, *, reference=False):
    f2p, p2p = test_nodes(row)
    client = docker.from_env(timeout=120)
    box = SmithSandbox(client, agent_task(row), output_dir.name + "-grade")
    try:
        preparation = box.prepare()
        commits = box.git("log", "-3", "--format=%s").splitlines()
        if commits[:2] != ["Remove F2P Tests", "Bug Patch"]:
            raise RuntimeError(f"Unexpected SWE-smith branch layout: {commits}")
        if reference:
            # Positive control only: restore pre-bug sources before restoring tests.
            # This code is never used in a model rollout or for its reward.
            patch = box.git("diff", "HEAD~1", "HEAD~2", "--binary")
        if patch:
            box.copy_bytes(patch.encode())
            box.git("apply", "--check", "/root/candidate.patch")
            box.git("apply", "/root/candidate.patch")
        paths = sorted({n.split("::", 1)[0] for n in f2p + p2p})
        box.git("checkout", "HEAD~1", "--", *paths)
        # Tests and runner live only in this separate grading container.
        argv = [
            "/usr/bin/timeout",
            "--kill-after=5",
            "300",
            "/opt/miniconda3/envs/testbed/bin/python",
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-rA",
            "-p",
            "no:cacheprovider",
            *dict.fromkeys(f2p + p2p),
        ]
        result = box.container.exec_run(
            argv,
            user="65534:65534",
            workdir="/testbed",
            environment={
                "HOME": "/tmp/agl-home",
                "PYTHONPATH": "/testbed",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PATH": "/opt/miniconda3/envs/testbed/bin:/usr/bin:/bin",
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
            },
        )
        output = result.output.decode(errors="replace")
        (output_dir / "test-output.txt").write_text(output)
        statuses = load_smith().parse_test_statuses(output)
        pass_f = sum(statuses.get(n) in ("PASSED", "XFAIL") for n in f2p)
        pass_p = sum(statuses.get(n) in ("PASSED", "XFAIL") for n in p2p)
        resolved = result.exit_code == 0 and pass_f == len(f2p) and pass_p == len(p2p)
        report = {
            "reward": float(resolved),
            "resolved": resolved,
            "pytest_exit": result.exit_code,
            "f2p_passed": pass_f,
            "f2p_total": len(f2p),
            "p2p_passed": pass_p,
            "p2p_total": len(p2p),
            "baseline": preparation,
            "reference_control": reference,
        }
        (output_dir / "grade.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        box.close()
        client.close()


def query_completion(llm, smith, *, gateway_wait_s=600.0, gateway_poll_s=5.0, **kwargs):
    """Reuse upstream error classifiers; paused requests do not consume a turn.

    Keep ordinary transport failures visible to the rollout manager rather than
    inventing empty assistant responses that lack sampled tokens for PPO.
    """
    if gateway_wait_s < 0 or gateway_poll_s <= 0:
        raise ValueError("Invalid gateway wait or poll interval")
    deadline = time.monotonic() + gateway_wait_s
    while True:
        try:
            return llm.chat.completions.create(**kwargs)
        except Exception as exc:
            if smith._is_context_overflow(exc):
                raise smith._ContextOverflow(str(exc)) from exc
            if not smith._is_gateway_paused(exc):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("SWE gateway remained paused beyond its wait budget") from exc
            time.sleep(min(gateway_poll_s, remaining))


class SmithDockerAgent:
    max_grading_tests = 200
    extra_workflow_hint = ""

    def action_rejection(self, smith, action):
        return smith._forbidden_action(action)

    def run(self):
        rollout_started = time.monotonic()
        row = json.loads(os.environ["AGL_TASK"])
        task = agent_task(row)
        test_nodes(row, max_tests=self.max_grading_tests)
        key = os.environ["AGL_KEY"]
        event_url = os.environ["AGL_EVENT_URL"]
        rid = event_url.split("/rollouts/")[1].split("/")[0]
        directory = Path(os.environ["AGL_RUN_DIR"]) / "agent" / rid
        directory.mkdir()
        smith = load_smith()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(os.environ["AGL_TRAIN_MODEL"], local_files_only=True)
        context_limit = int(os.environ.get("SMITH_CONTEXT", "12288"))
        obs_cap = int(os.environ.get("SMITH_OBS_CHAR_CAP", "4000"))
        model_timeout = int(os.environ.get("SMITH_MODEL_TIMEOUT", "240"))
        max_format_errors = int(os.environ.get("SMITH_MAX_FORMAT_ERRORS", "3"))
        gateway_wait_s = float(os.environ.get("SMITH_GATEWAY_WAIT_S", "600"))
        agent_wall_timeout = float(os.environ.get("SMITH_AGENT_WALL_TIMEOUT", "0"))
        if obs_cap <= 0 or model_timeout <= 0 or agent_wall_timeout < 0:
            raise ValueError("Observation/model budgets must be positive and the agent wall timeout nonnegative")
        agent_deadline = rollout_started + agent_wall_timeout if agent_wall_timeout else None
        messages = [
            {"role": "system", "content": smith.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": smith.INSTANCE_PROMPT.format(problem_statement=task["problem_statement"]),
            },
        ]
        verified_submission = os.environ.get("SMITH_VERIFY_SUBMISSION", "0") == "1"
        allow_new_repro = os.environ.get("SMITH_ALLOW_REPRO_FILES", "0") == "1"
        check_syntax = os.environ.get("SMITH_CHECK_SYNTAX", "0") == "1"
        use_editor = os.environ.get("SMITH_CHECKED_EDITOR", "0") == "1"
        if verified_submission:
            from smith_submission import WORKFLOW_HINT

            messages[-1]["content"] += WORKFLOW_HINT
        if use_editor:
            from smith_submission import EDITOR_HINT

            messages[-1]["content"] += EDITOR_HINT
        messages[-1]["content"] += self.extra_workflow_hint
        (directory / "initial-messages.json").write_text(json.dumps(messages))
        client = docker.from_env(timeout=120)
        box = SmithSandbox(client, task, rid)
        stop_reason = "turn_budget"
        n_format_errors = 0
        try:
            preparation = box.prepare()
            if use_editor:
                box.install_editor()
            with (
                OpenAI(
                    base_url=os.environ["AGL_OPENAI_BASE_URL"],
                    api_key=key,
                    timeout=model_timeout,
                    max_retries=2,
                ) as llm,
                (directory / "trajectory.jsonl").open("w") as trace,
            ):
                for turn in range(int(os.environ.get("SMITH_MAX_TURNS", "8"))):
                    # The controller owns a larger hard timeout. Stop launching
                    # model calls at the previous 3600 s boundary so patch
                    # export, isolated grading and reward delivery can finish
                    # instead of leaving the rollout without a score.
                    if agent_deadline is not None and time.monotonic() >= agent_deadline:
                        stop_reason = "wall_time_budget"
                        trace.write(
                            json.dumps(
                                {
                                    "stop_reason": stop_reason,
                                    "elapsed_seconds": time.monotonic() - rollout_started,
                                }
                            )
                            + "\n"
                        )
                        trace.flush()
                        break
                    token_count = len(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                    if token_count + int(os.environ.get("SMITH_MAX_TOKENS", "768")) > context_limit:
                        stop_reason = "context_budget"
                        trace.write(
                            json.dumps(
                                {
                                    "stop_reason": "context_limit",
                                    "prompt_tokens": token_count,
                                }
                            )
                            + "\n"
                        )
                        break
                    try:
                        response = query_completion(
                            llm,
                            smith,
                            gateway_wait_s=gateway_wait_s,
                            model="auto",
                            messages=messages,
                            max_tokens=int(os.environ.get("SMITH_MAX_TOKENS", "768")),
                            temperature=1.0,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                            **({"seed": int(row["_agl_sampling_seed"]) + turn} if "_agl_sampling_seed" in row else {}),
                        )
                    except smith._ContextOverflow:
                        stop_reason = "context_budget"
                        trace.write(
                            json.dumps({"stop_reason": "server_context_limit", "prompt_tokens": token_count}) + "\n"
                        )
                        trace.flush()
                        break
                    choice = response.choices[0]
                    content = choice.message.content or ""
                    messages.append({"role": "assistant", "content": content})
                    record = {"turn": turn + 1, "response": content}
                    submitted = False
                    try:
                        action = smith.parse_action(content)
                    except smith.FormatError as exc:
                        # Match upstream run_agent_loop: count consecutive parse
                        # failures, reset on any parseable action, and allow <=0
                        # to disable the limit. Keep the last completion in the
                        # trace before ending so PPO includes its response tokens.
                        n_format_errors += 1
                        record["consecutive_format_errors"] = n_format_errors
                        if 0 < max_format_errors <= n_format_errors:
                            stop_reason = "format_errors"
                            record.update(stop_reason=stop_reason, submitted=False)
                            trace.write(json.dumps(record) + "\n")
                            trace.flush()
                            break
                        observation = smith.format_error_message(exc.n_actions, choice.finish_reason)
                    else:
                        n_format_errors = 0
                        blocked = self.action_rejection(smith, action)
                        rc, out = (1, blocked) if blocked else box.execute(action)
                        observation = smith.render_observation(rc, out, obs_cap)
                        submitted = not blocked and smith.is_submission(out)
                        record.update(action=action, returncode=rc, output=out)
                        if submitted and verified_submission:
                            from smith_submission import submission_feedback, syntax_check_command

                            candidate = box.export_patch(allow_new_repro=allow_new_repro)
                            feedback = submission_feedback(*candidate)
                            if feedback is None and check_syntax:
                                included = [p for p in candidate[1] if p not in box.excluded_patch_paths]
                                syntax_rc, syntax_output = box.execute(syntax_check_command(included))
                                record["syntax_check"] = {"returncode": syntax_rc, "output": syntax_output}
                                if syntax_rc:
                                    feedback = (
                                        "Submission not accepted: changed Python source has a syntax error. "
                                        "Read the affected source, fix its syntax, verify your reproduction, "
                                        "then submit again. No grading tests were run.\n"
                                        + smith.render_observation(syntax_rc, syntax_output, obs_cap)
                                    )
                            if feedback:
                                submitted = False
                                observation = feedback
                                record["submission_feedback"] = feedback
                    record.update(observation=observation, submitted=submitted)
                    trace.write(json.dumps(record) + "\n")
                    trace.flush()
                    messages.append({"role": "user", "content": observation})
                    if submitted:
                        stop_reason = "submitted"
                        break
            patch, paths, rejection = box.export_patch(allow_new_repro=allow_new_repro)
            (directory / "model.patch").write_text(patch)
            (directory / "sandbox.json").write_text(
                json.dumps(
                    {
                        "baseline": preparation,
                        "changed_paths": paths,
                        "rejection": rejection,
                        "excluded_reproduction_paths": box.excluded_patch_paths,
                    },
                    indent=2,
                )
            )
        finally:
            box.close()
            client.close()
        if rejection:
            report = {"reward": 0.0, "resolved": False, "patch_rejection": rejection}
            (directory / "grade.json").write_text(json.dumps(report))
        else:
            report = grade(row, patch, directory)
        httpx.post(
            event_url,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "event_type": "reward",
                "data": {
                    "value": report["reward"],
                    "reason": stop_reason,
                    "source": "isolated_pytest",
                },
            },
            timeout=30,
        ).raise_for_status()
