# Copyright (c) Microsoft. All rights reserved.
"""Full Python split grading; reuse the checked agent and isolated sandbox."""

import ast
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

import docker
import sandbox
import smith_docker_agent as pilot

TORNADO_RUNNER = """import json, sys, unittest
def node(test):
    module, cls, method = test.id().rsplit('.', 2)
    return module.replace('.', '/') + '.py::' + cls + '::' + method
class RecordingResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        self.stream.writeln('PASSED ' + node(test))
    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.stream.writeln('FAILED ' + node(test))
    def addError(self, test, error):
        super().addError(test, error)
        self.stream.writeln('ERROR ' + node(test))
    def addExpectedFailure(self, test, error):
        super().addExpectedFailure(test, error)
        self.stream.writeln('XFAIL ' + node(test))
unittest.TextTestResult = RecordingResult
sys.argv = ['tornado-tests', '--verbose', *json.loads(sys.argv[1])]
from tornado.test.runtests import main
main()
"""

HTTP_FIXTURE_SHA256 = "633c5b6acb69cf38d6ca4da7ea9b87c09040636c10c100f3be3ace3a6c606216"
HTTP_FIXTURE_SERVER = """from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote
body = Path('/root/agl-http-fixture.html').read_bytes()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if unquote(self.path) != '/wiki/Заглавная_страница':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=UTF-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
HTTPServer(('127.0.0.1', 80), Handler).serve_forever()
"""


def install_http_fixture(box, nodes):
    if "tests/test_pyquery.py::TestWebScrappingEncoding::test_get" not in nodes:
        return None
    directory = Path(
        os.environ.get(
            "AGL_HTTP_FIXTURE_DIR",
            "/media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/http-fixtures-v1",
        )
    )
    body = (directory / "ru-wikipedia-main.html").read_bytes()
    if hashlib.sha256(body).hexdigest() != HTTP_FIXTURE_SHA256:
        raise RuntimeError("The recorded public HTTP response changed")
    box.copy_bytes(body, "agl-http-fixture.html")
    box.root(
        [
            "/opt/miniconda3/envs/testbed/bin/python",
            "-c",
            "from pathlib import Path; p=Path('/etc/hosts'); "
            "p.write_text(p.read_text()+'\\n127.0.0.1 ru.wikipedia.org\\n')",
        ]
    )
    box.container.exec_run(
        ["/opt/miniconda3/envs/testbed/bin/python", "-c", HTTP_FIXTURE_SERVER], user="0:0", detach=True
    )
    box.root(
        [
            "/opt/miniconda3/envs/testbed/bin/python",
            "-c",
            "import socket,time\n"
            "for attempt in range(50):\n"
            " try:\n"
            "  socket.create_connection(('127.0.0.1',80),timeout=.1).close(); break\n"
            " except OSError: time.sleep(.1)\n"
            "else: raise RuntimeError('Recorded HTTP server did not start')",
        ]
    )
    # Record actual provenance. The test and its assertions are unchanged, and
    # the container remains offline; the response is real captured public HTML.
    return json.loads((directory / "manifest.json").read_text())


def parse_statuses(output):
    # Some image configs force ANSI colors even without a terminal. Strip only
    # terminal styling before passing the output to the upstream status parser.
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    return pilot.load_smith().parse_test_statuses(plain)


def package_metadata_dirs(paths):
    allowed = {
        "PKG-INFO",
        "SOURCES.txt",
        "dependency_links.txt",
        "requires.txt",
        "top_level.txt",
        "entry_points.txt",
        "not-zip-safe",
    }
    groups = {}
    for path in paths:
        if path:
            parts = PurePosixPath(path).parts
            groups.setdefault(parts[0], []).append(parts)
    return sorted(
        name
        for name, files in groups.items()
        if re.fullmatch(r"[A-Za-z0-9_.-]+\.egg-info", name)
        and all(len(parts) == 2 and parts[1] in allowed for parts in files)
    )


def fixture_patch_allowed(before, after):
    """Permit local predicate repairs, preserving assertions, calls and hooks.

    Some published bugs are in conftest helper code itself. This generic rule
    does not consult a task's reference patch or reveal its changed paths.
    """
    if max(len(before), len(after)) > 65536:
        return False

    class Predicates(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) in (int, float):
                node.value = type(node.value)(0)
            elif isinstance(node.value, str) and node.value.startswith("/"):
                node.value = "/"
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            dirname = ast.dump(ast.parse("os.path.dirname", mode="eval").body)
            if (
                ast.dump(node.func) == dirname
                and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Call)
                and ast.dump(node.args[0].func) == dirname
            ):
                return node.args[0]
            return node

        def visit_Compare(self, node):
            self.generic_visit(node)
            node.ops = [ast.Eq() for _ in node.ops]
            return node

        def visit_UnaryOp(self, node):
            if isinstance(node.op, ast.Not):
                return self.visit(node.operand)
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            return node  # Preserve nested functions and their signatures.

    class FunctionBodies(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if not node.name.startswith(("pytest_", "test_")):
                node.body = [Predicates().visit(statement) for statement in node.body]
            return node

    try:
        old, new = [FunctionBodies().visit(ast.parse(source)) for source in (before, after)]
    except (SyntaxError, ValueError):
        return False
    return ast.dump(old) == ast.dump(new)


def embedded_tests(source):
    """Identify inline tests and doctest strings independently of line numbers."""
    tests = {}

    def visit(node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = (*scope, node.name)
            if node.name.startswith("test_") or (isinstance(node, ast.ClassDef) and node.name.startswith("Test")):
                tests["test:" + ".".join(scope)] = ast.dump(node)
                return
        # A control-flow mutation can move a return before the original string,
        # so it stops being Python's first-statement docstring. Preserve its
        # content while allowing the repair to move it back to the first line.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            doc = node.value.value
            if isinstance(doc, str) and ">>>" in doc:
                key = "doctest:" + ".".join(scope)
                tests.setdefault(key, []).append(doc)
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(ast.parse(source), ())
    return tests


class FullPythonSandbox(pilot.SmithSandbox):
    def container_options(self, task):
        if task["instance_id"].startswith("modin-project__modin."):
            return {
                "mem_limit": "8g",
                "pids_limit": 1024,
                "shm_size": "1g",
                "environment": {
                    "MODIN_CPUS": "2",
                    "MODIN_GPUS": "0",
                    "MODIN_MEMORY": "536870912",
                    "NUMEXPR_MAX_THREADS": "2",
                    "RAY_num_server_call_thread": "2",
                    "RAY_gcs_server_rpc_server_thread_num": "2",
                    "RAY_gcs_server_rpc_client_thread_num": "2",
                },
            }
        return {}

    def export_patch(self, *, allow_new_repro=False):
        for path in self.restored_source_paths:
            try:
                original = embedded_tests(self.git("show", "HEAD:" + path))
                candidate = embedded_tests(self.root(["cat", "--", path]))
            except (RuntimeError, SyntaxError, ValueError):
                return "", [path], "invalid_embedded_test_source"
            if original != candidate:
                return "", [path], "embedded_test_change"
        return super().export_patch(allow_new_repro=allow_new_repro)

    def allowed_protected_paths(self, paths):
        allowed = set()
        for path in paths:
            candidate = PurePosixPath(path)
            if candidate.name != "conftest.py" or sandbox.forbidden_patch_path(str(candidate.parent / "fixture.py")):
                continue
            try:
                before = self.git("show", "HEAD:" + path)
                after = self.root(["cat", "--", path])
            except RuntimeError:
                continue
            if fixture_patch_allowed(before, after):
                allowed.add(path)
        return allowed

    def prepare(self):
        deleted = self.root(["git", "diff", "--name-only", "--diff-filter=D", "-z"]).split("\0")
        deleted_tests = [
            path
            for path in deleted
            if path.endswith(".py")
            and (
                "tests" in PurePosixPath(path).parts
                or "test" in PurePosixPath(path).parts
                or PurePosixPath(path).name.startswith("test_")
            )
        ]
        if deleted_tests:
            # Some published images already have unrelated test deletions in
            # their worktree. Restore those from the image's current HEAD before
            # checking out the task (which removes that task's own F2P files).
            self.root(["git", "checkout", "HEAD", "--", *deleted_tests])
        # Editable installs can leave untracked package metadata in the image.
        # Preserve it for importlib.metadata; keep it out of exported patches.
        # Other untracked files and all tracked changes still fail the clean check.
        paths = self.root(["git", "ls-files", "--others", "--exclude-standard", "-z"]).split("\0")
        metadata = package_metadata_dirs(paths)
        resources = []
        if self.container.labels["agl.instance_id"].startswith("pwaller__pyfiglet."):
            fonts = [path for path in paths if path.startswith("pyfiglet/fonts/")]
            if fonts and all(
                PurePosixPath(path).suffix in {".flf", ".flc"} or path == "pyfiglet/fonts/__init__.py" for path in fonts
            ):
                self.root(
                    [
                        "/opt/miniconda3/envs/testbed/bin/python",
                        "-c",
                        "import pathlib,json,sys; paths=json.loads(sys.argv[1]); "
                        "assert all(not pathlib.Path(p).is_symlink() for p in paths); "
                        "assert not pathlib.Path('pyfiglet/fonts/__init__.py').read_text().strip()",
                        json.dumps(fonts),
                    ]
                )
                resources = ["pyfiglet/fonts"]
        if metadata or resources:
            self.root(
                [
                    "/opt/miniconda3/envs/testbed/bin/python",
                    "-c",
                    "import json,pathlib,sys; "
                    "paths=json.loads(sys.argv[1]); "
                    "assert all(not pathlib.Path(p).is_symlink() for p in paths); "
                    "f=pathlib.Path('.git/info/exclude'); "
                    "f.write_text(f.read_text()+'\\n'+'\\n'.join('/'+p+'/' for p in paths)+'\\n')",
                    json.dumps(sorted(metadata + resources)),
                ]
            )
        report = super().prepare()
        report["preserved_package_metadata"] = sorted(metadata)
        report["preserved_package_resources"] = resources
        report["restored_prebuilt_test_deletions"] = deleted_tests
        # The published removal commit deletes whole files, including production
        # modules that contain doctests/inline tests. Restore only non-test Python
        # source from the BUGGY parent, never from the pre-bug reference commit.
        removed = self.git("diff", "--name-only", "--diff-filter=D", "-z", "HEAD~1", "HEAD").split("\0")
        self.restored_source_paths = [
            path
            for path in removed
            if path.endswith(".py") and PurePosixPath(path).name != "test.py" and not sandbox.forbidden_patch_path(path)
        ]
        if self.restored_source_paths:
            original_head = report["baseline_head"]
            parent = self.git("rev-parse", "HEAD~1").strip()
            self.git("checkout", "HEAD~1", "--", *self.restored_source_paths)
            tree = self.git("write-tree").strip()
            # This private, disposable commit defines patch export's clean
            # baseline, preserving the original Bug Patch parent and history.
            prepared_head = self.git(
                "-c",
                "user.name=AGL sandbox",
                "-c",
                "user.email=sandbox@localhost",
                "commit-tree",
                tree,
                "-p",
                parent,
                "-m",
                "Remove F2P Tests",
            ).strip()
            self.git("update-ref", "HEAD", prepared_head, original_head)
            self.root(["chown", "65534:65534", "--", *self.restored_source_paths])
            if self.git("status", "--porcelain").strip():
                raise RuntimeError("Inline-source recovery did not produce a clean baseline")
            report["published_baseline_head"] = original_head
            report["baseline_head"] = prepared_head
        report["restored_buggy_source_paths"] = self.restored_source_paths
        if self.container.labels["agl.instance_id"].startswith("gruns__furl."):
            # This old project's tests require the original 3.10 URL parser.
            # Apply the unmodified upstream module inside this disposable image
            # for both agent actions and grading, leaving the host env untouched.
            content = (Path(__file__).parent / "vendor/cpython310/parse.py").read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != "f14ec821fe7ded126d2202eef195f66bd63f74975c9ae88322d0500f13eba8db":
                raise RuntimeError("CPython URL parser compatibility source changed")
            self.copy_bytes(content, "cpython-310-parse.py")
            self.root(
                ["cp", "/root/cpython-310-parse.py", "/opt/miniconda3/envs/testbed/lib/python3.10/urllib/parse.py"]
            )
            report["urllib_parse_compatibility"] = {"version": "CPython v3.10.0", "sha256": digest}
        if self.container.labels["agl.instance_id"].startswith("pydicom__pydicom."):
            name = "MR-SIEMENS-DICOM-WithOverlays.dcm"
            directory = Path("/media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/pydicom-fixtures-v1")
            content = (directory / name).read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != "094faf56c63bff84c30567e29de0c67d7c5a8ae05cf880ac12175491b6b645d2":
                raise RuntimeError("Pydicom's official test-data checksum differs")
            self.copy_bytes(content, name)
            cache = "/tmp/agl-home/.pydicom/data"
            self.root(["mkdir", "-p", cache])
            self.root(["cp", "/root/" + name, cache + "/" + name])
            self.root(["chmod", "444", cache + "/" + name])
            report["cached_test_data"] = {name: digest}
        if self.container.labels["agl.instance_id"].startswith("sunpy__sunpy."):
            content = Path(
                "/media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/iers-fixtures-v1/Leap_Second.dat"
            ).read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != "6cb6f5d4b819f2e568e25db4b0b26d89dedf031fdffb18bc94d40f4e94e268d7":
                raise RuntimeError("The pinned IERS leap-second table changed")
            self.copy_bytes(content, "Leap_Second.dat")
            self.root(
                [
                    "cp",
                    "/root/Leap_Second.dat",
                    "/opt/miniconda3/envs/testbed/lib/python3.11/site-packages/astropy_iers_data/data/Leap_Second.dat",
                ]
            )
            report["iers_leap_seconds_sha256"] = digest
        return report


def grade(row, patch, output_dir, *, reference=False):
    if row.get("grading_protocol") != "f2p_file":
        raise ValueError("Full Python agent requires the prepared f2p_file dataset")
    f2p, p2p = pilot.test_nodes(row, max_tests=None)
    f2p_paths = {node.split("::", 1)[0] for node in f2p}
    if any(node.split("::", 1)[0] not in f2p_paths for node in p2p):
        raise ValueError("P2P includes files outside the declared grading protocol")
    client = docker.from_env(timeout=370)
    box = FullPythonSandbox(client, pilot.agent_task(row), output_dir.name + "-grade")
    try:
        preparation = box.prepare()
        commits = box.git("log", "-3", "--format=%s").splitlines()
        if commits[:2] != ["Remove F2P Tests", "Bug Patch"]:
            raise RuntimeError(f"Unexpected SWE-smith branch layout: {commits}")
        if reference:
            patch = box.git("diff", "HEAD~1", "HEAD~2", "--binary")
        if patch:
            box.copy_bytes(patch.encode())
            box.git("apply", "--check", "/root/candidate.patch")
            box.git("apply", "/root/candidate.patch")
        # Inline tests already coexist with restored buggy source. Restoring the
        # whole module here would silently overwrite the model's actual repair.
        test_only_paths = sorted(f2p_paths - set(box.restored_source_paths))
        if test_only_paths:
            box.git("checkout", "HEAD~1", "--", *test_only_paths)
        # Preserve trusted project settings and plugins (asyncio, Django, etc.).
        # Like the official evaluator, override only xdist parallelism; clearing
        # addopts would also erase project-specific options such as --ds.
        probe = box.container.exec_run(
            [
                "/opt/miniconda3/envs/testbed/bin/python",
                "-c",
                "import importlib.util,json; print(json.dumps({name: importlib.util.find_spec(name) is not None "
                "for name in ['xdist','pytest_cov']}))",
            ],
            user="65534:65534",
            workdir="/testbed",
        )
        if probe.exit_code:
            raise RuntimeError("Could not inspect the image's pytest plugins")
        plugins = json.loads(probe.output)
        xdist = ["-n", "2"] if plugins["xdist"] else ["-p", "no:xdist"]
        if row["instance_id"].startswith("modin-project__modin."):
            # The selected tests start Ray themselves. Run one test process in
            # the bounded container instead of competing Ray clusters per worker.
            xdist = ["-n", "0"] if plugins["xdist"] else xdist
        # Whole-project coverage thresholds do not apply to selected test files.
        # Keep every F2P/P2P test and disable only coverage instrumentation.
        coverage = ["--no-cov"] if plugins["pytest_cov"] else []
        nodes = list(dict.fromkeys(f2p + p2p))
        if row["instance_id"].startswith("jd__tenacity."):
            # Keep each test module together in file order. Appending P2P after
            # F2P revisits asyncio tests after Tornado's AsyncTestCase has cleared
            # the event loop. Preserve the exact test set and every assertion.
            nodes.sort(key=lambda node: node.split("::", 1)[0])
        http_fixture = install_http_fixture(box, nodes)
        native_tornado = row["instance_id"].startswith("tornadoweb__tornado.")
        if native_tornado:
            # Reuse the project's runner, including warning and logging checks.
            # The wrapper only records exact IDs; it changes no test outcomes.
            native_ids = []
            for node in nodes:
                path, cls, method = node.split("::")
                native_ids.append(path.removesuffix(".py").replace("/", ".") + "." + cls + "." + method)
            invocation = ["-c", TORNADO_RUNNER, json.dumps(native_ids)]
        else:
            # Plugins such as pytest-snail require config.cache. Each grading
            # container gets a fresh cache outside the patchable source tree.
            # The image's old snail timing plugin also calls a removed pluggy
            # unregister API at shutdown, making an all-passing suite exit 1.
            invocation = [
                "-m",
                "pytest",
                "-rA",
                "--color=no",
                "-o",
                "cache_dir=/tmp/agl-pytest-cache",
                "-p",
                "no:snail",
                *xdist,
                *coverage,
                *nodes,
            ]
        result = box.container.exec_run(
            [
                "/usr/bin/timeout",
                "--kill-after=5",
                "300",
                "/opt/miniconda3/envs/testbed/bin/python",
                *invocation,
            ],
            user="65534:65534",
            workdir="/testbed",
            environment={
                "HOME": "/tmp/agl-home",
                "PYTHONPATH": "/testbed",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "",
                "PATH": "/opt/miniconda3/envs/testbed/bin:/usr/bin:/bin",
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
            },
        )
        output = result.output.decode(errors="replace")
        (output_dir / "test-output.txt").write_text(output)
        statuses = parse_statuses(output)
        pass_f = sum(statuses.get(node) in ("PASSED", "XFAIL") for node in f2p)
        pass_p = sum(statuses.get(node) in ("PASSED", "XFAIL") for node in p2p)
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
            "grading_protocol": "f2p_file",
            "test_runner": "native_tornado" if native_tornado else "pytest",
            "coverage_instrumentation_disabled": plugins["pytest_cov"],
            "recorded_http_fixture": http_fixture,
        }
        (output_dir / "grade.json").write_text(json.dumps(report))
        return report
    finally:
        box.close()
        client.close()


class FullPythonAgent(pilot.SmithDockerAgent):
    max_grading_tests = None
    extra_workflow_hint = """

Existing helper functions in conftest.py may contain task bugs. For these files,
only numeric constants, comparison operators, and boolean `not` inside existing
helper bodies can be repaired, plus slash-prefixed path strings and repeated
os.path.dirname path traversal. Keep every assertion, other call, signature, decorator,
and pytest hook unchanged. Adding skips or changing test cases is prohibited.
"""

    def action_rejection(self, smith, action):
        # Preserve every upstream action check. The final AST check independently
        # restricts what conftest edits can enter the fresh grading container.
        return smith._forbidden_action(action.replace("conftest.py", "fixture_helpers.py"))

    def run(self):
        row = json.loads(os.environ["AGL_TASK"])
        if row.get("grading_protocol") != "f2p_file":
            raise ValueError("Use the full Python preparation output")
        # Each local rollout has its own process. Preserve the shared checked
        # agent implementation; replace only its grader for this process.
        original = pilot.grade
        original_sandbox = pilot.SmithSandbox
        pilot.grade = grade
        pilot.SmithSandbox = FullPythonSandbox
        try:
            return super().run()
        finally:
            pilot.grade = original
            pilot.SmithSandbox = original_sandbox
