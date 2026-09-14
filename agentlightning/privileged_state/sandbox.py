"""Generic, bounded snapshots of one rollout's Docker sandbox.

The collector deliberately scopes filesystem reads to ``/testbed`` and never
follows symlinks. Docker's layer diff supplies the cumulative dirty-path set;
records are compared with a post-prepare baseline so a restored file disappears
from the resulting delta.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

_HELPER = r'''
import hashlib, json, os, stat, sys

ROOT = "/testbed"
CONTENT_LIMIT = int(sys.argv[1])
targets = json.loads(sys.argv[2])

def inside(path):
    path = posix_norm(path)
    return path == ROOT or path.startswith(ROOT + "/")

def posix_norm(path):
    return os.path.normpath(path).replace("//", "/")

def record(path, hash_files, with_content):
    try:
        st = os.lstat(path)
    except (FileNotFoundError, PermissionError):
        return None
    rel = os.path.relpath(path, ROOT)
    if rel == ".":
        return None
    mode = stat.S_IMODE(st.st_mode)
    base = {"path": rel, "mode": mode, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}
    if stat.S_ISREG(st.st_mode):
        base.update(type="file", size=st.st_size)
        if not hash_files:
            return base
        digest = hashlib.sha256()
        chunks = []
        total = 0
        readable = True
        try:
            # O_NOFOLLOW closes the symlink-swap hole between lstat and open.
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            with os.fdopen(fd, "rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    total += len(chunk)
                    if with_content and total <= CONTENT_LIMIT:
                        chunks.append(chunk)
        except (OSError, PermissionError):
            readable = False
        base["sha256"] = digest.hexdigest() if readable else None
        if with_content and readable:
            if total > CONTENT_LIMIT:
                base["content_omitted"] = "size_limit"
            else:
                raw = b"".join(chunks)
                if b"\0" in raw:
                    base["content_omitted"] = "binary"
                else:
                    try:
                        base["content"] = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        base["content_omitted"] = "non_utf8"
        return base
    if stat.S_ISDIR(st.st_mode):
        return {**base, "type": "directory"}
    if stat.S_ISLNK(st.st_mode):
        try:
            target = os.readlink(path)
        except OSError:
            target = None
        return {**base, "type": "symlink", "target": target}
    return {**base, "type": "other", "size": st.st_size}

def walk_target(path, hash_files, with_content):
    path = posix_norm(path)
    if not inside(path):
        return []
    if path == ROOT:
        result = []
    else:
        item = record(path, hash_files, with_content)
        if item is None:
            return []
        result = [item]
        if item["type"] != "directory":
            return result
    try:
        entries = sorted(os.scandir(path), key=lambda x: x.name)
    except (FileNotFoundError, PermissionError):
        return result
    for entry in entries:
        # Recursion uses lstat records and never follows directory symlinks.
        result.extend(walk_target(entry.path, hash_files, with_content))
    return result

def process_records():
    def proc_stat(pid):
        raw = open(f"/proc/{pid}/stat").read()
        end = raw.rfind(")")
        if end < 0:
            raise ValueError("Malformed proc stat")
        tail = raw[end + 2:].split()
        return tail[0], int(tail[1])

    own = set()
    pid = os.getpid()
    while pid > 0 and pid not in own:
        own.add(pid)
        try:
            _, pid = proc_stat(pid)
        except Exception:
            break
    rows = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) in own:
            continue
        pid = int(name)
        try:
            state, ppid = proc_stat(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
            continue
        try:
            raw = open(f"/proc/{pid}/cmdline", "rb").read()
            cmd = " ".join(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            cmd = None
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            cwd = None
        rows.append({"pid": pid, "ppid": ppid, "state": state, "command": cmd, "cwd": cwd})
    return sorted(rows, key=lambda row: row["pid"])

def socket_records(processes):
    owners = {}
    for proc in processes:
        try:
            for fd in os.listdir(f"/proc/{proc['pid']}/fd"):
                try:
                    target = os.readlink(f"/proc/{proc['pid']}/fd/{fd}")
                except OSError:
                    continue
                if target.startswith("socket:[") and target.endswith("]"):
                    owners.setdefault(target[8:-1], proc["pid"])
        except (FileNotFoundError, PermissionError):
            continue

    def addr(value, ipv6=False):
        host, port = value.split(":")
        raw = bytes.fromhex(host)
        if ipv6:
            groups = [raw[i:i+4][::-1] for i in range(0, 16, 4)]
            import socket
            host_text = socket.inet_ntop(socket.AF_INET6, b"".join(groups))
        else:
            host_text = ".".join(str(x) for x in raw[::-1])
        return f"{host_text}:{int(port, 16)}"

    rows = []
    for proto in ("tcp", "tcp6", "udp", "udp6"):
        path = f"/proc/net/{proto}"
        try:
            lines = open(path).read().splitlines()[1:]
        except (FileNotFoundError, PermissionError):
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10:
                continue
            state = fields[3]
            # TCP LISTEN and all bound UDP endpoints are useful runtime state.
            if proto.startswith("tcp") and state != "0A":
                continue
            inode = fields[9]
            rows.append({
                "protocol": proto,
                "local": addr(fields[1], proto.endswith("6")),
                "state": state,
                "owner_pid": owners.get(inode),
            })
    return sorted(rows, key=lambda row: (row["protocol"], row["local"], row["owner_pid"] or -1))

hash_files = sys.argv[3] == "1"
with_content = hash_files and targets != [ROOT]
records = {}
for target in targets:
    for item in walk_target(target, hash_files, with_content):
        records[item["path"]] = item
processes = process_records()
print(json.dumps({"filesystem": [records[k] for k in sorted(records)], "processes": processes,
                  "sockets": socket_records(processes)}, sort_keys=True, separators=(",", ":")))
'''


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _semantic_file(record: dict[str, Any]) -> dict[str, Any]:
    return {
        k: record.get(k)
        for k in ("path", "type", "mode", "size", "mtime_ns", "inode", "sha256", "target")
        if k in record
    }


def _content_semantics(record: dict[str, Any]) -> dict[str, Any]:
    """Compare durable state while allowing a restored file's stat clock to differ."""
    return {k: record.get(k) for k in ("path", "type", "mode", "size", "sha256", "target") if k in record}


def _stat_semantics(record: dict[str, Any]) -> dict[str, Any]:
    return {
        k: record.get(k)
        for k in ("path", "type", "mode", "size", "mtime_ns", "inode", "target")
        if k in record
    }


def _runtime_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(k) for k in ("pid", "ppid", "state", "command", "cwd"))


def _socket_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(k) for k in ("protocol", "local", "state", "owner_pid"))


class SandboxStateSnapshotter:
    """Capture cumulative state deltas for a single prepared container."""

    def __init__(self, container: Any, output_dir: Path, *, root: str = "/testbed", content_limit: int = 2 << 20):
        if root != "/testbed":
            raise ValueError("The initial SWE PI implementation is scoped to /testbed")
        if content_limit <= 0:
            raise ValueError("content_limit must be positive")
        self.container = container
        self.output_dir = output_dir
        self.root = root
        self.content_limit = content_limit
        self._baseline: dict[str, Any] | None = None
        (output_dir / "blobs").mkdir(parents=True, exist_ok=True)

    def _probe(self, targets: list[str], *, hash_files: bool) -> dict[str, Any]:
        command = [
            "/opt/miniconda3/envs/testbed/bin/python",
            "-c",
            _HELPER,
            str(self.content_limit),
            json.dumps(targets, separators=(",", ":")),
            "1" if hash_files else "0",
        ]
        result = self.container.exec_run(command, user="0:0", workdir=self.root)
        if result.exit_code:
            output = result.output.decode("utf-8", errors="replace")
            raise RuntimeError(f"Sandbox snapshot failed: {output[-2000:]}")
        return json.loads(result.output.decode("utf-8"))

    def initialize(self) -> dict[str, Any]:
        started = time.monotonic()
        state = self._probe([self.root], hash_files=True)
        state["schema_version"] = 1
        state["snapshot_duration_s"] = time.monotonic() - started
        self._baseline = state
        with gzip.open(self.output_dir / "initial.json.gz", "wb", compresslevel=6) as handle:
            handle.write(_canonical(state))
        return state

    def capture(self) -> dict[str, Any]:
        if self._baseline is None:
            raise RuntimeError("Snapshotter must be initialized after sandbox preparation")
        started_wall = time.time()
        started = time.monotonic()
        baseline_files = {item["path"]: item for item in self._baseline["filesystem"]}
        metadata = self._probe([self.root], hash_files=False)
        metadata_files = {item["path"]: item for item in metadata["filesystem"]}
        candidates = {
            path
            for path in baseline_files.keys() | metadata_files.keys()
            if path not in baseline_files
            or path not in metadata_files
            or _stat_semantics(baseline_files[path]) != _stat_semantics(metadata_files[path])
        }
        existing_targets = [self.root + "/" + path for path in sorted(candidates) if path in metadata_files]
        details = self._probe(
            existing_targets or [self.root + "/.__agl_no_such_path__"], hash_files=True
        )
        current_files = {item["path"]: item for item in details["filesystem"]}
        files = []
        for path in sorted(candidates):
            before = baseline_files.get(path)
            after = current_files.get(path)
            if before is not None and after is not None and _content_semantics(before) == _content_semantics(after):
                continue
            if before is None and after is None:
                continue
            if before is None:
                files.append({"change": "added", **after})
            elif after is None:
                files.append({"change": "removed", **_semantic_file(before)})
            else:
                files.append({"change": "modified", "before": _semantic_file(before), "after": after})

        base_processes = {_runtime_key(row): row for row in self._baseline["processes"]}
        now_processes = {_runtime_key(row): row for row in metadata["processes"]}
        base_sockets = {_socket_key(row): row for row in self._baseline["sockets"]}
        now_sockets = {_socket_key(row): row for row in metadata["sockets"]}
        delta = {
            "schema_version": 1,
            "filesystem": files,
            "processes": {
                "added": [
                    now_processes[k]
                    for k in sorted(now_processes.keys() - base_processes.keys(), key=lambda row: tuple(map(str, row)))
                ],
                "removed": [
                    base_processes[k]
                    for k in sorted(base_processes.keys() - now_processes.keys(), key=lambda row: tuple(map(str, row)))
                ],
            },
            "sockets": {
                "added": [
                    now_sockets[k]
                    for k in sorted(now_sockets.keys() - base_sockets.keys(), key=lambda row: tuple(map(str, row)))
                ],
                "removed": [
                    base_sockets[k]
                    for k in sorted(base_sockets.keys() - now_sockets.keys(), key=lambda row: tuple(map(str, row)))
                ],
            },
        }
        state_hash = hashlib.sha256(_canonical(delta)).hexdigest()
        blob = self.output_dir / "blobs" / f"{state_hash}.json.gz"
        if not blob.exists():
            with gzip.open(blob, "wb", compresslevel=6) as handle:
                handle.write(_canonical(delta))
        return {
            "delta": delta,
            "state_hash": state_hash,
            "state_ref": str(blob),
            "captured_at_start": started_wall,
            "captured_at_end": time.time(),
            "snapshot_duration_s": time.monotonic() - started,
        }
