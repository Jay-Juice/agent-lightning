"""Frozen prepared-baseline blob access for causal privileged-state diffs."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any

_SHA40 = re.compile(r"[0-9a-f]{40}\Z")


def _safe_path(path: str) -> str:
    if not isinstance(path, str) or "\x00" in path or path.startswith("/"):
        raise ValueError(f"invalid baseline path: {path!r}")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"invalid baseline path: {path!r}")
    pure = PurePosixPath(path)
    return str(pure)


class PreparedBaselineBlobStore:
    """Freeze the exact prepared checkout's path->blob map once per rollout."""

    def __init__(
        self,
        container: Any,
        *,
        exact_commit: str,
        git_dir: str = "/root/agl-private-git",
        max_text_bytes: int = 2 << 20,
    ):
        if not isinstance(exact_commit, str) or not _SHA40.fullmatch(exact_commit):
            raise ValueError("exact_commit must be a full 40-character hexadecimal SHA")
        self.container = container
        self.exact_commit = exact_commit
        self.git_dir = git_dir
        self.max_text_bytes = max_text_bytes
        self.entries: dict[str, dict[str, Any]] = {}
        self.manifest_hash: str | None = None

    def _run(self, args: list[str]):
        result = self.container.exec_run(args, user="0:0", workdir="/")
        if result.exit_code:
            raise RuntimeError(result.output.decode("utf-8", errors="replace")[-2000:])
        return result.output

    def initialize(self) -> dict[str, Any]:
        out = self._run(["git", "--git-dir", self.git_dir, "ls-tree", "-r", "-l", "-z", self.exact_commit, "--"])
        entries: dict[str, dict[str, Any]] = {}
        for line in out.decode("utf-8", errors="strict").split("\0"):
            if not line:
                continue
            meta, path = line.split("\t", 1)
            mode, kind, sha, size = meta.split()
            path = _safe_path(path)
            if kind != "blob":
                continue
            entries[path] = {"mode": mode, "type": kind, "blob_sha": sha, "size": int(size)}
        self.entries = entries
        manifest = {"commit": self.exact_commit, "entries": entries}
        self.manifest_hash = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {"commit": self.exact_commit, "count": len(entries), "manifest_hash": self.manifest_hash}

    def write_manifest(self, output_path) -> None:
        if self.manifest_hash is None:
            raise RuntimeError("initialize baseline store first")
        payload = {"commit": self.exact_commit, "manifest_hash": self.manifest_hash, "entries": self.entries}
        with gzip.open(output_path, "wb") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())

    def read_text(self, path: str) -> dict[str, Any]:
        path = _safe_path(path)
        entry = self.entries.get(path)
        if entry is None:
            return {"kind": "missing", "size": 0, "blob_sha": None}
        if entry["size"] > self.max_text_bytes:
            return {"kind": "too_large", "size": entry["size"], "blob_sha": entry["blob_sha"]}
        raw = self._run(["git", "--git-dir", self.git_dir, "cat-file", "blob", entry["blob_sha"]])
        if b"\x00" in raw:
            return {"kind": "binary", "size": len(raw), "blob_sha": entry["blob_sha"]}
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {"kind": "non_utf8", "size": len(raw), "blob_sha": entry["blob_sha"]}
        return {"kind": "text", "text": text, "size": len(raw), "blob_sha": entry["blob_sha"]}
