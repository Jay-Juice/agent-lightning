"""Generic semantic cumulative file diffs for privileged Critic input."""

from __future__ import annotations

import difflib
from typing import Any


def _lines(text: str) -> list[str]:
    return text.splitlines()


def _hunks(before: str, after: str, path: str) -> list[dict[str, Any]]:
    raw = list(difflib.unified_diff(_lines(before), _lines(after), fromfile=path, tofile=path, n=3, lineterm=""))
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in raw:
        if line.startswith("@@"):
            current = {"header": line, "lines": [], "changed_lines": 0}
            out.append(current)
        elif current is not None:
            current["lines"].append(line)
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                current["changed_lines"] += 1
    return out


def build_semantic_delta(raw_delta: dict[str, Any], baseline_store) -> dict[str, Any]:
    files = []
    for item in raw_delta.get("filesystem", []):
        change = item["change"]
        record = item.get("after", item)
        path = record["path"]
        before = baseline_store.read_text(path)
        after_kind = record.get("type")
        after_text = record.get("content") if after_kind == "file" else None
        before_text = before.get("text") if before.get("kind") == "text" else None
        can_diff = change == "removed" and before_text is not None
        can_diff = can_diff or (isinstance(after_text, str) and (before_text is not None or change == "added"))
        kind = "text" if can_diff else "metadata"
        row = {
            "path": path,
            "change": change,
            "kind": kind,
            "before_sha": before.get("blob_sha"),
            "after_sha": record.get("sha256"),
            "size": record.get("size"),
            "hunks": [],
        }
        if kind == "text":
            row["hunks"] = _hunks(before_text or "", after_text or "", path)
        files.append(row)
    return {"files": sorted(files, key=lambda row: row["path"]), "processes": raw_delta.get("processes", {}), "sockets": raw_delta.get("sockets", {})}
