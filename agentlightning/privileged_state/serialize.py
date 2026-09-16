"""Deterministic, token-budgeted serialization of sandbox-state deltas."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

MARKER = "[PRIVILEGED SANDBOX STATE — CRITIC ONLY]"


def _file_item(item: dict[str, Any]) -> tuple[str, str | None]:
    change = item["change"]
    record = item.get("after", item)
    prefix = {"added": "A", "modified": "M", "removed": "D"}[change]
    manifest = f"{prefix} {record['path']} type={record.get('type')} size={record.get('size', '-')}"
    content = record.get("content")
    if content is None:
        reason = record.get("content_omitted")
        return manifest, f"[CURRENT FILE OMITTED: {record['path']}; reason={reason}]" if reason else None
    return manifest, f"[CURRENT FILE: {record['path']}]\n{content}"


def budgeted_state_text(
    delta: dict[str, Any],
    *,
    token_length: Callable[[str], int],
    max_tokens: int,
    fits: Callable[[str], bool] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Serialize generic records in a fixed priority order.

    ``fits`` may enforce the complete Critic prompt limit, including chat-template
    overhead. It must depend only on pre-action data.
    """
    if max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative")
    files = delta.get("filesystem", [])
    proc = delta.get("processes", {})
    sockets = delta.get("sockets", {})
    manifests, contents = [], []
    for item in files:
        manifest, content = _file_item(item)
        manifests.append(manifest)
        if content:
            contents.append(content)
    summary = (
        f"{MARKER}\n"
        f"files_changed={len(files)} processes_added={len(proc.get('added', []))} "
        f"processes_removed={len(proc.get('removed', []))} sockets_added={len(sockets.get('added', []))} "
        f"sockets_removed={len(sockets.get('removed', []))}"
    )
    runtime = []
    for sign, name, group in (
        ("+", "process", proc.get("added", [])),
        ("-", "process", proc.get("removed", [])),
        ("+", "socket", sockets.get("added", [])),
        ("-", "socket", sockets.get("removed", [])),
    ):
        runtime.extend(f"{sign} {name} {row}" for row in group)
    selected_manifests: list[str] = []
    selected_runtime: list[str] = []
    selected_contents: list[str] = []
    omitted = 0

    def render() -> str:
        sections = [summary]
        if selected_manifests:
            sections.append("[FILE MANIFEST]\n" + "\n".join(selected_manifests))
        if selected_runtime:
            sections.append("[RUNTIME]\n" + "\n".join(selected_runtime))
        sections.extend(selected_contents)
        return "\n\n".join(sections)

    def valid(text: str) -> bool:
        return token_length(text) <= max_tokens and (fits(text) if fits else True)

    if max_tokens == 0 or not valid(summary):
        raw_records = len(manifests) + len(runtime) + len(contents)
        return "", {"serialized_tokens": 0, "truncated": bool(raw_records), "omitted_records": raw_records}
    # Preserve the highest-value structure under pressure: as many manifest
    # entries as fit, followed by runtime changes, followed by file bodies.
    for records, selected in (
        (manifests, selected_manifests),
        (runtime, selected_runtime),
        (contents, selected_contents),
    ):
        for record in records:
            selected.append(record)
            if not valid(render()):
                selected.pop()
                omitted += 1
    text = render()
    if omitted:
        footer = f"\n\n[OMITTED]\nrecords={omitted}"
        if valid(text + footer):
            text += footer
    return text, {
        "serialized_tokens": token_length(text),
        "truncated": omitted > 0,
        "omitted_records": omitted,
        "raw_records": len(manifests) + len(runtime) + len(contents),
    }


def budgeted_semantic_state_text(delta, *, token_length, max_tokens, fits=None):
    """Pack cumulative diff fragments fairly; coverage counts complete changed lines.

    Long diff lines are explicitly fragmented. No task-specific ranking is used.
    Every omitted fragment, manifest or runtime record is counted as truncation.
    """
    if max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative")
    files = sorted(delta.get("files", []), key=lambda row: row["path"])
    manifests, runtime, queues, hunks, changed = [], [], {}, set(), {}
    for row in files:
        path = row["path"]
        prefix = {"added": "A", "removed": "D", "modified": "M"}[row["change"]]
        before_sha = str(row.get("before_sha") or "-")[:12]
        after_sha = str(row.get("after_sha") or "-")[:12]
        manifests.append(f"{prefix} {path} size={row.get('size')} sha={before_sha}->{after_sha}")
        queue = queues.setdefault(path, [])
        for hi, hunk in enumerate(row.get("hunks", []), 1):
            hk = (path, hi)
            hunks.add(hk)
            pending, pending_keys = [], []

            def flush(pending=pending, pending_keys=pending_keys, queue=queue, path=path, hi=hi, hunk=hunk, hk=hk):
                if pending:
                    queue.append(
                        {
                            "text": f"[PATCH {path} hunk={hi}]\n{hunk['header']}\n" + "\n".join(pending),
                            "hunk": hk,
                            "lines": list(pending_keys),
                        }
                    )
                    pending.clear()
                    pending_keys.clear()

            for li, line in enumerate(hunk["lines"]):
                pieces = [line[i : i + 96] for i in range(0, len(line), 96)] or [""]
                lk = (path, hi, li)
                if line.startswith(("+", "-")):
                    changed[lk] = len(pieces)
                if len(pieces) > 1:
                    flush()
                    for part, piece in enumerate(pieces, 1):
                        header = f"[PATCH {path} hunk={hi} line={li + 1} fragment={part}/{len(pieces)}]"
                        queue.append({"text": f"{header}\n{hunk['header']}\n{piece}", "hunk": hk, "lines": [lk]})
                else:
                    if sum(len(value) + 1 for value in pending) + len(line) > 96:
                        flush()
                    pending.append(line)
                    pending_keys.append(lk)
            flush()
    for namespace in ("processes", "sockets"):
        for sign, change in (("+", "added"), ("-", "removed")):
            for item in delta.get(namespace, {}).get(change, []):
                runtime.append(f"{sign} {namespace} {item}")
    total_chunks = sum(map(len, queues.values()))
    summary = f"{MARKER}\nfiles_changed={len(files)} hunks={len(hunks)} changed_lines={len(changed)}"
    # Reserve an explicit, small omission notice before filling the budget.
    notice = "[OMITTED] Some records or diff fragments are not shown."
    chosen_manifest, chosen_runtime, selected = [], [], []

    def render(with_notice=True):
        sections = [summary]
        if chosen_manifest:
            sections.append("[FILE MANIFEST]\n" + "\n".join(chosen_manifest))
        if chosen_runtime:
            sections.append("[RUNTIME]\n" + "\n".join(chosen_runtime))
        sections.extend(item["text"] for item in selected)
        if with_notice:
            sections.append(notice)
        return "\n\n".join(sections)

    def valid(text):
        return token_length(text) <= max_tokens and (fits(text) if fits else True)

    enabled = max_tokens > 0 and valid(render())
    if enabled:
        for rows, kept in ((manifests, chosen_manifest), (runtime, chosen_runtime)):
            for row in rows:
                kept.append(row)
                if not valid(render()):
                    kept.pop()
        # One fragment per file per round; long files cannot consume later rounds
        # before the other files have had an opportunity to contribute.
        for offset in range(max(map(len, queues.values()), default=0)):
            for queue in queues.values():
                if offset < len(queue):
                    selected.append(queue[offset])
                    if not valid(render()):
                        selected.pop()
    included_parts = {}
    for item in selected:
        for key in item["lines"]:
            included_parts[key] = included_parts.get(key, 0) + 1
    included_lines = sum(included_parts.get(key, 0) == parts for key, parts in changed.items())
    included_hunks = {item["hunk"] for item in selected}
    total_per_hunk, selected_per_hunk = {}, {}
    for queue in queues.values():
        for item in queue:
            key = item["hunk"]
            total_per_hunk[key] = total_per_hunk.get(key, 0) + 1
    for item in selected:
        key = item["hunk"]
        selected_per_hunk[key] = selected_per_hunk.get(key, 0) + 1
    full_hunks = sum(selected_per_hunk.get(key, 0) == count for key, count in total_per_hunk.items())
    omitted_records = len(manifests) - len(chosen_manifest) + len(runtime) - len(chosen_runtime)
    truncated = len(selected) < total_chunks or omitted_records > 0
    text = render(truncated) if enabled else ""
    if not enabled:
        truncated = bool(files or runtime or total_chunks)
    stats = {
        "serialized_tokens": token_length(text),
        "truncated": truncated,
        "changed_files_total": len(files),
        "changed_text_files_total": sum(row.get("kind") == "text" for row in files),
        "files_with_semantic_content": len({key[0] for key in changed if included_parts.get(key, 0) == changed[key]}),
        "hunks_total": len(hunks),
        "hunks_included": len(included_hunks),
        "hunks_fully_included": full_hunks,
        "hunk_chunks_total": total_chunks,
        "hunk_chunks_included": len(selected),
        "changed_lines_total": len(changed),
        "changed_lines_included": included_lines,
        "omitted_hunks": len(hunks) - full_hunks,
        "omitted_changed_lines": len(changed) - included_lines,
        "manifest_records_total": len(manifests),
        "manifest_records_included": len(chosen_manifest),
        "runtime_records_total": len(runtime),
        "runtime_records_included": len(chosen_runtime),
        "omitted_records": omitted_records,
    }
    return text, stats
