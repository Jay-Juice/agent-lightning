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


def budgeted_semantic_state_text(
    delta: dict[str, Any], *, token_length: Callable[[str], int], max_tokens: int, fits: Callable[[str], bool] | None = None
) -> tuple[str, dict[str, Any]]:
    """Serialize semantic patch hunks with deterministic round-robin coverage."""
    if max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative")
    files = list(delta.get("files", []))
    runtime = []
    for sign, name, group in (("+", "process", delta.get("processes", {}).get("added", [])), ("-", "process", delta.get("processes", {}).get("removed", [])), ("+", "socket", delta.get("sockets", {}).get("added", [])), ("-", "socket", delta.get("sockets", {}).get("removed", []))):
        runtime.extend(f"{sign} {name} {row}" for row in group)
    manifest = []
    hunk_items = []
    for row in files:
        digest = f" before={str(row.get('before_sha') or '-')[:12]} after={str(row.get('after_sha') or '-')[:12]}"
        manifest.append(f"{'A' if row['change']=='added' else 'D' if row['change']=='removed' else 'M'} {row['path']} size={row.get('size','-')}{digest}")
        for hi, hunk in enumerate(row.get("hunks", []), 1):
            lines = [hunk["header"], *hunk["lines"]]
            hunk_items.append({"path": row["path"], "hunk": hi, "lines": lines, "changed_lines": hunk.get("changed_lines", 0)})
    summary = f"{MARKER}\nfiles_changed={len(files)} text_files={sum(row.get('kind')=='text' for row in files)} hunks={len(hunk_items)}"
    selected_manifest = list(manifest)
    selected_runtime: list[str] = []
    selected_chunks: list[dict[str, Any]] = []
    omitted_hunks = 0
    omitted_changed_lines = 0

    def render() -> str:
        parts = [summary]
        if selected_manifest:
            parts.append("[FILE MANIFEST]\n" + "\n".join(selected_manifest))
        if selected_runtime:
            parts.append("[RUNTIME]\n" + "\n".join(selected_runtime))
        for chunk in selected_chunks:
            parts.append(chunk["text"])
        return "\n\n".join(parts)

    def valid(text: str) -> bool:
        return token_length(text) <= max_tokens and (fits(text) if fits else True)

    if not valid(summary):
        return "", {"serialized_tokens": 0, "truncated": True, "changed_files_total": len(files), "changed_text_files_total": sum(row.get('kind') == 'text' for row in files), "files_with_semantic_content": 0, "hunks_total": len(hunk_items), "hunks_included": 0, "hunk_chunks_total": len(hunk_items), "hunk_chunks_included": 0, "changed_lines_total": sum(x['changed_lines'] for x in hunk_items), "changed_lines_included": 0, "omitted_hunks": len(hunk_items), "omitted_changed_lines": sum(x['changed_lines'] for x in hunk_items)}
    # Manifest is useful even when no hunk fits; then allocate semantic chunks round-robin.
    if not valid(render()):
        selected_manifest = []
    for item in runtime:
        selected_runtime.append(item)
        if not valid(render()):
            selected_runtime.pop()
    by_file: dict[str, list[dict[str, Any]]] = {}
    for item in hunk_items:
        by_file.setdefault(item["path"], []).append(item)
    paths = sorted(by_file)
    max_parts = max((len(by_file[path]) for path in paths), default=0)
    for part in range(max_parts):
        for path in paths:
            items = by_file[path]
            if part >= len(items):
                continue
            item = items[part]
            lines = item["lines"]
            text = "\n".join([f"[PATCH {path} hunk={item['hunk']} part=1/1", *lines])
            selected_chunks.append({"text": text, "changed_lines": item["changed_lines"], "path": path, "hunk": item["hunk"]})
            if not valid(render()):
                selected_chunks.pop()
                omitted_hunks += 1
                omitted_changed_lines += item["changed_lines"]
    included_hunks = len({(x['path'], x['hunk']) for x in selected_chunks})
    included_lines = sum(x['changed_lines'] for x in selected_chunks)
    text = render()
    omitted = len(hunk_items) - included_hunks
    footer = f"\n\n[OMITTED] hunks={omitted} changed_lines={sum(x['changed_lines'] for x in hunk_items)-included_lines}"
    if omitted and valid(text + footer):
        text += footer
    stats = {
        "serialized_tokens": token_length(text), "truncated": omitted > 0,
        "changed_files_total": len(files), "changed_text_files_total": sum(row.get("kind") == "text" for row in files),
        "files_with_semantic_content": len({x["path"] for x in selected_chunks}),
        "hunks_total": len(hunk_items), "hunks_included": included_hunks,
        "hunk_chunks_total": len(hunk_items), "hunk_chunks_included": len(selected_chunks),
        "changed_lines_total": sum(x["changed_lines"] for x in hunk_items), "changed_lines_included": included_lines,
        "omitted_hunks": omitted, "omitted_changed_lines": sum(x["changed_lines"] for x in hunk_items) - included_lines,
    }
    return text, stats
