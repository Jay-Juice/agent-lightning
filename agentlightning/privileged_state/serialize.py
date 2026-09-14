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
