#!/opt/miniconda3/envs/testbed/bin/python
# Copyright (c) Microsoft. All rights reserved.
"""Literal, checked source replacement for a disposable SWE task container."""

import argparse
import difflib
import sys
from pathlib import Path


def parse_edit(text):
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "<<<<<<< SEARCH":
        raise ValueError("Start the input with <<<<<<< SEARCH")
    markers = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == "======="]
    if len(markers) != 1 or lines[-1].rstrip("\r\n") != ">>>>>>> REPLACE":
        raise ValueError("Expected one SEARCH / ======= / REPLACE block")
    middle = markers[0]
    old, new = "".join(lines[1:middle]), "".join(lines[middle + 1 : -1])
    if not old:
        raise ValueError("SEARCH must contain exact existing text")
    return old, new


def replace(path, old, new, *, root=Path("/testbed")):
    root = root.resolve()
    path = root / path
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or path.is_symlink() or not resolved.is_file():
        raise ValueError("Edit an existing regular source file under /testbed")
    before = resolved.read_text()
    count = before.count(old)
    if count != 1:
        raise ValueError(f"SEARCH matched {count} times; read the file and include exact, unique context")
    if old == new:
        raise ValueError("SEARCH and REPLACE are identical; no edit was made")
    after = before.replace(old, new, 1)
    resolved.write_text(after)
    return "".join(
        difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=str(path), tofile=str(path))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        old, new = parse_edit(sys.stdin.read())
        print(replace(args.path, old, new))
        print("Applied exactly one literal replacement. Verify the behavior before submitting.")
    except (ValueError, OSError) as error:
        print(f"Edit rejected: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
