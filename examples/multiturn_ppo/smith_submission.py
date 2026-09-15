# Copyright (c) Microsoft. All rights reserved.

"""Source-only submission feedback; never inspect grading tests or reference fixes."""

import json

WORKFLOW_HINT = """

## Reliable editing and verification
- Put temporary reproduction scripts under /tmp (for example /tmp/repro.py),
  not in /testbed. Tests and configuration changes cannot be submitted.
- Use a multiline Python heredoc when a script contains def, try, with, or loops:
  python - <<'PY'
  # ordinary multiline Python here
  PY
  Keep this entire shell command inside your single bash code block.
- An exit code of zero from sed does not mean it matched or changed anything.
  For replacements, assert that the expected old text exists, write the change,
  then read back the relevant lines to confirm it actually happened.
- Re-run your reproduction with assertions after editing. A printed result alone
  is not a passing assertion. Fix all affected paths described in the issue.
- Submit only after changing the source and checking the result.
"""

EDITOR_HINT = """

## Checked source editor
Use agl-edit for source edits. It performs a literal replacement, requires exactly
one match, and shows the actual diff. Read the file first and copy its old text
exactly, including indentation. No regular expressions or Python string escaping
are needed. Put one command like this inside your single bash code block:

agl-edit src/path.py <<'EDIT'
<<<<<<< SEARCH
exact existing lines
=======
replacement lines
>>>>>>> REPLACE
EDIT

If the editor rejects a match, read the relevant lines again and correct SEARCH.
After an edit, run a reproduction with assertions before submitting.
"""


def submission_feedback(patch, paths, rejection):
    if rejection:
        return (
            "Submission not accepted: your patch contains a prohibited test, configuration, "
            "or symlink change. Changed paths: " + ", ".join(paths) + ". "
            "Restore protected files; move your temporary reproduction scripts to /tmp. "
            "Keep your source fix, verify it, then submit again. No grading tests were run."
        )
    if not patch.strip():
        return (
            "Submission not accepted: there are no changes in /testbed. "
            "Your editing command may have matched no text. Read back the relevant source, "
            "make the actual fix, verify it with your reproduction, then submit again. "
            "No grading tests were run."
        )
    return None


def syntax_check_command(paths):
    """Parse changed Python sources inside the task's Python environment.

    Does not execute model code, import packages, write bytecode, or use tests.
    JSON embedded in a quoted heredoc keeps filenames out of shell syntax.
    """
    encoded = repr(json.dumps([p for p in paths if p.endswith(".py")]))
    return (
        "python - <<'AGL_SYNTAX_CHECK'\n"
        "import ast, json, pathlib\n"
        f"for name in json.loads({encoded}):\n"
        "    path = pathlib.Path(name)\n"
        "    if path.is_file():\n"
        "        ast.parse(path.read_bytes(), filename=name)\n"
        "print('Changed Python sources parse successfully')\n"
        "AGL_SYNTAX_CHECK"
    )
