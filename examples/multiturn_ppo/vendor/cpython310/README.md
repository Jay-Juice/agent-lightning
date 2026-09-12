# CPython URL parsing compatibility for the Furl image

`parse.py` is an unmodified copy of CPython v3.10.0 `Lib/urllib/parse.py`.

- Source: https://github.com/python/cpython/blob/v3.10.0/Lib/urllib/parse.py
- SHA-256: `f14ec821fe7ded126d2202eef195f66bd63f74975c9ae88322d0500f13eba8db`
- License: the accompanying unmodified CPython v3.10.0 `LICENSE`.
- License SHA-256: `d0285b61e1a8e420c7deb95836738a5d4a0d26463138b17601f5971212684c4b`

The supplied `gruns__furl.da386f68` image runs Python 3.10.16, whose URL
validation and reconstruction behavior differs from that expected by three
unchanged project tests. Its reference repair therefore fails. Replacing only
this standard-library module with the original 3.10 implementation makes the
empty-patch negative and reference-patch positive controls pass (1 F2P, 72 P2P).

`FullPythonSandbox.prepare` verifies this hash and installs the copy only in
the disposable, offline Furl containers. Agent actions and grading use the same
compatibility module. The host environment and other projects are unchanged.
Do not format or otherwise modify the vendored file.
