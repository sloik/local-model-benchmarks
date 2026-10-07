#!/usr/bin/env python3
"""Check the public file inventory and every reachable Git blob for release leaks."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

EXCLUDED_ROOTS = {".nightshift", ".agent-context", ".serena", ".venv", "reports", "runs", "knowledge", "outputs", "exercises"}
FORBIDDEN = {
    "personal-machine-path": re.compile(rb"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "private-storage-reference": re.compile(rb"Drop[b]ox|KRU[-]Destylat|Analizy[-]Spolek|przeanalizo[w]ane", re.I),
    "excluded-issuer-source": re.compile(rb"\b(?:KRU[K]|Moo[d]y|Diagnos[t]yka)\b", re.I),
    "personal-name": re.compile(bytes.fromhex("c581756b61737a")),
    "provider-key": re.compile(rb"\bsk-(?:ant-[A-Za-z0-9_-]+|[A-Za-z0-9_-]{30,})\b"),
    "github-token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b|github_pat_[A-Za-z0-9_]{50,}"),
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws-access-key": re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
}
EMAIL = re.compile(rb"[A-Za-z0-9_.+%-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def content_findings(data: bytes) -> list[str]:
    findings = [name for name, pattern in FORBIDDEN.items() if pattern.search(data)]
    for match in EMAIL.finditer(data):
        if match.group(0).lower() == b"noreply@github.com":
            continue
        if match.group(1).lower() not in {b"users.noreply.github.com", b"example.com", b"example.org", b"example.net", b"example.invalid"}:
            findings.append("non-public-contact-address")
            break
    return findings


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def audit_tree(root: Path) -> dict:
    files = [p.decode() for p in git(root, "ls-files", "-z").split(b"\0") if p]
    findings = []
    for name in files:
        path = root / name
        if name.split("/")[0] in EXCLUDED_ROOTS:
            findings.append({"file": name, "kind": "internal-artifact"})
        elif path.is_symlink():
            findings.append({"file": name, "kind": "symlink-needs-review"})
        elif path.is_file():
            findings.extend({"file": name, "kind": kind} for kind in content_findings(path.read_bytes()))
        else:
            findings.append({"file": name, "kind": "missing-tracked-file"})
    return {"files": len(files), "findings": findings}


def audit_history(root: Path) -> dict:
    if subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], capture_output=True).returncode:
        return {"commits": 0, "objects": 0, "findings": []}
    refs = git(root, "for-each-ref", "--format=%(refname)").decode().splitlines()
    objects = git(root, "rev-list", "--objects", "--all").decode().splitlines()
    findings = []
    for line in objects:
        oid, _, name = line.partition(" ")
        kind = git(root, "cat-file", "-t", oid).decode().strip()
        if kind in {"blob", "commit", "tag"}:
            findings.extend({"object": oid, "file": name, "kind": item}
                            for item in content_findings(git(root, "cat-file", "-p", oid)))
        if name and name.split("/")[0] in EXCLUDED_ROOTS:
            findings.append({"object": oid, "file": name, "kind": "internal-artifact"})
    return {"commits": int(git(root, "rev-list", "--count", "--all")),
            "objects": len(objects), "refs": refs, "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    tree, history = audit_tree(args.root), audit_history(args.root)
    result = {"tree": tree, "history": history,
              "scope": "Tracked files and every reachable blob/commit/tag; pattern checks complement provenance review."}
    print(json.dumps(result, indent=2))
    return 1 if tree["findings"] or history["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
