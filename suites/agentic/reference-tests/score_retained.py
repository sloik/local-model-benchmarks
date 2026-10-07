"""Score retained model implementations against the reference suites (SPEC-132 AC9).

Reads every results/<cell>/EVAL-00N/src/ produced by a past eval run and scores it
against our reference suite instead of the model's self-authored tests. Emits a
self-vs-reference delta table.

No model is re-run: this uses output already on disk. Counts come from pytest's own
report, not the sed parser that BUG-001 describes.

Usage:  python3 reference-tests/score_retained.py [--json out.json]
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference-tests"
RESULTS = ROOT / "results"

SUITES = {
    "EVAL-001": ("fizzbuzz", REF / "test_eval001_fizzbuzz.py", 7),
    "EVAL-002": ("csv_parser", REF / "test_eval002_csv_parser.py", 12),
    "EVAL-003": ("retry", REF / "test_eval003_retry.py", 10),
    "EVAL-005": ("calculator", REF / "test_eval005_calculator.py", 7),
}


def score(src_dir: Path, suite: Path) -> dict:
    """Run one reference suite against one src dir. Returns pytest's own counts."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(suite), "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=str(ROOT),
        env={"EVAL_SRC_DIR": str(src_dir), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout
    passed = failed = errors = 0
    for line in out.splitlines():
        if " passed" in line or " failed" in line or " error" in line:
            for tok, name in ((" passed", "passed"), (" failed", "failed"), (" error", "errors")):
                if tok in line:
                    try:
                        n = int(line.split(tok)[0].strip().split()[-1])
                    except (ValueError, IndexError):
                        continue
                    if name == "passed":
                        passed = n
                    elif name == "failed":
                        failed = n
                    else:
                        errors = n
    collected = passed + failed + errors
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "collected": collected,
        "exit_code": proc.returncode,
        # importorskip yields 0 collected: the module is missing or unimportable
        "importable": collected > 0,
    }


def main() -> None:
    rows = []
    for cell_dir in sorted(RESULTS.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name.startswith("."):
            continue
        for eval_id, (_mod, suite, ac_count) in SUITES.items():
            src = cell_dir / eval_id / "src"
            if not src.is_dir():
                continue
            rj = cell_dir / eval_id / "result.json"
            self_status = self_ts = None
            if rj.exists():
                try:
                    j = json.loads(rj.read_text())
                    self_status = j.get("status")
                    self_ts = (j.get("timestamp") or "")[:10]
                except (json.JSONDecodeError, OSError):
                    pass
            r = score(src, suite)
            ref_status = "pass" if (r["importable"] and r["failed"] == 0
                                    and r["errors"] == 0 and r["passed"] == ac_count) else "fail"
            rows.append({
                "cell": cell_dir.name, "eval": eval_id, "run_date": self_ts,
                "self_status": self_status, "ref_status": ref_status,
                "ref_passed": r["passed"], "ref_total": ac_count,
                "importable": r["importable"],
            })

    hdr = f"{'cell':<36}{'eval':<10}{'date':<12}{'self':<9}{'REFERENCE':<12}{'delta'}"
    print(hdr)
    print("-" * len(hdr))
    deltas = 0
    for r in rows:
        d = ""
        if r["self_status"] == "pass" and r["ref_status"] == "fail":
            d = "<<< SELF-PASS / REF-FAIL"
            deltas += 1
        elif r["self_status"] in ("fail", "failed") and r["ref_status"] == "pass":
            d = "ref-pass (self-fail)"
            deltas += 1
        ref = f"{r['ref_passed']}/{r['ref_total']}"
        if not r["importable"]:
            ref += " (no mod)"
        print(f"{r['cell']:<36}{r['eval']:<10}{str(r['run_date']):<12}"
              f"{str(r['self_status']):<9}{ref:<12}{d}")
    print(f"\n{deltas} self-vs-reference disagreements across {len(rows)} scored implementations")

    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
