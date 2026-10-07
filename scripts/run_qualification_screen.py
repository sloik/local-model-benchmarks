#!/usr/bin/env python3
"""
run_qualification_screen.py -- SPEC-003-017 live driver.

Loads exactly one arm at a time via `lms load`/`lms unload`, running the
full corpus through scripts/qualify_compact_models.py's subject_fn adapter
and scripts/run_capability_suite.py's run_case dispatcher, journaling every
(arm, case, repetition, reload_block) result. Resumable: re-running this
script skips any journal entry already present.

Usage:
    python3 scripts/run_qualification_screen.py --pilot   # 5 cases, 1 rep, 1 arm, timing only
    python3 scripts/run_qualification_screen.py --run      # full screen, all arms
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH_DIR / "suites" / "capability"))

import qualify_compact_models as qcm  # noqa: E402
import run_capability_suite as runner  # noqa: E402
import score_capability_suite as scorer  # noqa: E402
import contract as capability_contract  # noqa: E402

REPETITIONS = 3
RELOAD_BLOCKS = 3


def lms_load(model_key: str, context_length: int, timeout: int = 300) -> None:
    subprocess.run(
        ["lms", "load", model_key, "--context-length", str(context_length), "--yes"],
        check=True, capture_output=True, text=True, timeout=timeout,
    )


def lms_unload(model_key: str, timeout: int = 60) -> None:
    subprocess.run(["lms", "unload", model_key], check=True, capture_output=True, text=True, timeout=timeout)


def lms_ps() -> list[dict]:
    proc = subprocess.run(["lms", "ps", "--json"], capture_output=True, text=True, timeout=30)
    proc.check_returncode()
    return json.loads(proc.stdout)


def run_arm(arm: dict, corpus: dict, common_context: int, reps: int, reload_blocks: int,
            case_limit: int | None = None) -> dict:
    """Run one arm's full (reload_block x repetition x case) matrix, journaling
    each result. Reloads the model fresh at each reload_block boundary so
    repeats are drawn from genuinely independent process/KV-cache state."""
    arm_id = arm["arm_id"]
    model_key = arm["model_key"]
    cases = corpus["cases"][:case_limit] if case_limit else corpus["cases"]
    journal = qcm.read_journal()
    identity = {"model_key": model_key, "runtime": "lmstudio"}
    timings = []

    for reload_block in range(reload_blocks):
        pre_inventory = lms_ps()
        if pre_inventory:
            return {"arm_id": arm_id, "status": "halted", "reason": "runtime_not_idle_before_load",
                     "reload_block": reload_block}

        lms_load(model_key, common_context)
        loaded = lms_ps()
        if not loaded or loaded[0].get("modelKey") != model_key:
            return {"arm_id": arm_id, "status": "halted", "reason": "load_identity_mismatch",
                     "reload_block": reload_block, "observed": loaded}

        subject_fn = qcm.build_subject_fn(model_key, "lmstudio", common_context)

        for rep in range(reps):
            for case in cases:
                if qcm.already_done(journal, arm_id, case["case_id"], rep, reload_block):
                    continue
                t0 = time.time()
                try:
                    result = runner.run_case(case, subject_fn, identity, {"structured_snapshot"})
                except runner.RunnerError as exc:
                    result = {"profile": "structured_snapshot", "final_response": None,
                              "tool_calls_made": [], "error": f"identity_mismatch:{exc}", "coverage": True}
                elapsed = time.time() - t0
                timings.append(elapsed)
                verdict = scorer.score_run_result(case, result)
                entry = {
                    "arm_id": arm_id, "case_id": case["case_id"], "repetition": rep,
                    "reload_block": reload_block, "logical_pair_id": case["logical_pair_id"],
                    "capability": case["capability"], "scenario_skin": case["scenario_skin"],
                    "verdict": verdict["verdict"], "reason": verdict["reason"],
                    "elapsed_s": round(elapsed, 3), "raw_final_response": result.get("final_response"),
                    "raw_error": result.get("error"),
                }
                qcm.append_journal(entry)
                journal.append(entry)

        # R2: unload the run-owned instance on every terminal path within this block.
        lms_unload(model_key)
        post_inventory = lms_ps()
        if post_inventory:
            return {"arm_id": arm_id, "status": "halted", "reason": "cleanup_failure_instance_remained",
                     "reload_block": reload_block, "observed": post_inventory}

    return {"arm_id": arm_id, "status": "complete", "n_calls": len(timings),
            "mean_elapsed_s": round(sum(timings) / len(timings), 3) if timings else None,
            "total_elapsed_s": round(sum(timings), 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="5 cases, 1 rep, 1 reload block, qwen arm only -- timing pilot")
    parser.add_argument("--run", action="store_true", help="full screen across all available arms")
    parser.add_argument("--arm", help="restrict to one arm_id")
    args = parser.parse_args()

    if not args.pilot and not args.run:
        parser.print_help()
        return 1

    if lms_ps():
        print("REFUSING: LM Studio runtime is not idle (lms ps returned a loaded instance).", file=sys.stderr)
        return 1

    manifest = qcm.build_candidate_manifest()
    corpus = capability_contract.load_corpus()
    measured = qcm.measure_max_serialized_input(corpus)
    selected = qcm.select_common_context(manifest, measured)
    common_context = selected["common_context"]

    qcm.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    qcm.MANIFEST_PATH.write_text(json.dumps({
        "manifest": manifest, "measured_context": measured, "selected_context": selected,
        "authorization": qcm.AUTHORIZATION,
    }, indent=2), encoding="utf-8")

    available_arms = [a for a in manifest["arms"].values() if a["status"] == "available"]

    if args.pilot:
        arm = next(a for a in available_arms if a["arm_id"] == "qwen2.5-0.5b")
        print(f"PILOT: arm={arm['arm_id']} common_context={common_context}")
        result = run_arm(arm, corpus, common_context, reps=1, reload_blocks=1, case_limit=5)
        print(json.dumps(result, indent=2))
        return 0

    targets = [a for a in available_arms if (not args.arm or a["arm_id"] == args.arm)]
    for arm in targets:
        print(f"ARM {arm['arm_id']}: common_context={common_context} reps={REPETITIONS} reload_blocks={RELOAD_BLOCKS}")
        result = run_arm(arm, corpus, common_context, reps=REPETITIONS, reload_blocks=RELOAD_BLOCKS)
        print(json.dumps(result, indent=2))
        if result["status"] != "complete":
            print(f"HALTING further arms: {arm['arm_id']} did not complete cleanly.", file=sys.stderr)
            return 1

    if lms_ps():
        print("POST-RUN CHECK FAILED: runtime not idle after all arms.", file=sys.stderr)
        return 1
    print("All arms complete. Runtime confirmed idle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
