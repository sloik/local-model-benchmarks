#!/usr/bin/env python3
"""Unified benchmark registry, preflight, and Hermes agentic runner."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "suites" / "registry.json"
AGENTIC_SUITE_PATH = ROOT / "suites" / "agentic" / "suite.json"
RESULTS_ROOT = ROOT / "results"
DEFAULT_CONTEXT = 131072
MINIMUM_CONTEXT = 131072
PARALLEL_PREDICTIONS = 1
MAX_TIMEOUT_SECONDS = 86400
MAX_FREE_MEMORY_MB = 1048576

REQUIRED_SUITE_KEYS = {
    "id", "family", "purpose", "runner", "scorer", "case_registry", "cases",
    "prerequisites", "canonical_result_root",
}
class ContractError(ValueError):
    """The suite contract or requested policy is invalid."""


class PreflightError(RuntimeError):
    """A runtime prerequisite is missing or inconsistent."""


class ContextDriftError(PreflightError):
    """A case lost its exact model-instance/context invariant."""


def load_json(path: Path) -> dict[str, Any]:
    """The one JSON configuration reader used by registry, suite, and schema paths."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON contract must be an object: {path}")
    return value


def validate_registry(registry: dict[str, Any], root: Path = ROOT) -> None:
    suites = registry.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ContractError("registry.suites must be a non-empty list")
    families = set()
    for suite in suites:
        if not isinstance(suite, dict):
            raise ContractError("each registry suite must be an object")
        missing = REQUIRED_SUITE_KEYS - suite.keys()
        if missing:
            raise ContractError(f"suite {suite.get('id', '<unknown>')} missing {sorted(missing)}")
        if not suite["cases"] or not suite["prerequisites"]:
            raise ContractError(f"suite {suite['id']} must declare cases and prerequisites")
        for key in ("runner", "scorer"):
            if not (root / suite[key]).is_file():
                raise ContractError(f"suite {suite['id']} {key} is absent: {suite[key]}")
        case_registry = root / suite["case_registry"]
        if suite["family"] == "canonical":
            if not case_registry.is_dir():
                raise ContractError(f"canonical case registry is absent: {case_registry}")
        elif not case_registry.is_file():
            raise ContractError(f"suite case registry is absent: {case_registry}")
        families.add(suite["family"])
    if families != {"canonical", "agentic"}:
        raise ContractError(f"registry must cover canonical and agentic families, got {families}")


def validate_agentic_suite(suite: dict[str, Any], root: Path = ROOT) -> None:
    policy = suite.get("policy") or {}
    if policy.get("default_context") != DEFAULT_CONTEXT:
        raise ContractError(f"agentic default context must be exactly {DEFAULT_CONTEXT}")
    if policy.get("minimum_context") != MINIMUM_CONTEXT:
        raise ContractError(f"agentic minimum context must be exactly {MINIMUM_CONTEXT}")
    validate_runtime_policy(policy)
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) != 5:
        raise ContractError("agentic suite must declare the five canonical EVAL cases")
    reference_root = root / suite.get("reference_root", "")
    fixture_root = root / suite.get("fixture_root", "")
    if not reference_root.is_dir() or not fixture_root.is_dir():
        raise ContractError("agentic fixture_root and reference_root must be benchmark-owned directories")
    if reference_root == fixture_root or reference_root in fixture_root.parents:
        raise ContractError("held-out reference root must remain outside the model fixture tree")
    importer = suite.get("legacy_importer")
    if not isinstance(importer, str) or not (root / importer).is_file():
        raise ContractError("agentic legacy_importer must be a benchmark-owned file")
    validation = suite.get("validation")
    if not isinstance(validation, dict):
        raise ContractError("agentic validation contract must be benchmark-owned metadata")
    required_validation = {"manifest_schema", "test_files", "spec", "test_command"}
    missing_validation = required_validation - validation.keys()
    if missing_validation:
        raise ContractError(f"agentic validation contract missing {sorted(missing_validation)}")
    if validation["manifest_schema"] != suite.get("result_schema"):
        raise ContractError("agentic validation manifest schema must equal result_schema")
    validation_paths = [validation["manifest_schema"], validation["spec"]]
    test_files = validation["test_files"]
    if not isinstance(test_files, list) or not test_files:
        raise ContractError("agentic validation test_files must be a non-empty list")
    validation_paths.extend(test_files)
    if any(not isinstance(path, str) or not (root / path).is_file() for path in validation_paths):
        raise ContractError("agentic validation paths must be benchmark-owned files")
    if validation["test_command"] != "python3 -m pytest tests/":
        raise ContractError("agentic validation test_command must be the canonical project gate")
    for case in cases:
        if case.get("source_basis") != "synthetic":
            raise ContractError(f"{case.get('id', '<unknown>')} source_basis must be synthetic")
        for key in ("spec", "source", "self_test", "prompt"):
            if not case.get(key):
                raise ContractError(f"{case.get('id', '<unknown>')} missing {key}")
        if not (root / case["spec"]).is_file():
            raise ContractError(f"case spec absent: {case['spec']}")
        if not (fixture_root / case["source"]).is_file():
            raise ContractError(f"case source fixture absent: {case['source']}")
        if not (fixture_root / case["self_test"]).is_file():
            raise ContractError(f"case test fixture absent: {case['self_test']}")
        reference_test = case.get("reference_test")
        if reference_test and not (root / reference_test).is_file():
            raise ContractError(f"held-out reference test absent: {reference_test}")


def validate_runtime_policy(policy: dict[str, Any]) -> None:
    if policy.get("case_model_isolation") != "unload_all_then_load_exact_context":
        raise ContractError(
            "case_model_isolation must be unload_all_then_load_exact_context"
        )
    if policy.get("parallel_predictions") != PARALLEL_PREDICTIONS:
        raise ContractError(
            f"parallel_predictions must be exactly {PARALLEL_PREDICTIONS}"
        )
    timeout = policy.get("timeout_seconds")
    minimum_memory = policy.get("minimum_free_memory_mb")
    if not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ContractError(f"timeout_seconds must be in range 1..{MAX_TIMEOUT_SECONDS}")
    if not isinstance(minimum_memory, int) or not 1 <= minimum_memory <= MAX_FREE_MEMORY_MB:
        raise ContractError(f"minimum_free_memory_mb must be in range 1..{MAX_FREE_MEMORY_MB}")
    poll_seconds = policy.get("context_watchdog_poll_seconds")
    failure_tolerance = policy.get("context_api_failure_tolerance")
    if not isinstance(poll_seconds, (int, float)) or not 0 < poll_seconds <= 30:
        raise ContractError("context_watchdog_poll_seconds must be in range (0, 30]")
    if not isinstance(failure_tolerance, int) or not 1 <= failure_tolerance <= 5:
        raise ContractError("context_api_failure_tolerance must be in range 1..5")


def resolve_context(requested: int, minimum: int = MINIMUM_CONTEXT) -> int:
    if requested < minimum:
        raise ContractError(f"context {requested} is below required minimum {minimum}")
    return requested


def load_command(
    model: str, context: int, parallel_predictions: int = PARALLEL_PREDICTIONS
) -> list[str]:
    if parallel_predictions != PARALLEL_PREDICTIONS:
        raise ContractError(
            f"parallel_predictions must be exactly {PARALLEL_PREDICTIONS}"
        )
    return [
        "lms", "load", model, "--context-length", str(resolve_context(context)),
        "--parallel", str(parallel_predictions), "--yes",
    ]


@dataclass(frozen=True)
class LoadedObservation:
    model: str
    instance_id: str
    context_length: int
    observed_at: str

    def as_evidence(self) -> dict[str, Any]:
        return {
            "state": "observed",
            "model": self.model,
            "instance_id": self.instance_id,
            "context_length": self.context_length,
            "observed_at": self.observed_at,
        }


def observe_loaded_instance(
    model: str, metadata: dict[str, Any], *, now: Callable[[], float] = time.time
) -> LoadedObservation:
    models = metadata.get("models", metadata.get("data", []))
    record = next(
        (item for item in models if item.get("key", item.get("id")) == model), None
    )
    if record is None:
        raise PreflightError(f"model {model} absent from LM Studio native metadata")
    instances = record.get("loaded_instances") or []
    if not instances:
        raise PreflightError(f"model {model} is not loaded after lms load")
    if len(instances) != 1:
        raise PreflightError(f"model {model} has {len(instances)} loaded instances; expected exactly one")
    instance = instances[0]
    observed = (instance.get("config") or {}).get("context_length")
    if not isinstance(observed, int):
        raise PreflightError(f"model {model} has no integer observed context")
    identity = next(
        (
            str(instance[key]) for key in ("id", "instance_id", "identifier", "key")
            if instance.get(key) is not None
        ),
        None,
    )
    if identity is None:
        identity = "sha256:" + hashlib.sha256(
            json.dumps(instance, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return LoadedObservation(
        model=model,
        instance_id=identity,
        context_length=observed,
        observed_at=datetime.fromtimestamp(now(), timezone.utc).isoformat(),
    )


def verify_observed_context(model: str, requested: int, metadata: dict[str, Any]) -> int:
    observed = observe_loaded_instance(model, metadata).context_length
    if observed != requested:
        raise PreflightError(
            f"model {model} observed context {observed} does not match requested {requested}"
        )
    return observed


def verify_observed_parallel(
    model: str, requested: int, processes: Any
) -> int:
    if not isinstance(processes, list):
        raise PreflightError("lms ps --json returned a non-list process state")
    matches = [
        item for item in processes
        if isinstance(item, dict)
        and model in {item.get("modelKey"), item.get("identifier")}
    ]
    if not matches:
        raise PreflightError(f"model {model} absent from lms ps process state")
    if len(matches) != 1:
        raise PreflightError(
            f"model {model} has {len(matches)} process records; expected exactly one"
        )
    observed = matches[0].get("parallel")
    if not isinstance(observed, int):
        raise PreflightError(f"model {model} has no integer observed parallel value")
    if observed != requested:
        raise PreflightError(
            f"model {model} observed parallel {observed} does not match requested {requested}"
        )
    return observed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_identity(root: Path) -> dict[str, Any]:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and not {".pytest_cache", ".ruff_cache"}.intersection(path.parts)
    )
    rows = [{"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for path in files]
    digest = hashlib.sha256(
        "".join(f"{row['path']}\0{row['sha256']}\n" for row in rows).encode()
    ).hexdigest()
    return {"sha256": digest, "files": rows}


def validate_manifest(manifest: dict[str, Any]) -> None:
    schema = load_json(ROOT / "suites" / "schemas" / "agentic-run-manifest.schema.json")
    missing = set(schema["required"]) - manifest.keys()
    if missing:
        raise ContractError(f"run manifest missing {sorted(missing)}")
    if manifest["schema_version"] != 1 or manifest["suite_id"] != "agentic-coding":
        raise ContractError("unsupported agentic run manifest identity")
    resolve_context(manifest["requested_context"])
    if manifest["provenance"] not in {"native-unified", "imported-legacy-layout"}:
        raise ContractError("invalid manifest provenance")
    if manifest["provenance"] == "imported-legacy-layout":
        legacy = manifest.get("legacy_import")
        required_legacy = {
            "source_absolute_path", "source_portable_path", "source_git_commit",
            "source_summary_path", "source_summary_sha256", "source_aggregate_sha256",
            "destination_aggregate_sha256", "imported_at", "files",
        }
        if not isinstance(legacy, dict):
            raise ContractError("imported legacy manifest requires legacy_import object")
        legacy_missing = required_legacy - legacy.keys()
        if legacy_missing:
            raise ContractError(f"legacy_import missing {sorted(legacy_missing)}")
        if not isinstance(manifest.get("case_statuses"), dict):
            raise ContractError("imported legacy manifest requires case_statuses object")
        if legacy["source_aggregate_sha256"] != legacy["destination_aggregate_sha256"]:
            raise ContractError("imported legacy source/destination aggregate hashes differ")
    observed = manifest["observed_contexts"]
    if not isinstance(observed, dict):
        raise ContractError("observed_contexts must be an object")
    for model, value in observed.items():
        if value is not None and value != manifest["requested_context"]:
            raise ContractError(f"observed context mismatch for {model}: {value}")
    requested_parallel = manifest.get("requested_parallel_predictions")
    if requested_parallel is not None and requested_parallel != PARALLEL_PREDICTIONS:
        raise ContractError(
            f"requested parallel predictions must be exactly {PARALLEL_PREDICTIONS}"
        )
    observed_parallel = manifest.get("observed_parallel_predictions")
    if observed_parallel is not None:
        if not isinstance(observed_parallel, dict):
            raise ContractError("observed_parallel_predictions must be an object")
        for model, value in observed_parallel.items():
            if value is not None and value != PARALLEL_PREDICTIONS:
                raise ContractError(f"observed parallel mismatch for {model}: {value}")


def _version(argv: list[str], runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> str:
    proc = runner(argv, capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        raise PreflightError(f"dependency probe failed: {' '.join(argv)}")
    output = (proc.stdout or proc.stderr).strip().splitlines()
    return output[0] if output else "available-version-unreported"


def resolve_credential(env: dict[str, str]) -> tuple[str, str] | None:
    if env.get("LM_STUDIO_TOKEN"):
        return env["LM_STUDIO_TOKEN"], "LM_STUDIO_TOKEN"
    if env.get("LM_API_KEY"):
        return env["LM_API_KEY"], "LM_API_KEY"
    return None


def free_memory_mb() -> int:
    if sys.platform != "darwin":
        return 999999
    page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True))
    text = subprocess.check_output(["vm_stat"], text=True)
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages purgeable"):
        for line in text.splitlines():
            if line.startswith(label):
                pages += int(line.split(":", 1)[1].strip().rstrip("."))
    return pages * page_size // 1048576


def preflight(
    suite: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    version_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    api_probe: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Probe capabilities only; never load a model, invoke Hermes chat, or write results."""
    env = dict(os.environ if env is None else env)
    if sys.platform != "darwin":
        raise PreflightError(f"agentic suite requires macOS, found {platform.system()}")
    required = ("zsh", "python3", "lms", "hermes", "caffeinate")
    paths = {name: which(name) for name in required}
    missing = [name for name, path in paths.items() if not path]
    if missing:
        raise PreflightError(f"missing required commands: {', '.join(missing)}")
    credential = resolve_credential(env)
    if credential is None:
        raise PreflightError("missing LM Studio credential (LM_STUDIO_TOKEN or LM_API_KEY)")
    token, credential_source = credential
    minimum_memory = suite["policy"]["minimum_free_memory_mb"]
    memory = free_memory_mb()
    if memory < minimum_memory:
        raise PreflightError(f"free memory {memory}MB is below required {minimum_memory}MB")
    versions = {
        "platform": platform.platform(),
        "zsh": _version([paths["zsh"], "--version"], version_runner),
        "python": platform.python_version(),
        "pytest": _version([paths["python3"], "-m", "pytest", "--version"], version_runner),
        "lms": _version([paths["lms"], "--version"], version_runner),
        "hermes": _version([paths["hermes"], "--version"], version_runner),
        "caffeinate": paths["caffeinate"],
        "credential_source": credential_source,
        "free_memory_mb": memory,
    }
    api_root = suite["dependencies"]["lm_studio_api"]
    try:
        api_payload = (api_probe or native_models)(api_root, token)
    except Exception as exc:
        raise PreflightError(f"LM Studio API is unreachable: {exc}") from exc
    if not isinstance(api_payload, dict):
        raise PreflightError("LM Studio API probe returned a non-object response")
    api_models = api_payload.get("models", api_payload.get("data", []))
    versions["lm_studio_api"] = {
        "endpoint": api_root,
        "status": "reachable",
        "model_count": len(api_models) if isinstance(api_models, list) else None,
        "api_version": api_payload.get("version"),
    }
    return versions


def native_models(api_root: str, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(f"{api_root.rstrip('/')}/api/v1/models", headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def build_manifest(
    models: list[str], suite: dict[str, Any], dependencies: dict[str, Any], context: int,
    cases: list[str], output_path: Path, observed: dict[str, int | None], *, dry_run: bool,
    model_artifacts: dict[str, Any] | None = None,
    observed_parallel: dict[str, int | None] | None = None,
    force_requested: bool = False,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    fixture_root = ROOT / suite["fixture_root"]
    scorer = ROOT / suite["scorer"]
    manifest = {
        "schema_version": 1,
        "run_id": now.strftime("%Y-%m-%dT%H%M%SZ") + "_agentic",
        "suite_id": suite["suite_id"],
        "models": models,
        "harness": {"name": "hermes", "version": dependencies["hermes"]},
        "requested_context": resolve_context(context),
        "observed_contexts": observed,
        "requested_parallel_predictions": suite["policy"]["parallel_predictions"],
        "observed_parallel_predictions": observed_parallel or {
            model: None for model in models
        },
        "observation_state": "not-loaded-dry-run" if dry_run else "verified-exact",
        "timeout_seconds": suite["policy"]["timeout_seconds"],
        "minimum_free_memory_mb": suite["policy"]["minimum_free_memory_mb"],
        "max_turns": suite["policy"]["max_turns"],
        "tools": suite["policy"]["toolsets"],
        "dependencies": dependencies,
        "suite_identity": {"path": str(AGENTIC_SUITE_PATH.relative_to(ROOT)),
                           "sha256": _sha256(AGENTIC_SUITE_PATH)},
        "model_artifacts": model_artifacts or {
            model: {"state": "not-inspected-dry-run"} for model in models
        },
        "fixture_identity": tree_identity(fixture_root),
        "scorer_identity": {"path": suite["scorer"], "sha256": _sha256(scorer)},
        "cases": cases,
        "started_at": now.isoformat(),
        "canonical_output_path": str(output_path),
        "provenance": "native-unified",
        "force_policy": {
            "requested": force_requested,
            "deprecated_compatibility_flag": force_requested,
            "overwrite_existing_evidence": False,
            "skip_existing_evidence": False,
            "behavior": "always-create-new-timestamped-run",
        },
    }
    validate_manifest(manifest)
    return manifest


@dataclass
class CommandExecutor:
    """Injectable subprocess boundary used by the live runner and unit tests."""

    run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    popen: Callable[..., subprocess.Popen] = subprocess.Popen

    def load_model(
        self, model: str, context: int,
        parallel_predictions: int = PARALLEL_PREDICTIONS,
    ) -> int:
        unload = self.run(["lms", "unload", "--all"], capture_output=True, text=True)
        if unload.returncode != 0:
            raise PreflightError(f"lms unload failed before loading {model}: {unload.stderr.strip()}")
        proc = self.run(
            load_command(model, context, parallel_predictions),
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise PreflightError(f"lms load failed for {model}: {proc.stderr.strip()}")
        state = self.run(["lms", "ps", "--json"], capture_output=True, text=True)
        if state.returncode != 0:
            raise PreflightError(f"lms ps failed after loading {model}: {state.stderr.strip()}")
        try:
            processes = json.loads(state.stdout)
        except json.JSONDecodeError as exc:
            raise PreflightError(f"lms ps returned invalid JSON after loading {model}") from exc
        return verify_observed_parallel(model, parallel_predictions, processes)

    def start_hermes(
        self, model: str, prompt: str, cwd: Path, suite: dict[str, Any],
        env: dict[str, str], log_stream: Any,
    ) -> subprocess.Popen:
        policy = suite["policy"]
        return self.popen(
            [
                "hermes", "chat", "-q", prompt, "--provider", "lmstudio", "-m", model,
                "--yolo", "-Q", "--ignore-rules", "--max-turns", str(policy["max_turns"]),
                "-t", ",".join(policy["toolsets"]),
            ],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            text=True,
        )


@dataclass(frozen=True)
class IsolatedHermesEnvironment:
    home: Path
    env: dict[str, str]
    resolved_context: int


@contextmanager
def isolated_hermes_environment(
    *, model: str, context: int, api_root: str, token: str,
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    base_env: dict[str, str] | None = None,
) -> Iterator[IsolatedHermesEnvironment]:
    """Create and prove a private Hermes config without consulting ambient state."""
    context = resolve_context(context)
    base_url = f"{api_root.rstrip('/')}/v1"
    with tempfile.TemporaryDirectory(prefix="argo-hermes-home-") as raw_home:
        home = Path(raw_home)
        config = {
            "model": {
                "provider": "lmstudio",
                "default": model,
                "base_url": base_url,
                "context_length": context,
            }
        }
        config_path = home / "config.yaml"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        config_path.chmod(0o600)
        env = dict(os.environ if base_env is None else base_env)
        for key in ("HERMES_INFERENCE_MODEL", "HERMES_MODEL", "OPENAI_BASE_URL", "LM_BASE_URL"):
            env.pop(key, None)
        env["HERMES_HOME"] = str(home)
        env["LM_API_KEY"] = token
        proc = command_runner(
            ["hermes", "config", "get", "model", "--json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            raise PreflightError(
                f"Hermes isolated config inspection failed: {(proc.stderr or '').strip()}"
            )
        try:
            resolved = json.loads(proc.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PreflightError("Hermes isolated config inspection returned invalid JSON") from exc
        expected = {
            "provider": "lmstudio",
            "default": model,
            "base_url": base_url,
            "context_length": context,
        }
        if not isinstance(resolved, dict) or any(resolved.get(key) != value for key, value in expected.items()):
            safe_observed = {
                key: resolved.get(key) if isinstance(resolved, dict) else None for key in expected
            }
            raise PreflightError(
                f"Hermes isolated config did not resolve requested policy: {safe_observed}"
            )
        yield IsolatedHermesEnvironment(home, env, resolved["context_length"])


@dataclass(frozen=True)
class WatchOutcome:
    status: str
    returncode: int
    reason: str | None
    requested_context: int
    observed_context: int | None
    invalidated_at: str | None
    evidence: list[dict[str, Any]]


def terminate_process_tree(process: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    """Terminate the process group created for Hermes, escalating if needed."""
    if process.poll() is not None:
        return
    try:
        process_group = os.getpgid(process.pid)
        os.killpg(process_group, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=grace_seconds)


def _invalid_watch_outcome(
    process: subprocess.Popen, reason: str, requested_context: int,
    evidence: list[dict[str, Any]], observed_context: int | None,
    terminate_tree: Callable[[subprocess.Popen], None],
) -> WatchOutcome:
    terminate_tree(process)
    timestamp = datetime.now(timezone.utc).isoformat()
    return WatchOutcome(
        "invalid_context", process.returncode if process.returncode is not None else 143,
        reason, requested_context, observed_context, timestamp, evidence,
    )


def observe_before_case(
    model: str,
    api_probe: Callable[[], dict[str, Any]],
    *,
    api_failure_tolerance: int,
    poll_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[LoadedObservation | None, list[dict[str, Any]]]:
    """Get the mandatory pre-case observation with the same bounded tolerance."""
    evidence: list[dict[str, Any]] = []
    for attempt in range(api_failure_tolerance):
        try:
            observation = observe_loaded_instance(model, api_probe())
        except Exception as exc:
            evidence.append({
                "state": "api_error",
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            if attempt + 1 < api_failure_tolerance:
                sleep(poll_seconds)
        else:
            evidence.append(observation.as_evidence())
            return observation, evidence
    return None, evidence


def watch_hermes_process(
    process: subprocess.Popen,
    *,
    model: str,
    requested_context: int,
    baseline: LoadedObservation,
    api_probe: Callable[[], dict[str, Any]],
    timeout_seconds: int,
    poll_seconds: float,
    api_failure_tolerance: int,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    terminate_tree: Callable[[subprocess.Popen], None] = terminate_process_tree,
) -> WatchOutcome:
    """Control Hermes while continuously enforcing one exact LM Studio instance."""
    started = clock()
    evidence = [baseline.as_evidence()]
    failures = 0
    try:
        while True:
            returncode = process.poll()
            if returncode is None and clock() - started >= timeout_seconds:
                terminate_tree(process)
                return WatchOutcome(
                    "timeout", process.returncode if process.returncode is not None else 124,
                    "timeout", requested_context, baseline.context_length, None, evidence,
                )
            try:
                current = observe_loaded_instance(model, api_probe())
            except Exception as exc:
                failures += 1
                evidence.append({
                    "state": "api_error",
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })
                if failures >= api_failure_tolerance:
                    return _invalid_watch_outcome(
                        process, "observation_unavailable", requested_context,
                        evidence, None, terminate_tree,
                    )
                sleep(poll_seconds)
                continue
            else:
                failures = 0
                evidence.append(current.as_evidence())
                if current.context_length != requested_context:
                    return _invalid_watch_outcome(
                        process, "context_changed", requested_context,
                        evidence, current.context_length, terminate_tree,
                    )
                if current.instance_id != baseline.instance_id:
                    return _invalid_watch_outcome(
                        process, "instance_changed", requested_context,
                        evidence, current.context_length, terminate_tree,
                    )
            if returncode is not None:
                return WatchOutcome(
                    "completed", returncode, None, requested_context,
                    current.context_length if failures == 0 else None, None, evidence,
                )
            sleep(poll_seconds)
    except BaseException:
        terminate_tree(process)
        raise


def select_cases(suite: dict[str, Any], selected: list[str] | None) -> list[dict[str, Any]]:
    cases = suite["cases"]
    if not selected:
        return cases
    by_id = {case["id"]: case for case in cases}
    unknown = sorted(set(selected) - by_id.keys())
    if unknown:
        raise ContractError(f"unknown agentic cases: {', '.join(unknown)}")
    return [by_id[case_id] for case_id in selected]


def model_artifact_identity(model: str, metadata: dict[str, Any]) -> dict[str, Any]:
    models = metadata.get("models", metadata.get("data", []))
    record = next(
        (item for item in models if item.get("key", item.get("id")) == model), None
    )
    if record is None:
        raise PreflightError(f"model {model} absent from LM Studio native metadata")
    return {
        key: record.get(key)
        for key in (
            "key", "id", "display_name", "path", "format", "architecture", "size_bytes",
            "max_context_length", "quantization", "publisher", "model_type", "compatibility_type",
        )
        if record.get(key) is not None
    } | {
        "metadata_sha256": hashlib.sha256(
            json.dumps(
                {key: value for key, value in record.items() if key != "loaded_instances"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    }


def start_keep_awake() -> subprocess.Popen | None:
    executable = shutil.which("caffeinate")
    if executable is None:
        return None
    return subprocess.Popen(
        [executable, "-ims", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_case(
    model: str, case: dict[str, Any], suite: dict[str, Any], model_output: Path,
    executor: CommandExecutor, context: int, *,
    api_probe: Callable[[], dict[str, Any]], api_root: str, token: str,
    terminate_tree: Callable[[subprocess.Popen], None] = terminate_process_tree,
) -> dict[str, Any]:
    fixture_root = ROOT / suite["fixture_root"]
    case_output = model_output / case["id"]
    case_output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="argo-agentic-") as tmp:
        workspace = Path(tmp) / "project"
        shutil.copytree(
            fixture_root,
            workspace,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache", "*.pyc"),
        )
        spec_text = (ROOT / case["spec"]).read_text(encoding="utf-8")
        prompt = (
            "You are working inside a synthetic Python project. The graded spec is embedded below; "
            "it is not a filesystem path and must not be modified.\n\n"
            f"----- BEGIN {case['id']} -----\n{spec_text}\n----- END {case['id']} -----\n\n"
            f"Edit only {case['source']} and {case['self_test']}. {case['prompt']} "
            f"When finished, run python3 -m pytest {case['self_test']} -v and fix failures."
        )
        started = datetime.now(timezone.utc)
        invalid_reason: str | None = None
        baseline, context_evidence = observe_before_case(
            model,
            api_probe,
            api_failure_tolerance=suite["policy"]["context_api_failure_tolerance"],
            poll_seconds=suite["policy"]["context_watchdog_poll_seconds"],
        )
        if baseline is None:
            watch_status = "invalid_context"
            invalid_reason = "observation_unavailable_before_case"
            final_observed = None
            hermes_exit_code = None
            invalidated_at = datetime.now(timezone.utc).isoformat()
            (case_output / "hermes.log").write_text(
                "Hermes not started: LM Studio state unavailable before case.\n",
                encoding="utf-8",
            )
        elif baseline.context_length != context:
            watch_status = "invalid_context"
            invalid_reason = "context_changed_before_case"
            final_observed = baseline.context_length
            hermes_exit_code = None
            invalidated_at = datetime.now(timezone.utc).isoformat()
            (case_output / "hermes.log").write_text(
                "Hermes not started: LM Studio context pre-case mismatch.\n", encoding="utf-8"
            )
        else:
            with isolated_hermes_environment(
                model=model, context=context, api_root=api_root, token=token,
                command_runner=executor.run,
            ) as isolated:
                with (case_output / "hermes.log").open("w", encoding="utf-8") as log_stream:
                    process = executor.start_hermes(
                        model, prompt, workspace, suite, isolated.env, log_stream
                    )
                    outcome = watch_hermes_process(
                        process,
                        model=model,
                        requested_context=context,
                        baseline=baseline,
                        api_probe=api_probe,
                        timeout_seconds=suite["policy"]["timeout_seconds"],
                        poll_seconds=suite["policy"]["context_watchdog_poll_seconds"],
                        api_failure_tolerance=suite["policy"]["context_api_failure_tolerance"],
                        terminate_tree=terminate_tree,
                    )
            watch_status = outcome.status
            invalid_reason = outcome.reason if outcome.status == "invalid_context" else None
            final_observed = outcome.observed_context
            hermes_exit_code = outcome.returncode
            invalidated_at = outcome.invalidated_at
            context_evidence = outcome.evidence
        timed_out = watch_status == "timeout"

        src_out = case_output / "src"
        src_out.mkdir()
        shutil.copy2(workspace / case["source"], src_out / Path(case["source"]).name)
        tests_out = case_output / "tests"
        tests_out.mkdir()
        shutil.copy2(workspace / case["self_test"], tests_out / Path(case["self_test"]).name)

        log_text = (case_output / "hermes.log").read_text(encoding="utf-8", errors="replace")
        model_invoked = "session_id:" in log_text
        if watch_status == "invalid_context":
            reference = None
            status = "invalid_context"
        else:
            scorer = ROOT / suite["scorer"]
            score_cmd = [
                sys.executable, str(scorer), "--eval-id", case["id"], "--src-dir", str(src_out),
                "--junit", str(case_output / "reference-junit.xml"),
                "--log", str(case_output / "reference-pytest.log"),
                "--intree-grader", str(workspace / "tests" / "test_calculator.py"),
            ]
            score = executor.run(score_cmd, capture_output=True, text=True, cwd=ROOT)
            lines = score.stdout.splitlines()
            reference = json.loads(lines[3]) if len(lines) == 4 else {
                "status": "fail", "reason": "reference scorer produced no valid verdict"
            }
            if case["id"] == "EVAL-004":
                source_lines = (workspace / case["source"]).read_text().splitlines()
                reference = {
                    "status": "pass" if not timed_out and model_invoked and len(source_lines) <= 2 else "fail",
                    "reason": "refusal trap scored by untouched synthetic stub",
                }
            status = "timeout" if timed_out else reference["status"]
        result = {
            "eval_id": case["id"],
            "model": model,
            "harness": "hermes",
            "status": status,
            "reference": reference,
            "hermes_exit_code": hermes_exit_code,
            "model_invoked": model_invoked,
            "timed_out": timed_out,
            "requested_context": context,
            "observed_context": final_observed,
            "context_evidence": context_evidence,
            "invalid_context_reason": invalid_reason,
            "invalidated_at": invalidated_at,
            "timeout_limit_s": suite["policy"]["timeout_seconds"],
            "minimum_free_memory_mb": suite["policy"]["minimum_free_memory_mb"],
            "max_turns": suite["policy"]["max_turns"],
            "toolsets": suite["policy"]["toolsets"],
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        (case_output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def run_models(
    models: list[str], suite: dict[str, Any], dependencies: dict[str, Any], context: int,
    selected: list[str] | None, api_root: str, token: str, executor: CommandExecutor,
    force_requested: bool = False,
) -> Path:
    context = resolve_context(context)
    cases = select_cases(suite, selected)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    output = RESULTS_ROOT / f"{stamp}_agentic-unified-ctx{context}"
    keep_awake = start_keep_awake()
    try:
        observed: dict[str, int] = {}
        observed_parallel: dict[str, int] = {}
        artifacts: dict[str, Any] = {}
        for model in models:
            observed_parallel[model] = executor.load_model(
                model, context, suite["policy"]["parallel_predictions"]
            )
            metadata = native_models(api_root, token)
            observed[model] = verify_observed_context(model, context, metadata)
            artifacts[model] = model_artifact_identity(model, metadata)
        manifest = build_manifest(
            models, suite, dependencies, context, [case["id"] for case in cases], output,
            observed, dry_run=False, model_artifacts=artifacts,
            observed_parallel=observed_parallel,
            force_requested=force_requested,
        )
        output.mkdir(parents=True, exist_ok=False)
        (output / "run-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for model in models:
            model_output = output / model.replace("/", "--")
            model_output.mkdir()
            for case in cases:
                executor.load_model(
                    model, context, suite["policy"]["parallel_predictions"]
                )
                verify_observed_context(model, context, native_models(api_root, token))
                if free_memory_mb() < suite["policy"]["minimum_free_memory_mb"]:
                    raise PreflightError("free memory fell below policy before Hermes start")
                result = run_case(
                    model, case, suite, model_output, executor, context,
                    api_probe=lambda: native_models(api_root, token),
                    api_root=api_root,
                    token=token,
                )
                if result["status"] == "invalid_context":
                    raise ContextDriftError(
                        f"{model} {case['id']} invalid_context: "
                        f"{result.get('invalid_context_reason')}"
                    )
    finally:
        if keep_awake is not None:
            keep_awake.terminate()
        if hasattr(keep_awake, "wait"):
            try:
                keep_awake.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        unload_runner = getattr(executor, "run", subprocess.run)
        unload_runner(["lms", "unload", "--all"], capture_output=True, text=True)
    return output
