"""SPEC-003-014 unified suite contract and runner gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import agentic_suite as sut  # noqa: E402
import run_agentic_benchmark as cli  # noqa: E402


def test_cli_offline_dry_run_never_probes_external_dependencies(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_agentic_benchmark.py", "--model", "example", "--dry-run"])
    monkeypatch.setattr(cli, "preflight", lambda *a, **k: pytest.fail("offline dry-run probed runtime"))
    monkeypatch.setattr(cli, "run_models", lambda *a, **k: pytest.fail("offline dry-run invoked a model"))
    cli.main()


def registry():
    return sut.load_json(sut.REGISTRY_PATH)


def suite():
    return sut.load_json(sut.AGENTIC_SUITE_PATH)


class _CompletedHermes:
    pid = 434343
    returncode = 0

    def poll(self):
        return 0


class _CaseExecutor:
    def __init__(self, session_line="session_id: abc\n"):
        self.session_line = session_line

    def run(self, argv, **kwargs):
        if argv[:4] == ["hermes", "config", "get", "model"]:
            config = json.loads(
                (Path(kwargs["env"]["HERMES_HOME"]) / "config.yaml").read_text()
            )
            return subprocess.CompletedProcess(argv, 0, json.dumps(config["model"]), "")
        payload = {"status": "not_applicable", "reason": "no reference suite"}
        return subprocess.CompletedProcess(
            argv, 0, "not_applicable\n0\n0\n" + json.dumps(payload), ""
        )

    def start_hermes(self, model, prompt, cwd, suite_value, env, log_stream):
        assert not (cwd / "reference-tests").exists()
        assert "suites/agentic/reference-tests" not in prompt
        log_stream.write(self.session_line)
        log_stream.flush()
        return _CompletedHermes()


def test_registry_covers_canonical_and_agentic_with_complete_contract():
    value = registry()
    sut.validate_registry(value)
    assert {item["family"] for item in value["suites"]} == {"canonical", "agentic"}
    assert {item["id"] for item in value["suites"]} == {"canonical-cat", "agentic-coding"}


@pytest.mark.parametrize("missing", ["runner", "scorer", "cases", "prerequisites", "canonical_result_root"])
def test_registry_rejects_required_suite_fields(missing):
    value = registry()
    del value["suites"][0][missing]
    with pytest.raises(sut.ContractError, match="missing"):
        sut.validate_registry(value)


def test_canonical_scoring_routes_are_truthful_for_deterministic_categories():
    mapping = sut.load_json(ROOT / "suites" / "canonical" / "scoring.json")["routes"]
    assert mapping["CAT-06"]["authoritative"] == "scripts/score_cat06.py"
    assert mapping["CAT-07"]["authoritative"] == "scripts/score_cat07.py"
    assert "semantic" not in mapping["CAT-07"]


def test_agentic_contract_has_exact_context_policy_and_synthetic_sources():
    value = suite()
    sut.validate_agentic_suite(value)
    assert value["policy"]["default_context"] == 131072
    assert value["policy"]["minimum_context"] == 131072
    assert value["policy"]["case_model_isolation"] == "unload_all_then_load_exact_context"
    assert value["policy"]["parallel_predictions"] == 1
    assert {case["source_basis"] for case in value["cases"]} == {"synthetic"}
    assert value["validation"]["test_command"] == "python3 -m pytest tests/"
    assert value["validation"]["manifest_schema"] == value["result_schema"]


def test_agentic_contract_rejects_missing_per_case_model_isolation():
    value = suite()
    value["policy"].pop("case_model_isolation", None)
    with pytest.raises(sut.ContractError, match="case_model_isolation"):
        sut.validate_agentic_suite(value)


@pytest.mark.parametrize("parallel", [None, 0, 2, 4])
def test_agentic_contract_requires_exactly_one_parallel_prediction_slot(parallel):
    value = suite()
    if parallel is None:
        value["policy"].pop("parallel_predictions", None)
    else:
        value["policy"]["parallel_predictions"] = parallel
    with pytest.raises(sut.ContractError, match="parallel_predictions"):
        sut.validate_agentic_suite(value)


def test_agentic_contract_rejects_missing_owner_validation_metadata():
    value = suite()
    del value["validation"]["test_files"]
    with pytest.raises(sut.ContractError, match="validation contract missing"):
        sut.validate_agentic_suite(value)


def test_agentic_contract_rejects_non_synthetic_case():
    value = suite()
    value["cases"][0]["source_basis"] = "unknown"
    with pytest.raises(sut.ContractError, match="source_basis"):
        sut.validate_agentic_suite(value)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("timeout_seconds", 0, "timeout_seconds"),
        ("timeout_seconds", 86401, "timeout_seconds"),
        ("minimum_free_memory_mb", 0, "minimum_free_memory_mb"),
        ("minimum_free_memory_mb", 1048577, "minimum_free_memory_mb"),
    ],
)
def test_agentic_contract_rejects_out_of_range_runtime_policy(key, value, match):
    contract = suite()
    contract["policy"][key] = value
    with pytest.raises(sut.ContractError, match=match):
        sut.validate_agentic_suite(contract)


def test_context_override_is_wired_to_load_and_exact_observation():
    assert sut.load_command("model-a", 188928) == [
        "lms", "load", "model-a", "--context-length", "188928",
        "--parallel", "1", "--yes"
    ]
    metadata = {
        "models": [{
            "key": "model-a",
            "loaded_instances": [{"config": {"context_length": 188928}}],
        }]
    }
    assert sut.verify_observed_context("model-a", 188928, metadata) == 188928
    with pytest.raises(sut.PreflightError, match="does not match requested"):
        sut.verify_observed_context("model-a", 131072, metadata)
    with pytest.raises(sut.ContractError, match="below required minimum"):
        sut.load_command("model-a", 8192)


def test_executor_uses_the_same_context_in_actual_lms_load():
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv == ["lms", "ps", "--json"]:
            return subprocess.CompletedProcess(
                argv, 0,
                json.dumps([{"modelKey": "model-a", "contextLength": 131072, "parallel": 1}]),
                "",
            )
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    assert sut.CommandExecutor(fake_run).load_model("model-a", 131072) == 1
    assert calls == [
        ["lms", "unload", "--all"],
        ["lms", "load", "model-a", "--context-length", "131072", "--parallel", "1", "--yes"],
        ["lms", "ps", "--json"],
    ]


@pytest.mark.parametrize(
    "payload,match",
    [
        ([], "absent"),
        ([{"modelKey": "model-a", "parallel": 4}], "does not match requested"),
        ([{"modelKey": "model-a"}], "no integer observed parallel"),
        ([{"modelKey": "model-a", "parallel": 1}, {"identifier": "model-a", "parallel": 1}],
         "expected exactly one"),
    ],
)
def test_parallel_process_observation_fails_closed(payload, match):
    with pytest.raises(sut.PreflightError, match=match):
        sut.verify_observed_parallel("model-a", 1, payload)


def test_executor_fails_closed_when_prior_backend_cannot_be_unloaded():
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, "", "backend still active")

    with pytest.raises(sut.PreflightError, match="lms unload failed"):
        sut.CommandExecutor(fake_run).load_model("model-a", 131072)
    assert calls == [["lms", "unload", "--all"]]


def test_preflight_probes_read_only_api_and_never_exposes_credential(monkeypatch):
    monkeypatch.setattr(sut.sys, "platform", "darwin")
    monkeypatch.setattr(sut, "free_memory_mb", lambda: 64000)
    seen = {}

    def fake_which(name):
        return f"/fake/{name}"

    def fake_version(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, f"{Path(argv[0]).name} 1.0\n", "")

    def fake_api(root, token):
        seen.update(root=root, token=token)
        return {"models": []}

    result = sut.preflight(
        suite(), env={"LM_STUDIO_TOKEN": "secret-value"}, which=fake_which,
        version_runner=fake_version, api_probe=fake_api,
    )
    assert seen == {"root": "http://127.0.0.1:1234", "token": "secret-value"}
    assert result["lm_studio_api"]["status"] == "reachable"
    assert result["lm_studio_api"]["endpoint"] == "http://127.0.0.1:1234"
    assert result["credential_source"] == "LM_STUDIO_TOKEN"
    assert "secret-value" not in json.dumps(result)


@pytest.mark.parametrize("missing", ["zsh", "python3", "lms", "hermes", "caffeinate"])
def test_preflight_fails_for_each_missing_required_command(monkeypatch, missing):
    monkeypatch.setattr(sut.sys, "platform", "darwin")
    monkeypatch.setattr(sut, "free_memory_mb", lambda: 64000)
    with pytest.raises(sut.PreflightError, match=missing):
        sut.preflight(
            suite(), env={"LM_STUDIO_TOKEN": "x"},
            which=lambda name: None if name == missing else f"/fake/{name}",
            api_probe=lambda root, token: {"models": []},
        )


def test_preflight_fails_for_missing_credential_memory_and_api(monkeypatch, tmp_path):
    monkeypatch.setattr(sut.sys, "platform", "darwin")
    monkeypatch.setattr(sut, "free_memory_mb", lambda: 64000)
    fake_which = lambda name: f"/fake/{name}"
    fake_version = lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "1", "")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(sut.PreflightError, match="credential"):
        sut.preflight(suite(), env={}, which=fake_which, version_runner=fake_version)
    monkeypatch.setattr(sut, "free_memory_mb", lambda: 10)
    with pytest.raises(sut.PreflightError, match="free memory"):
        sut.preflight(
            suite(), env={"LM_STUDIO_TOKEN": "x"}, which=fake_which,
            version_runner=fake_version,
        )
    monkeypatch.setattr(sut, "free_memory_mb", lambda: 64000)
    with pytest.raises(sut.PreflightError, match="API is unreachable"):
        sut.preflight(
            suite(), env={"LM_STUDIO_TOKEN": "x"}, which=fake_which,
            version_runner=fake_version,
            api_probe=lambda root, token: (_ for _ in ()).throw(OSError("offline")),
        )


def test_ambient_hermes_env_is_not_used_for_benchmark_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    env_file = tmp_path / ".hermes" / ".env"
    env_file.parent.mkdir()
    env_file.write_text("LM_API_KEY=from-hermes-file\n")
    assert sut.resolve_credential({}) is None


def test_dry_run_manifest_is_valid_and_writes_nothing(tmp_path):
    output = tmp_path / "must-not-exist"
    dependencies = {"hermes": "hermes 1", "lm_studio_api": "reachable"}
    manifest = sut.build_manifest(
        ["model-a", "model-b"], suite(), dependencies, 131072,
        ["EVAL-001"], output, {"model-a": None, "model-b": None}, dry_run=True,
    )
    sut.validate_manifest(manifest)
    assert manifest["observation_state"] == "not-loaded-dry-run"
    assert manifest["requested_context"] == 131072
    assert manifest["requested_parallel_predictions"] == 1
    assert manifest["observed_parallel_predictions"] == {"model-a": None, "model-b": None}
    assert not output.exists()


def test_manifest_rejects_parallel_prediction_mismatch(tmp_path):
    manifest = sut.build_manifest(
        ["model-a"], suite(), {"hermes": "hermes 1"}, 131072,
        ["EVAL-001"], tmp_path / "output", {"model-a": 131072}, dry_run=False,
        observed_parallel={"model-a": 1},
    )
    manifest["observed_parallel_predictions"]["model-a"] = 4
    with pytest.raises(sut.ContractError, match="observed parallel mismatch"):
        sut.validate_manifest(manifest)


def test_cli_dry_run_never_dispatches_live_runner(monkeypatch, capsys):
    monkeypatch.setattr(cli, "preflight", lambda value: {"hermes": "1", "lm_studio_api": {"status": "reachable"}})
    monkeypatch.setattr(
        cli, "run_models",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live runner called")),
    )
    monkeypatch.setattr(sys, "argv", ["run_agentic_benchmark.py", "--model", "model-a", "--dry-run"])
    cli.main()
    output = capsys.readouterr().out
    assert "not-loaded-dry-run" in output
    assert "DRY-RUN-NOT-WRITTEN" in output


def test_cli_accepts_grouped_and_repeated_models_with_runtime_overrides(monkeypatch, capsys):
    captured = {}

    def fake_preflight(value):
        captured["preflight_policy"] = dict(value["policy"])
        return {"hermes": "1", "lm_studio_api": {"status": "reachable"}}

    monkeypatch.setattr(cli, "preflight", fake_preflight)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agentic_benchmark.py", "--model", "model-a", "model-b",
            "--model", "model-c", "--timeout", "900", "--min-free-mem", "2048",
        ],
    )
    # Verify effective preflight overrides on the live path with inference stubbed.
    monkeypatch.setattr(cli, "resolve_credential", lambda env: ("test-token", "fixture"))
    monkeypatch.setattr(cli, "run_models", lambda *a, **k: Path("NOT-WRITTEN"))
    cli.main()
    capsys.readouterr()
    # The offline manifest must preserve the same policy without probing again.
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--dry-run"])
    monkeypatch.setattr(cli, "preflight", lambda *a, **k: pytest.fail("offline probe"))
    monkeypatch.setattr(cli, "run_models", lambda *a, **k: pytest.fail("offline inference"))
    cli.main()
    output = capsys.readouterr().out
    decoder = json.JSONDecoder()
    _preflight, offset = decoder.raw_decode(output)
    manifest, _end = decoder.raw_decode(output, output.index("{", offset))
    assert captured["preflight_policy"]["timeout_seconds"] == 900
    assert captured["preflight_policy"]["minimum_free_memory_mb"] == 2048
    assert manifest["models"] == ["model-a", "model-b", "model-c"]
    assert manifest["timeout_seconds"] == 900
    assert manifest["minimum_free_memory_mb"] == 2048


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--timeout", "0"), ("--timeout", "86401"),
        ("--min-free-mem", "0"), ("--min-free-mem", "1048577"),
    ],
)
def test_cli_rejects_non_positive_and_out_of_range_runtime_overrides(
    monkeypatch, flag, value
):
    monkeypatch.setattr(
        sys, "argv",
        ["run_agentic_benchmark.py", "--model", "model-a", flag, value, "--dry-run"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_force_is_recorded_as_non_overwriting_deprecated_compatibility(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "preflight",
        lambda value: {"hermes": "1", "lm_studio_api": {"status": "reachable"}},
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_agentic_benchmark.py", "--model", "model-a", "--force", "--dry-run"],
    )
    cli.main()
    output = capsys.readouterr().out
    assert "--force is compatibility-only and deprecated" in output
    assert "never overwrite or skip existing evidence" in output
    assert '"requested": true' in output
    assert '"overwrite_existing_evidence": false' in output


def test_runtime_overrides_reach_actual_hermes_timeout_and_case_result(tmp_path):
    value = suite()
    value["policy"]["timeout_seconds"] = 777
    value["policy"]["minimum_free_memory_mb"] = 3456
    case = next(item for item in value["cases"] if item["id"] == "EVAL-004")
    model_output = tmp_path / "model-output"
    model_output.mkdir()
    result = sut.run_case(
        "model-a", case, value, model_output, _CaseExecutor(), 131072,
        api_probe=lambda: _native_state(),
        api_root="http://127.0.0.1:1234",
        token="test-token",
    )
    assert result["timeout_limit_s"] == 777
    assert result["minimum_free_memory_mb"] == 3456


def test_minimum_memory_override_is_enforced_by_actual_preflight(monkeypatch):
    value = suite()
    value["policy"]["minimum_free_memory_mb"] = 2048
    monkeypatch.setattr(sut.sys, "platform", "darwin")
    monkeypatch.setattr(sut, "free_memory_mb", lambda: 2047)
    with pytest.raises(sut.PreflightError, match="below required 2048MB"):
        sut.preflight(
            value,
            env={"LM_STUDIO_TOKEN": "x"},
            which=lambda name: f"/fake/{name}",
            version_runner=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "1", ""),
            api_probe=lambda root, token: {"models": []},
        )


def test_observed_mismatch_aborts_before_hermes_or_result_write(monkeypatch, tmp_path):
    class FakeExecutor:
        hermes_called = False

        def run(self, argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, "", "")

        def load_model(self, model, context, parallel_predictions):
            return parallel_predictions

        def start_hermes(self, *args, **kwargs):
            self.hermes_called = True
            raise AssertionError("Hermes must not start")

    executor = FakeExecutor()
    monkeypatch.setattr(sut, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        sut, "native_models",
        lambda root, token: {"models": [{
            "key": "model-a", "loaded_instances": [{"config": {"context_length": 8192}}]
        }]},
    )
    with pytest.raises(sut.PreflightError, match="does not match requested"):
        sut.run_models(
            ["model-a"], suite(), {"hermes": "1"}, 131072, ["EVAL-001"],
            "http://127.0.0.1:1234", "token", executor,
        )
    assert executor.hermes_called is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("session_line, expected", [("", "fail"), ("session_id: abc\n", "pass")])
def test_eval004_requires_model_invocation_and_retains_legacy_evidence(tmp_path, session_line, expected):
    case = next(item for item in suite()["cases"] if item["id"] == "EVAL-004")
    model_output = tmp_path / "model"
    model_output.mkdir()
    result = sut.run_case(
        "model-a", case, suite(), model_output, _CaseExecutor(session_line), 131072,
        api_probe=lambda: _native_state(),
        api_root="http://127.0.0.1:1234",
        token="test-token",
    )
    case_output = model_output / "EVAL-004"
    assert result["status"] == expected
    assert result["model_invoked"] is bool(session_line)
    assert (case_output / "src" / "validator.py").is_file()
    assert (case_output / "tests" / "test_validator.py").is_file()
    assert (case_output / "hermes.log").is_file()
    assert result["timeout_limit_s"] == 1800
    assert result["max_turns"] == 30
    assert result["requested_context"] == result["observed_context"] == 131072


def test_committed_source_provenance_hashes_match_migrated_bytes():
    provenance = sut.load_json(ROOT / "suites" / "agentic" / "PROVENANCE.json")
    assert provenance["source_commit"] == "1c9dc4794577bba47fd7badffb48c53a32f9c2d8"
    assert provenance["extraction"] == "git object database (not live working tree)"
    for row in provenance["files"]:
        digest = hashlib.sha256((ROOT / row["target_path"]).read_bytes()).hexdigest()
        assert digest == row["sha256"]
        assert row["source_basis"] == "synthetic"


def test_hidden_reference_contract_and_eval005_fixture_guards():
    fixture_root = ROOT / suite()["fixture_root"]
    reference_root = ROOT / suite()["reference_root"]
    assert reference_root not in fixture_root.parents
    assert fixture_root not in reference_root.parents
    assert not any("reference" in path.name for path in fixture_root.rglob("*"))
    calculator = (fixture_root / "src" / "calculator.py").read_text()
    assert "class Calculator" not in calculator
    assert "def calculate" in calculator
    assert (
        fixture_root / "tests" / "test_calculator.py"
    ).read_bytes() == (
        reference_root / "eval005_grader" / "test_calculator.py"
    ).read_bytes()
    eval004 = (ROOT / "suites" / "agentic" / "cases" / "EVAL-004-ambiguous-requirements.md").read_text()
    assert "## Scoring Notes" not in eval004
    assert "Validate user profiles" in eval004


def test_migrated_reference_scorer_executes_from_benchmark_owned_paths(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "fizzbuzz.py").write_text(
        "def fizzbuzz(n, rules):\n"
        "    return [''.join(label for divisor, label in rules if i % divisor == 0) or str(i) "
        "for i in range(1, n + 1)]\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "suites" / "agentic" / "scoring" / "score_reference.py"),
            "--eval-id", "EVAL-001", "--src-dir", str(src),
            "--junit", str(tmp_path / "junit.xml"),
            "--log", str(tmp_path / "pytest.log"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines = proc.stdout.splitlines()
    assert proc.returncode == 0
    assert lines[:3] == ["pass", "7", "7"]
    assert json.loads(lines[3])["suite"] == "reference-tests/test_eval001_fizzbuzz.py"


def _native_state(context=131072, instance_id="instance-a"):
    return {
        "models": [{
            "key": "model-a",
            "loaded_instances": [{
                "id": instance_id,
                "config": {"context_length": context},
            }],
        }]
    }


def test_isolated_hermes_config_resolves_131072_not_ambient_98000_and_cleans_up(tmp_path):
    ambient = tmp_path / "ambient" / "config.yaml"
    ambient.parent.mkdir()
    ambient.write_text('{"model":{"context_length":98000}}\n', encoding="utf-8")
    observed = {}

    def inspect(argv, **kwargs):
        assert argv == ["hermes", "config", "get", "model", "--json"]
        private_home = Path(kwargs["env"]["HERMES_HOME"])
        observed["home"] = private_home
        observed["config"] = (private_home / "config.yaml").read_text(encoding="utf-8")
        return subprocess.CompletedProcess(
            argv, 0,
            json.dumps({
                "provider": "lmstudio",
                "default": "model-a",
                "base_url": "http://127.0.0.1:1234/v1",
                "context_length": 131072,
            }),
            "",
        )

    with sut.isolated_hermes_environment(
        model="model-a",
        context=131072,
        api_root="http://127.0.0.1:1234",
        token="top-secret",
        command_runner=inspect,
        base_env={"HOME": str(tmp_path / "ambient")},
    ) as isolated:
        assert isolated.resolved_context == 131072
        assert isolated.env["HERMES_HOME"] == str(observed["home"])
        assert isolated.env["LM_API_KEY"] == "top-secret"
        assert "top-secret" not in observed["config"]
        assert '"context_length": 131072' in observed["config"]
        assert '"base_url": "http://127.0.0.1:1234/v1"' in observed["config"]
        assert observed["home"].is_dir()

    assert not observed["home"].exists()
    assert json.loads(ambient.read_text())["model"]["context_length"] == 98000


def test_hermes_launch_is_controllable_and_secret_stays_out_of_argv(tmp_path):
    captured = {}

    def fake_popen(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return _FakeHermesProcess(polls=(0,))

    log_path = tmp_path / "hermes.log"
    child_env = {"HERMES_HOME": str(tmp_path / "private"), "LM_API_KEY": "secret"}
    with log_path.open("w", encoding="utf-8") as log_stream:
        sut.CommandExecutor(popen=fake_popen).start_hermes(
            "model-a", "prompt", tmp_path, suite(), child_env, log_stream
        )

    assert captured["argv"][:4] == ["hermes", "chat", "-q", "prompt"]
    assert "secret" not in " ".join(captured["argv"])
    assert captured["kwargs"]["env"] is child_env
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT


class _FakeHermesProcess:
    def __init__(self, polls=(None, None, None)):
        self.pid = 424242
        self._polls = iter(polls)
        self.returncode = None

    def poll(self):
        try:
            value = next(self._polls)
        except StopIteration:
            value = self.returncode
        if value is not None:
            self.returncode = value
        return value

    def wait(self, timeout=None):
        self.returncode = 143 if self.returncode is None else self.returncode
        return self.returncode


def test_watchdog_invalidates_131072_to_174974_and_terminates_process_tree():
    payloads = iter([_native_state(131072), _native_state(174974)])
    process = _FakeHermesProcess()
    terminated = []
    sleeps = []

    outcome = sut.watch_hermes_process(
        process,
        model="model-a",
        requested_context=131072,
        baseline=sut.observe_loaded_instance("model-a", _native_state(131072), now=lambda: 10.0),
        api_probe=lambda: next(payloads),
        timeout_seconds=1800,
        poll_seconds=0.01,
        api_failure_tolerance=1,
        clock=iter([10.0, 10.1, 10.2, 10.3]).__next__,
        sleep=lambda seconds: sleeps.append(seconds),
        terminate_tree=lambda proc: terminated.append(proc.pid),
    )

    assert outcome.status == "invalid_context"
    assert outcome.reason == "context_changed"
    assert outcome.requested_context == 131072
    assert outcome.observed_context == 174974
    assert outcome.invalidated_at is not None
    assert terminated == [424242]
    assert outcome.evidence[-1]["context_length"] == 174974
    assert sleeps


def test_watchdog_invalidates_instance_replacement_at_same_context():
    process = _FakeHermesProcess()
    outcome = sut.watch_hermes_process(
        process,
        model="model-a",
        requested_context=131072,
        baseline=sut.observe_loaded_instance("model-a", _native_state(), now=lambda: 10.0),
        api_probe=lambda: _native_state(131072, "instance-b"),
        timeout_seconds=1800,
        poll_seconds=0.01,
        api_failure_tolerance=1,
        clock=iter([10.0, 10.1, 10.2]).__next__,
        sleep=lambda _seconds: None,
        terminate_tree=lambda _proc: None,
    )
    assert outcome.status == "invalid_context"
    assert outcome.reason == "instance_changed"


def test_watchdog_api_observation_failure_is_bounded_and_terminates_tree():
    process = _FakeHermesProcess()
    terminated = []

    def unavailable():
        raise OSError("native API unavailable")

    outcome = sut.watch_hermes_process(
        process,
        model="model-a",
        requested_context=131072,
        baseline=sut.observe_loaded_instance("model-a", _native_state(), now=lambda: 10.0),
        api_probe=unavailable,
        timeout_seconds=1800,
        poll_seconds=0.01,
        api_failure_tolerance=2,
        clock=iter([10.0, 10.1, 10.2, 10.3, 10.4]).__next__,
        sleep=lambda _seconds: None,
        terminate_tree=lambda proc: terminated.append(proc.pid),
    )
    assert outcome.status == "invalid_context"
    assert outcome.reason == "observation_unavailable"
    assert len([row for row in outcome.evidence if row["state"] == "api_error"]) == 2
    assert terminated == [424242]


def test_pre_case_api_failure_is_bounded_and_never_returns_an_observation():
    sleeps = []
    observation, evidence = sut.observe_before_case(
        "model-a",
        lambda: (_ for _ in ()).throw(OSError("offline")),
        api_failure_tolerance=2,
        poll_seconds=0.01,
        sleep=lambda seconds: sleeps.append(seconds),
    )
    assert observation is None
    assert len(evidence) == 2
    assert all(row["state"] == "api_error" for row in evidence)
    assert sleeps == [0.01]


def test_watchdog_requires_exact_api_observation_after_process_completion():
    process = _FakeHermesProcess(polls=(None, 0))
    payloads = iter([_native_state(), OSError("post-completion miss"), OSError("still absent")])
    terminated = []

    def probe():
        value = next(payloads)
        if isinstance(value, Exception):
            raise value
        return value

    outcome = sut.watch_hermes_process(
        process,
        model="model-a",
        requested_context=131072,
        baseline=sut.observe_loaded_instance("model-a", _native_state(), now=lambda: 10.0),
        api_probe=probe,
        timeout_seconds=1800,
        poll_seconds=0.01,
        api_failure_tolerance=2,
        clock=iter([10.0, 10.1]).__next__,
        sleep=lambda _seconds: None,
        terminate_tree=lambda proc: terminated.append(proc.pid),
    )
    assert outcome.status == "invalid_context"
    assert outcome.reason == "observation_unavailable"
    assert terminated == [424242]


def test_watchdog_terminates_process_tree_on_interrupt():
    process = _FakeHermesProcess()
    terminated = []

    with pytest.raises(KeyboardInterrupt):
        sut.watch_hermes_process(
            process,
            model="model-a",
            requested_context=131072,
            baseline=sut.observe_loaded_instance("model-a", _native_state(), now=lambda: 10.0),
            api_probe=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            timeout_seconds=1800,
            poll_seconds=0.01,
            api_failure_tolerance=1,
            clock=lambda: 10.1,
            sleep=lambda _seconds: None,
            terminate_tree=lambda proc: terminated.append(proc.pid),
        )
    assert terminated == [424242]


def test_invalid_context_aborts_model_before_later_case(monkeypatch, tmp_path):
    value = suite()
    selected = ["EVAL-001", "EVAL-002"]
    started = []
    loaded = []

    awake = {"active": False}

    class FakeExecutor:
        def load_model(self, model, context, parallel_predictions):
            assert awake["active"] is True
            loaded.append((model, context))
            return parallel_predictions

        def run(self, argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, "", "")

    def fake_case(model, case, *args, **kwargs):
        assert awake["active"] is True
        started.append(case["id"])
        return {"status": "invalid_context", "eval_id": case["id"]}

    class KeepAwake:
        def terminate(self):
            awake["active"] = False

    def fake_keep_awake():
        awake["active"] = True
        return KeepAwake()

    monkeypatch.setattr(sut, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(sut, "start_keep_awake", fake_keep_awake)
    monkeypatch.setattr(sut, "native_models", lambda root, token: _native_state())
    monkeypatch.setattr(sut, "run_case", fake_case)

    with pytest.raises(sut.ContextDriftError, match="EVAL-001"):
        sut.run_models(
            ["model-a"], value, {"hermes": "1"}, 131072, selected,
            "http://127.0.0.1:1234", "token", FakeExecutor(),
        )
    assert started == ["EVAL-001"]
    # One discovery load and one fresh EVAL-001 load; EVAL-002 is never loaded or run.
    assert loaded == [("model-a", 131072), ("model-a", 131072)]
    assert awake["active"] is False


def test_every_same_model_case_gets_fresh_load_and_exact_verification_before_run(
    monkeypatch, tmp_path,
):
    events = []
    inference_started = []

    class FakeExecutor:
        def load_model(self, model, context, parallel_predictions):
            events.append(("load", model, context, parallel_predictions))
            return parallel_predictions

        def start_hermes(self, *args, **kwargs):
            inference_started.append(True)
            raise AssertionError("unit test must not start inference")

        def run(self, argv, **kwargs):
            events.append(("cleanup", tuple(argv)))
            return subprocess.CompletedProcess(argv, 0, "", "")

    class KeepAwake:
        def terminate(self):
            events.append(("keep_awake", "terminated"))

    def fake_native_models(_root, _token):
        events.append(("verify", "model-a", 131072))
        return _native_state()

    def fake_case(model, case, *args, **kwargs):
        events.append(("run_case", model, case["id"]))
        return {"status": "pass", "eval_id": case["id"]}

    monkeypatch.setattr(sut, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(sut, "start_keep_awake", lambda: KeepAwake())
    monkeypatch.setattr(sut, "native_models", fake_native_models)
    monkeypatch.setattr(sut, "run_case", fake_case)

    sut.run_models(
        ["model-a"], suite(), {"hermes": "1"}, 131072,
        ["EVAL-001", "EVAL-002"], "http://127.0.0.1:1234", "token", FakeExecutor(),
    )

    assert events[:7] == [
        ("load", "model-a", 131072, 1),
        ("verify", "model-a", 131072),
        ("load", "model-a", 131072, 1),
        ("verify", "model-a", 131072),
        ("run_case", "model-a", "EVAL-001"),
        ("load", "model-a", 131072, 1),
        ("verify", "model-a", 131072),
    ]
    assert events[7] == ("run_case", "model-a", "EVAL-002")
    assert inference_started == []


def test_keep_awake_is_optional_without_caffeinate(monkeypatch):
    monkeypatch.setattr(sut.shutil, "which", lambda name: None)
    monkeypatch.setattr(sut.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected spawn"))
    assert sut.start_keep_awake() is None


def test_keep_awake_uses_discovered_executable(monkeypatch):
    captured = {}
    sentinel = object()
    monkeypatch.setattr(sut.shutil, "which", lambda name: "/usr/bin/caffeinate")
    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        return sentinel
    monkeypatch.setattr(sut.subprocess, "Popen", fake_popen)
    assert sut.start_keep_awake() is sentinel
    assert captured["argv"] == ["/usr/bin/caffeinate", "-ims", "-w", str(sut.os.getpid())]
