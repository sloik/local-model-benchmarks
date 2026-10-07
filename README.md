# Local Model Benchmarks

A local-first harness for source-grounded text tasks, tool use and multi-turn coding.
It supports LM Studio's OpenAI-compatible endpoint and a Hermes coding runner, with
deterministic scoring where possible and explicit run provenance.

## Install and test

Python 3.11 or later. The test suite needs no model server, model weights or credentials.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests/ -q
```

Live evaluation needs a configured LM Studio server. The coding suite also requires
macOS, zsh, `lms`, `hermes`, `caffeinate` and a credential exported as `LM_STUDIO_TOKEN`
or `LM_API_KEY`. Model weights and credentials are not bundled. Scripts read exported
environment variables; they do not automatically load `.env`. See `.env.example`.

## Suites

| Suite | Tasks | Contract |
| --- | --- | --- |
| canonical-cat | extraction, distillation, reasoning, coding, classification, fidelity and tool use (CAT-01..07) | `suites/registry.json` and `suites/canonical/scoring.json` |
| agentic-coding | five multi-turn coding/refusal cases with held-out reference tests | `suites/agentic/suite.json` |
| compact capability | packet-based synthetic decisions and mutation sensitivity | `suites/capability/suite.json` |

CAT-01..04 use semantic judges with executable support for coding. CAT-05 exact labels,
CAT-06 source fidelity and CAT-07 tool selection/commitment have deterministic authorities.
Keep objective gates, coverage and judge identity beside any semantic average.

## Run

Inspect a coding-run manifest without invoking a model:

```sh
python scripts/run_agentic_benchmark.py --model YOUR_MODEL --dry-run
```

Run canonical inference against the configured local model:

```sh
bash scripts/run_all.sh --model YOUR_MODEL --skip-eval
```

Exercise refusal controls without model access:

```sh
python scripts/score_cat06.py --null always_refuse
python scripts/score_cat06.py --null always_answer
python scripts/score_cat06_t2.py --null always_refuse \
  --extracts-dir test_data/cat06/synthetic-report
```

Both null policies should fail paired fidelity: answering every question or refusing
all questions is insufficient. The bundled fictional report is a smoke-test source;
it does not demonstrate realistic long-context difficulty.

Coding-run defaults are exactly 131072 context, one prediction slot, 1800 seconds and
30 turns per case. The runner loads a fresh model backend for every case, verifies
effective context/slots before, during and after inference, and scores generated code
against held-out tests. Reference tests never enter the model workspace. `--dry-run`
skips live preflight and inference.

## Public corpus and legacy scores

**`public-synthetic-v1` is a new corpus.** Eight historical financial cases and seven
filing-based fidelity cases are replaced with original fictional scenarios. Two caption
excerpts are replaced with synthetic transcript stand-ins; their video metadata remains
factual. Identifiers preserve task-family continuity, not score equivalence.

[PUBLICATION.json](PUBLICATION.json) records transformations, exclusions and fixture
hashes. No issuer filings, paid analyst text, actual captions or model weights are
distributed. Original code and synthetic content are MIT-licensed.

[LEADERBOARD.md](LEADERBOARD.md), `leaderboard.csv` and numeric summaries in `results/`
measure the **earlier legacy corpus**, not the revised public corpus. They support
model-identity, provenance and scorer regressions. Raw source documents, model outputs,
logs and judge rationales are omitted; this is not a full historical reproducibility
archive. Four reviewed tool-only traces retain factual video metadata or fictional
financial output for commitment-scoring regressions.

## Development

```sh
python -m pytest tests/ -q
python scripts/audit_public_release.py
```

Tests must leave tracked files unchanged. The repository starts with clean public
history and excludes local workflow/context files. See [SPEC.md](SPEC.md) for scoring
and provenance requirements. Dependency updates start from `requirements.in` and
must regenerate `requirements.txt` and pass the full suite.

## License

[MIT](LICENSE) for original code and synthetic fixtures. Factual video metadata is
not permission to download or redistribute third-party media or captions.
