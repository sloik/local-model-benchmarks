# Public benchmark contract — public-synthetic-v1

The public corpus is original synthetic material. It preserves task families and case
identifiers, but it is a different corpus from the private historical evaluations.
Do not compare its scores directly with the retained legacy aggregate snapshot.

| Suite | Cases | Authority |
| --- | --- | --- |
| canonical-cat | CAT-01 extraction, CAT-02 distillation, CAT-03 reasoning, CAT-04 coding, CAT-05 classification, CAT-06 fidelity, CAT-07 tool use | suites/canonical/scoring.json |
| agentic-coding | EVAL-001..005 | suites/agentic/suite.json and held-out reference scorer |
| compact capability | 480 synthetic cases in 16 families | suites/capability/suite.json |

Semantic judge scores require a recorded judge identity, every prompt-level rationale,
and source-grounded adjudication of disagreements. Deterministic and executable checks
outrank judge opinion. Report exact model artifact, quantization, context, serving runtime,
sampling settings, output budgets, timeouts, turn limits and selected cases with results.
Never silently substitute a prefix-matched model or infer missing historical provenance.

CAT-06 pairs supported-answer twins with unsupported probes. An always-answer or
always-refuse policy fails the pair. The long-context leg accepts an explicitly supplied
source directory; the bundled fictional report is a smoke-test source, not a claim
of realistic long-context difficulty.

CAT-07 uses frozen local fixtures; no network lookup occurs during fixture scoring.
Two transcript fixtures are synthetic stand-ins despite retaining real video metadata.
Gold fingerprints reflect those authored fixture bytes. The tool shim is intended to
test selection, arguments, source fidelity and final-answer commitment.

Agentic coding uses isolated pristine workspaces and held-out tests. The model never
receives the reference tests. Local runner defaults are 131072 context, one prediction
slot, 1800 seconds and 30 turns per case; preflight verifies effective settings before
inference. Timeout, refusal, no-invocation and fixture-reuse outcomes remain distinct.

The historical results tree includes metadata, numeric summary aggregates and four
reviewed tool-only regression traces. It excludes all other raw model outputs, logs,
source documents and judge rationales. It supports regression tests but cannot serve
as a full reproducibility archive. All omissions and corpus transformations are
recorded in PUBLICATION.json.

## General capability suite

The general capability suite uses original synthetic state packets, not trivia.
Its contract (`suites/capability/suite.json`) defines paired mutation-sensitive cases,
strict output schema, abstention, counterfactual controls and deterministic scoring.
Keep this specialist capability map separate from generative aggregate scores.
