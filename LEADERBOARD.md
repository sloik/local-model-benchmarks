# Historical aggregate score snapshot

These are legacy-corpus aggregates, not measurements of public-synthetic-v1.
The serving/status columns describe a historical inventory snapshot, not the reader's machine.
Raw evidence and private run narratives are not part of this distribution.

## 2. Full model leaderboard (all tested, with metadata)

Context = model max (served at 64K for the agentic bar). Quality = latest CAT overall (/5).
Agentic = **reference-scored code correctness /3** (EVAL-001/002/003 vs held-out suites; EVAL-004/005
excluded as free passes). `—` = never run on the agentic bar; `N/A` = can't run.

| model | params | arch | quant | max ctx | disk | tool-trained | vision | serving | quality (CAT) | aider code | Hermes code | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-4-31b | 31B | gemma4 | Q4_K_M | 262144 | 19.9 GB | ✓ | ✓ | LM Studio | 4.10 | **3/3** | **3/3** | **fleet** |
| qwen3.8-27b | 27B | qwen3_5 | 4bit | 262144 | 15.0 GB | ✓ | ✓ | LM Studio | 4.70 | — | — | **fleet** |
| qwen/qwen3.6-27b | 27B | qwen3_5 | 5bit | 262144 | 18.1 GB | ✓ | ✓ | LM Studio | 4.42 | — | — | **fleet** |
| muse-glimmer-30b | 28B | muse-glimmer | kquant | 131072 | 15.6 GB | ✓ | ✗ | LM Studio | 4.72 | — | — | **fleet** |
| Ornith-1.0-35B | 35B MoE | qwen35moe | Q6_K | 262144 | 28.5 GB | ✓ | ✗ | LM Studio | 4.22 | **2/3** | **3/3** | deleted |
| Ornith-1.0-9B | 9B | qwen35 | Q8_0 | 262144 | 9.5 GB | ✓ | ✗ | LM Studio | 4.10 | 1/3 | 2/3 | deleted |
| Mistral-small-3.2 | 24B | mistral3 | 4bit | 131072 | 13.5 GB | ✗ | ✓ | LM Studio | 4.00 | 1/3 | 1/3 | deleted |
| Apple FM (system) | ~3B | Apple FM | — | ~4096 | on-device | ✓ (guided) | ✗ | `fm serve` (native) | 3.60 | N/A | N/A | **fleet (native)** |
| qwen2.5-vl-72b-instruct | 72B | qwen2-vl | — | ~32K | (deleted) | ✗ | ✓ | LM Studio | 3.96 | 1/3 (hist.) | — | deleted |
| qwen2.5-coder-32b | 32B | qwen2 | — | ~131K | (deleted) | ✗ | ✗ | LM Studio | 3.72 | — | — | deleted |
| deepseek-r1-distill-qwen-32b | 32B | qwen2 | — | ~131K | (deleted) | ✗ | ✗ | LM Studio | 3.52 | 1/3 (hist.) | — | deleted |
| qwen3-32b | 32B | qwen3 | — | ~41K | (deleted) | ✓ | ✗ | LM Studio | 3.46 | — | — | deleted |
| llama-3.2-3b-instruct | 3B | llama3.2 | — | 131072 | (deleted) | ✗ | ✗ | LM Studio | 3.33 | — | — | deleted |
| qwen3.5-9b-mlx | 9B | qwen3.5 (MLX) | MLX | ~262K | (deleted) | ✓ | ✗ | LM Studio | 3.22 | no output (hist.) | — | deleted (looped) |
| glm-4.7-flash | ~9B | glm4 | — | ~131K | (deleted) | ✗ | ✗ | LM Studio | 2.96 | — | — | deleted |

Notes:
- Deleted-model metadata (params/ctx) is best-effort from model specs (can't query LM Studio — removed).
  Quality = their last recorded CAT overall.
- Deleted models' `(hist.)` code scores come from re-scoring their **retained implementations** against
  the same reference suites, so they *are* comparable on code correctness even though their original
  runs used mixed/older setups. Their old pass/5 figures were not comparable and are dropped.
- **CAT quality does not predict code correctness.** Ornith-35B leads CAT (4.22) but trails Gemma
  (4.10) on aider code, and Mistral (4.00) scores 1/3 despite writing tests every time. Do not use the
  CAT bar to pick an executor.
- **No promote/reject decision should cite the retired pass/5 bar.** Ornith-9B's "NOT PROMOTED at 2/5"
  was a no-op score; its real code correctness is 1/3 (aider) / 2/3 (Hermes). The rejection may still
  be right — it now needs to be re-argued on numbers that mean something.
- **Status reconciled 2026-09-16 (SPEC-003-021) against live `lms ls`.** Ornith-1.0-35B,
  Ornith-1.0-9B, and Mistral-small-3.2 are none of them currently installed — corrected from
  `fleet` to `deleted`; their historical scores above are unchanged (see R7). `qwen3.8-27b`,
  `qwen/qwen3.6-27b`, and `muse-glimmer-30b` — the three currently-installed mid-size models this
  table previously omitted entirely — gain rows above. Their `quality (CAT)` values are the
  canonical model key's `overall_mean` from `leaderboard.csv` (SPEC-003-019's repopulated
  leaderboard), each model's most complete/recent run, judge preferred in the order Argo →
  `qwen/qwen3.6-27b` → `muse-glimmer-30b`, always skipping a model judging itself. `aider code`/
  `Hermes code` are `—`: no new agentic run was executed for this reconciliation (out of scope —
  see the spec's Out of Scope section).


## 3. Public corpus

Run the versioned synthetic suite to produce new comparable scores. No score for
the public corpus is claimed by this release.
