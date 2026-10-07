#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Complete benchmark pipeline for local LLM models
#
# Runs sequentially: inference → evaluation → report
# Checks LM Studio availability before starting.
# Evaluator is selected interactively (or via --evaluator flag).
#
# USAGE:
#   ./run_all.sh --model <model_id>                          # full pipeline
#   ./run_all.sh --model <model_id> --categories 05 01       # selected categories only
#   ./run_all.sh --model <model_id> --skip-eval              # inference only
#   ./run_all.sh --model <model_id> --evaluator claude       # evaluator without prompt
#   ./run_all.sh --list-models                               # show available models
#   ./run_all.sh --help                                      # this help
#
# EXAMPLES:
#   ./run_all.sh --model ornith-1.5-35b-a3b-mlx
#   ./run_all.sh --model qwen/qwen3.6-27b --categories 05
#   ./run_all.sh --model ornith-1.5-35b-a3b-mlx --evaluator claude
#
# ENVIRONMENT VARIABLES (optional):
#   LM_STUDIO_BASE   — Endpoint URL (default: http://127.0.0.1:1234/v1)
#   LM_STUDIO_TOKEN  — Optional API token (empty for unauthenticated localhost)
#   ANTHROPIC_API_KEY — Required when --evaluator claude
#
# RESULTS STRUCTURE:
#   results/{date}_{model}/
#     raw/                  ← raw model responses (one .txt per prompt)
#     scores.json           ← evaluator scores (1-5) with justifications
#     REPORT.md             ← human-readable report
#   leaderboard.csv         ← comparative table of all runs
#
# EVALUATORS:
#   local   — local model in LM Studio (default judge: qwen/qwen3.6-27b)
#             no API key required, works offline
#   claude  — Claude Sonnet via Anthropic API (gold standard from SPEC.md)
#             requires ANTHROPIC_API_KEY variable
#   argo    — Argo evaluates manually in Cowork session
#             generates argo_eval_tasks.json → fill scores → --finalize
# =============================================================================

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$BENCH_DIR/tests"
RESULTS_DIR="$BENCH_DIR/results"

LM_BASE="${LM_STUDIO_BASE:-http://127.0.0.1:1234/v1}"
LM_TOKEN="${LM_STUDIO_TOKEN:-}"

# ── Default argument values ─────────────────────────────────────────────
MODEL=""
CATEGORIES=""          # empty = all
TEST_IDS=""
EVALUATOR_TYPE=""      # empty = ask interactively
EVALUATOR_MODEL=""          # empty = default per type
EVALUATOR_MODEL_MAP=""      # empty = no overrides; format: CAT-XX:model_id,CAT-YY:model_id
EVALUATOR_MAX_TOKENS=""     # empty = default evaluate.py (512); recommended: 2048 for local
JUDGE_PANEL=false            # Qwen primary scores.json + separate Muse dissent scores
MAX_TOKENS=""               # empty = default run_benchmark.py (32768)
RETRY_MAX_TOKENS=""         # empty = default run_benchmark.py (81920)
RUN_TAG=""             # empty = {date}_{model}; with tag = {date}_{model}_{tag}
ROUTE_PREFLIGHT_FILE=""     # empty = off; a JSON/JSONL file enables the A1 routing preflight report
SKIP_EVAL=false
SKIP_REPORT=false
LIST_MODELS=false
ALL_MODELS=false      # test all loaded models (skipping embeddings)
DRY_RUN=false
FORCE=false           # --force: overwrite existing results (inference + evaluation)
AUTO_YES=false        # --yes: skip all interactive prompts (auto-confirm)

# ── Funkcje pomocnicze ────────────────────────────────────────────────────────

log_info()    { echo -e "${BLUE}ℹ${NC}  $*"; }
log_ok()      { echo -e "${GREEN}✅${NC} $*"; }
log_warn()    { echo -e "${YELLOW}⚠️${NC}  $*"; }
log_error()   { echo -e "${RED}❌${NC} $*" >&2; }
log_section() { echo -e "\n${BOLD}${CYAN}━━━ $* ━━━${NC}"; }
log_dim()     { echo -e "${DIM}   $*${NC}"; }

# Separator wizualny
sep() { echo -e "${DIM}────────────────────────────────────────────────────${NC}"; }

# ── Argument parsing ─────────────────────────────────────────────────
usage() {
    cat <<'HELP'
run_all.sh — Complete benchmark pipeline for local LLM models
Runs sequentially: inference → evaluation → report
Checks LM Studio availability before starting.
Evaluator is selected interactively (or via --evaluator flag).

USAGE:
  ./run_all.sh --model <model_id>                          # full pipeline, single model
  ./run_all.sh --all-models                                # test ALL models
  ./run_all.sh --model <model_id> --categories 05 01       # selected categories only
  ./run_all.sh --model <model_id> --skip-eval              # inference only
  ./run_all.sh --model <model_id> --evaluator claude       # evaluator without prompt
  ./run_all.sh --list-models                               # show available models
  ./run_all.sh --help                                      # this help

FLAGS:
  --model <id>            Model ID in LM Studio (required or --all-models)
  --all-models            Run benchmark on all loaded models
                          (automatically skips embedding models)
  --categories <nn ...>   Category numbers to run (e.g., 05 01 03)
                          Default: all available (01–05)
  --test-ids <id ...>     Run only specific tests (e.g., CAT-05-001)
  --evaluator <type>      claude | local | argo (skip interactive prompt)
  --evaluator-model <id>        Specific evaluator model (default: qwen/qwen3.6-27b)
  --judge-panel                Run calibrated Qwen primary + Muse dissent local passes
  --evaluator-model-map <map>   Per-category evaluator overrides.
                                Format: CAT-XX:model_id,CAT-YY:model_id
                                Example: CAT-04:google/gemma-4-31b
                                Categories not in the map use --evaluator-model default.
  --evaluator-max-tokens <n>    Max tokens for evaluator response (default: 512)
                                Recommended: 2048 for local, 512 sufficient for claude
  --max-tokens <n>              Standard generation ceiling (default: 32768)
  --retry-max-tokens <n>        One retry ceiling for empty/truncated responses
                                (default: 81920; 0 disables)
  --run-tag <tag>         Tag to distinguish multiple runs on same day
                          e.g., --run-tag ctx4k → folder: {date}_{model}_ctx4k
                          Allows testing different configs without overwriting
  --route-preflight <f>   Print the A1 verifiability-routing report for a JSON/JSONL
                          file of {question, field} items before the run. Non-blocking,
                          opt-in; makes no model calls. Omit = unchanged behaviour.
  --skip-eval             Skip evaluation step — inference only
  --skip-report           Skip report step
  --force                 Force re-run: overwrite existing raw responses,
                          scores.json, and report. Without this flag,
                          existing files are skipped.
  --list-models           Show available models in LM Studio and exit
  --dry-run               Show what would be sent — without API calls

EXAMPLES:
  ./run_all.sh --model ornith-1.5-35b-a3b-mlx
  ./run_all.sh --all-models                                # benchmark "all everything"
  ./run_all.sh --all-models --categories 05                # all, classification only
  ./run_all.sh --all-models --evaluator local              # without evaluator prompt
  ./run_all.sh --all-models --max-tokens 32768 --run-tag full
  ./run_all.sh --model gemma-4-31b-it --categories 05
  ./run_all.sh --model ornith-1.5-35b-a3b-mlx --evaluator claude
  ./run_all.sh --model gemma-4-31b-it --judge-panel

ENVIRONMENT VARIABLES (optional):
  LM_STUDIO_BASE    Endpoint URL (default: http://127.0.0.1:1234/v1)
  LM_STUDIO_TOKEN   Optional API token (empty for unauthenticated localhost)
  ANTHROPIC_API_KEY Required when --evaluator claude

RESULTS STRUCTURE:
  results/{date}_{model}/
    raw/          ← raw model responses (one .txt per prompt)
    scores.json   ← evaluator scores (1–5) with justifications
    REPORT.md     ← human-readable report
  leaderboard.csv ← comparative table of all runs

EVALUATORS:
  local   Qwen3.6 primary local judge; use --judge-panel to add Muse dissent
          No API key required, works offline
  claude  Claude Sonnet via Anthropic API — gold standard (SPEC.md Phase 1)
          Requires ANTHROPIC_API_KEY
  argo    Argo (Claude in Cowork session) evaluates manually
          Generates argo_eval_tasks.json → fill scores → --finalize
          After completion: python evaluate.py --run-dir ... --evaluator-type argo --finalize
HELP
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)           MODEL="$2";           shift 2 ;;
        --categories)      CATEGORIES="$2";      shift 2 ;;
        --test-ids)        TEST_IDS="$2";        shift 2 ;;
        --evaluator)       EVALUATOR_TYPE="$2";  shift 2 ;;
        --evaluator-model) EVALUATOR_MODEL="$2"; shift 2 ;;
        --evaluator-model-map) EVALUATOR_MODEL_MAP="$2"; shift 2 ;;
        --judge-panel) JUDGE_PANEL=true; shift ;;
        --evaluator-max-tokens) EVALUATOR_MAX_TOKENS="$2"; shift 2 ;;
        --max-tokens)      MAX_TOKENS="$2";      shift 2 ;;
        --retry-max-tokens) RETRY_MAX_TOKENS="$2"; shift 2 ;;
        --run-tag)         RUN_TAG="$2";         shift 2 ;;
        --route-preflight) ROUTE_PREFLIGHT_FILE="$2"; shift 2 ;;
        --skip-eval)       SKIP_EVAL=true;       shift   ;;
        --skip-report)     SKIP_REPORT=true;     shift   ;;
        --list-models)     LIST_MODELS=true;     shift   ;;
        --all-models)      ALL_MODELS=true;      shift   ;;
        --dry-run)         DRY_RUN=true;         shift   ;;
        --force)           FORCE=true;           shift   ;;
        --yes|-y)          AUTO_YES=true;        shift   ;;
        --help|-h)         usage ;;
        *) log_error "Unknown flag: $1. Use --help"; exit 1 ;;
    esac
done

# ── Environment validation ──────────────────────────────────────────────────────

check_deps() {
    log_section "Checking dependencies"

    # python3
    if ! command -v python3 &>/dev/null; then
        log_error "python3 not found"; exit 1
    fi
    log_ok "python3 $(python3 --version 2>&1 | cut -d' ' -f2)"

    # jq (used to display test descriptions)
    if command -v jq &>/dev/null; then
        log_ok "jq $(jq --version)"
        JQ_AVAILABLE=true
    else
        log_warn "jq unavailable — test descriptions will not be displayed"
        JQ_AVAILABLE=false
    fi

    # curl (to check LM Studio)
    if ! command -v curl &>/dev/null; then
        log_error "curl not found"; exit 1
    fi
    log_ok "curl available"
}

# ── Tunnel and LM Studio check ───────────────────────────────────────────

check_lm_studio() {
    log_section "Checking LM Studio"
    log_dim "Endpoint: $LM_BASE"

    # Ping with 5s timeout
    local response
    response=$(curl -s --max-time 5 \
        -H "Authorization: Bearer $LM_TOKEN" \
        "$LM_BASE/models" 2>/dev/null) || true

    if [[ -z "$response" ]]; then
        log_error "No response from LM Studio ($LM_BASE)"
        echo ""
        echo -e "  ${YELLOW}Start tunnel:${NC}"
        echo -e "  ${CYAN}  _Tools/lmstudio-tunnel.sh start${NC}"
        echo ""
        exit 1
    fi

    # Parse available models
    if command -v jq &>/dev/null; then
        AVAILABLE_MODELS=$(echo "$response" | jq -r '.data[].id' 2>/dev/null || echo "")
    else
        AVAILABLE_MODELS=$(echo "$response" | python3 -c \
            "import json,sys; d=json.load(sys.stdin); print('\n'.join(m['id'] for m in d['data']))" 2>/dev/null || echo "")
    fi

    if [[ -z "$AVAILABLE_MODELS" ]]; then
        log_error "LM Studio responded but no models loaded"
        log_warn "Load a model in LM Studio and try again"
        exit 1
    fi

    log_ok "LM Studio available"
    log_dim "Loaded models:"
    while IFS= read -r m; do
        log_dim "  • $m"
    done <<< "$AVAILABLE_MODELS"
}

# ── Model list (--list-models) ─────────────────────────────────────────────

list_models_and_exit() {
    echo ""
    echo -e "${BOLD}Available models in LM Studio:${NC}"
    while IFS= read -r m; do
        echo -e "  ${CYAN}$m${NC}"
    done <<< "$AVAILABLE_MODELS"
    echo ""
    echo -e "Usage: ${CYAN}./run_all.sh --model <model_id>${NC}"
    exit 0
}

# ── Model validation ─────────────────────────────────────────────────────────

validate_model() {
    if [[ -z "$MODEL" ]]; then
        log_error "Missing --model. Use --list-models or --all-models."
        exit 1
    fi

    if ! echo "$AVAILABLE_MODELS" | grep -qxF "$MODEL"; then
        log_warn "Model '$MODEL' is not currently loaded in LM Studio"
        log_dim "Loaded: $(echo "$AVAILABLE_MODELS" | tr '\n' ' ')"
        echo ""
        if [[ "$AUTO_YES" == "true" ]]; then
            log_dim "Auto-yes: continuing anyway"
        else
            read -rp "$(echo -e "${YELLOW}Continue anyway? [y/N]:${NC} ")" confirm
            [[ "$confirm" =~ ^[tTyY]$ ]] || exit 1
        fi
    else
        log_ok "Model verified: $MODEL"
    fi
}

# ── Calculate run_dir (mirror sanitize_model_id from run_benchmark.py) ─────────
#
# Pattern: {date}_{model_safe}[_{run_tag}]
# sanitize: / → --, spaces → _, : → -
#
compute_run_dir() {
    local model_id="$1"
    local model_safe tag_suffix
    model_safe=$(printf '%s' "$model_id" | sed 's|/|--|g; s| |_|g; s|:|-|g')
    tag_suffix=$( [[ -n "$RUN_TAG" ]] && echo "_${RUN_TAG}" || echo "" )
    echo "$RESULTS_DIR/$(date +%Y-%m-%d)_${model_safe}${tag_suffix}"
}

# ── Model filtering — skip embeddings and non-chat models ───────────

# Model name patterns that are NOT chat models
EMBEDDING_PATTERNS=("embed" "embedding" "e5-" "bge-" "nomic-")

filter_chat_models() {
    # Takes list of models (newline-separated), returns only chat models
    local input="$1"
    local result=""
    while IFS= read -r m; do
        local skip=false
        for pat in "${EMBEDDING_PATTERNS[@]}"; do
            if echo "$m" | grep -qi "$pat"; then
                skip=true
                break
            fi
        done
        if [[ "$skip" == "false" ]]; then
            result+="$m"$'\n'
        fi
    done <<< "$input"
    echo -n "$result"
}

# ── Run pipeline for one model (used in --all-models loop) ────────

run_pipeline_for_model() {
    local current_model="$1"
    local model_num="$2"
    local model_total="$3"

    echo ""
    echo -e "${BOLD}${CYAN}┌─────────────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}${CYAN}│  Model ${model_num}/${model_total}: ${current_model}│${NC}"
    echo -e "${BOLD}${CYAN}└─────────────────────────────────────────────────────┘${NC}"
    echo ""

    # Temporarily set global MODEL
    local prev_model="$MODEL"
    MODEL="$current_model"

    # Calculate expected RUN_DIR before running inference
    RUN_DIR=$(compute_run_dir "$MODEL")

    # Inference
    log_section "Step 1/3 — Inference"
    log_dim "Model: $MODEL"
    echo ""

    local inf_args=("--model" "$MODEL")
    [[ -n "$CATEGORIES" ]]  && inf_args+=("--categories" $CATEGORIES)
    [[ -n "$TEST_IDS" ]]    && inf_args+=("--test-ids" $TEST_IDS)
    [[ -n "$MAX_TOKENS" ]]  && inf_args+=("--max-tokens" "$MAX_TOKENS")
    [[ -n "$RETRY_MAX_TOKENS" ]] && inf_args+=("--retry-max-tokens" "$RETRY_MAX_TOKENS")
    [[ -n "$RUN_TAG" ]]     && inf_args+=("--run-tag" "$RUN_TAG")
    [[ "$DRY_RUN" == "true" ]] && inf_args+=("--dry-run")
    [[ "$FORCE" == "true" ]] && inf_args+=("--force")

    if python3 "$SCRIPT_DIR/run_benchmark.py" "${inf_args[@]}"; then
        log_ok "Inference OK → $(basename "$RUN_DIR")"
    else
        log_error "Inference FAILED for $MODEL — skipping"
        MODEL="$prev_model"
        return 1
    fi

    # Evaluation
    if [[ "$SKIP_EVAL" == "false" ]]; then
        if [[ "$EVALUATOR_TYPE" == "argo" ]]; then
            log_section "Step 2/3 — Generating Argo tasks"
            log_dim "Run dir: $RUN_DIR"
            echo ""
            python3 "$SCRIPT_DIR/evaluate.py" \
                "--run-dir" "$RUN_DIR" \
                "--evaluator-type" "argo" || true
            log_ok "Argo tasks → $(basename "$RUN_DIR")/argo_eval_tasks.json"
            log_warn "Step 3/3 (Report) skipped — run after argo finalization"
        else
            log_section "Step 2/3 — Evaluation"
            log_dim "Evaluator: $EVALUATOR_TYPE ($EVALUATOR_MODEL)"
            echo ""

            local eval_args=("--run-dir" "$RUN_DIR" "--evaluator-type" "$EVALUATOR_TYPE")
            [[ -n "$EVALUATOR_MODEL" ]] && eval_args+=("--evaluator-model" "$EVALUATOR_MODEL")
            [[ -n "$EVALUATOR_MODEL_MAP" ]] && eval_args+=("--evaluator-model-map" "$EVALUATOR_MODEL_MAP")
            [[ -n "$EVALUATOR_MAX_TOKENS" ]] && eval_args+=("--evaluator-max-tokens" "$EVALUATOR_MAX_TOKENS")
            [[ "$FORCE" == "true" ]] && eval_args+=("--force")

            if python3 "$SCRIPT_DIR/evaluate.py" "${eval_args[@]}"; then
                log_ok "Evaluation OK → $(basename "$RUN_DIR")/scores.json"
            else
                log_warn "Evaluation FAILED for $MODEL — skipping report"
                MODEL="$prev_model"
                return 1
            fi

            # Report
            if [[ "$SKIP_REPORT" == "false" ]]; then
                log_section "Step 3/3 — Report"
                echo ""
                if python3 "$SCRIPT_DIR/report.py" "--run-dir" "$RUN_DIR"; then
                    log_ok "Report OK → $(basename "$RUN_DIR")/REPORT.md"
                else
                    log_warn "Report FAILED for $MODEL"
                fi
            fi
        fi
    fi

    MODEL="$prev_model"
    return 0
}

# ── Comparative summary after --all-models ──────────────────────────────────

show_all_models_summary() {
    local run_ids=("$@")   # list of run_ids that were executed in this session

    echo ""
    log_section "Comparative summary — all models"
    echo ""

    local lb="$BENCH_DIR/leaderboard.csv"
    if [[ ! -f "$lb" || "$JQ_AVAILABLE" != "true" ]]; then
        log_warn "Missing leaderboard.csv or jq — cannot generate comparison"
        return
    fi

    # Table header
    printf "  %-40s  %6s  %6s  %6s  %6s  %6s  %7s\n" \
        "Model" "CAT-01" "CAT-02" "CAT-03" "CAT-04" "CAT-05" "Overall"
    echo -e "  ${DIM}$(printf '%.0s─' {1..84})${NC}"

    # Read leaderboard.csv and filter rows from current session
    local best_overall=0
    local best_model=""

    python3 - "$lb" "${run_ids[@]}" <<'PYEOF'
import csv, sys

lb_file = sys.argv[1]
run_ids = set(sys.argv[2:])

rows = []
with open(lb_file, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if not run_ids or row["run_id"] in run_ids:
            rows.append(row)

if not rows:
    print("  No results to display")
    sys.exit(0)

# Find best overall
best = max(rows, key=lambda r: float(r["overall_mean"] or 0))

for row in rows:
    model = row["model_id"][:38]
    cats = [row.get(f"CAT-0{i}_mean", "") or "—" for i in range(1, 6)]
    overall = row.get("overall_mean", "") or "—"

    # Format values
    def fmt(v):
        try: return f"{float(v):5.2f}"
        except: return "  —  "

    is_best = row["run_id"] == best["run_id"]
    star = " ★" if is_best else "  "
    print(f"  {model:<40} {fmt(cats[0])}  {fmt(cats[1])}  {fmt(cats[2])}  {fmt(cats[3])}  {fmt(cats[4])}  {fmt(overall)}{star}")
PYEOF

    echo ""
    echo -e "  ${DIM}★ = best overall in this session${NC}"
    echo -e "  ${DIM}Full leaderboard: cat $lb${NC}"
    echo ""
}

# ── Interactive evaluator selection ─────────────────────────────────────────────

choose_evaluator() {
    if [[ -n "$EVALUATOR_TYPE" ]]; then
        # Provided via --evaluator — validation
        if [[ "$EVALUATOR_TYPE" != "claude" && "$EVALUATOR_TYPE" != "local" && "$EVALUATOR_TYPE" != "argo" ]]; then
            log_error "--evaluator must be 'claude', 'local' or 'argo'"; exit 1
        fi
        if [[ "$EVALUATOR_TYPE" == "claude" ]]; then
            [[ -z "$EVALUATOR_MODEL" ]] && EVALUATOR_MODEL="claude-sonnet-4-6"
            if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
                log_error "Missing ANTHROPIC_API_KEY variable (required for --evaluator claude)"
                echo -e "  Set: ${CYAN}export ANTHROPIC_API_KEY=sk-ant-...${NC}"
                exit 1
            fi
        fi
        [[ "$EVALUATOR_TYPE" == "local" && -z "$EVALUATOR_MODEL" ]] && EVALUATOR_MODEL="qwen/qwen3.6-27b"
        return 0
    fi

    log_section "Choose evaluator"
    echo ""
    echo -e "  ${BOLD}1)${NC} ${GREEN}local${NC}  — local model in LM Studio"
    echo -e "     ${DIM}No API key required. Quality-first evaluator: qwen/qwen3.6-27b${NC}"
    echo ""
    echo -e "  ${BOLD}2)${NC} ${CYAN}claude${NC} — Claude Sonnet (Anthropic API)"
    echo -e "     ${DIM}Gold standard (SPEC.md Phase 1). Requires ANTHROPIC_API_KEY.${NC}"
    echo ""
    echo -e "  ${BOLD}3)${NC} ${YELLOW}argo${NC}   — Argo evaluates manually in Cowork session"
    echo -e "     ${DIM}Generates task file → Argo enters scores → script finalization.${NC}"
    echo ""

    while true; do
        read -rp "$(echo -e "${BOLD}Choose [1/2/3]:${NC} ")" choice
        case "$choice" in
            1) EVALUATOR_TYPE="local"
               [[ -z "$EVALUATOR_MODEL" ]] && EVALUATOR_MODEL="qwen/qwen3.6-27b"
               log_ok "Evaluator: local ($EVALUATOR_MODEL)"
               break ;;
            2) EVALUATOR_TYPE="claude"
               [[ -z "$EVALUATOR_MODEL" ]] && EVALUATOR_MODEL="claude-sonnet-4-6"
               # Check API key
               if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
                   log_error "Missing ANTHROPIC_API_KEY variable"
                   echo -e "  Set: ${CYAN}export ANTHROPIC_API_KEY=sk-ant-...${NC}"
                   exit 1
               fi
               log_ok "Evaluator: claude ($EVALUATOR_MODEL)"
               break ;;
            3) EVALUATOR_TYPE="argo"
               log_ok "Evaluator: argo (Argo will evaluate in Cowork session)"
               break ;;
            *) log_warn "Enter 1, 2 or 3" ;;
        esac
    done
}

# ── Display test descriptions before running ──────────────────────────────────

show_tests_info() {
    log_section "Tests to run"

    # Determine which JSON files to load
    local cat_files=()
    if [[ -n "$CATEGORIES" ]]; then
        for cat in $CATEGORIES; do
            # Find file matching category number
            local f
            f=$(ls "$TESTS_DIR"/cat-"$cat"-*.json 2>/dev/null | head -1 || true)
            [[ -n "$f" ]] && cat_files+=("$f") || log_warn "No file for category $cat"
        done
    else
        while IFS= read -r -d '' f; do
            cat_files+=("$f")
        done < <(find "$TESTS_DIR" -name "cat-0[0-9]-*.json" -print0 | sort -z)
    fi

    if [[ ${#cat_files[@]} -eq 0 ]]; then
        log_error "No test files in $TESTS_DIR"; exit 1
    fi

    local total_tests=0
    local total_prompts=0

    for f in "${cat_files[@]}"; do
        local cat_name
        cat_name=$(basename "$f" .json)

        if [[ "$JQ_AVAILABLE" == "true" ]]; then
            # Get test descriptions from JSON
            local n_tests n_prompts
            n_tests=$(jq '.tests | length' "$f")
            n_prompts=$(jq '[.tests[].prompts | length] | add // 0' "$f")

            echo -e "\n  ${BOLD}${cat_name}${NC}  ${DIM}(${n_tests} tests, ${n_prompts} prompts)${NC}"

            # Display each test with description
            jq -r '.tests[] | "  \(.id)  [\(.difficulty)]  \(.name)\n  ↳ scoring=\(.scoring)  prompts=\(.prompts | length)  min_score=\(.expected_min_score)\n  ↳ \(.notes // "-")"' "$f" \
                | while IFS= read -r line; do
                    echo -e "  ${DIM}${line}${NC}"
                done

            total_tests=$((total_tests + n_tests))
            total_prompts=$((total_prompts + n_prompts))
        else
            # Fallback without jq
            echo -e "\n  ${BOLD}${cat_name}${NC}"
        fi
    done

    echo ""
    log_info "Total: ${BOLD}${total_tests} tests${NC}, ${BOLD}${total_prompts} prompts${NC}"
    sep
}

# ── Step 1: Inference ─────────────────────────────────────────────────────────

run_inference() {
    # Calculate expected RUN_DIR (identical pattern as run_benchmark.py)
    RUN_DIR=$(compute_run_dir "$MODEL")

    log_section "Step 1/3 — Inference (run_benchmark.py)"
    log_dim "Model: $MODEL"
    log_dim "Results will be saved to: $RUN_DIR"
    echo ""

    # Build arguments for run_benchmark.py
    local args=("--model" "$MODEL")
    [[ -n "$CATEGORIES" ]]  && args+=("--categories" $CATEGORIES)
    [[ -n "$TEST_IDS" ]]    && args+=("--test-ids" $TEST_IDS)
    [[ -n "$MAX_TOKENS" ]]  && args+=("--max-tokens" "$MAX_TOKENS")
    [[ -n "$RETRY_MAX_TOKENS" ]] && args+=("--retry-max-tokens" "$RETRY_MAX_TOKENS")
    [[ -n "$RUN_TAG" ]]     && args+=("--run-tag" "$RUN_TAG")
    [[ "$DRY_RUN" == "true" ]] && args+=("--dry-run")
    [[ "$FORCE" == "true" ]] && args+=("--force")

    python3 "$SCRIPT_DIR/run_benchmark.py" "${args[@]}"

    log_ok "Inference completed → $(basename "$RUN_DIR")"
}

# ── Step 2: Evaluation ───────────────────────────────────────────────────────

run_evaluation() {
    if [[ "$EVALUATOR_TYPE" == "argo" ]]; then
        log_section "Step 2/3 — Generating Argo tasks (evaluate.py --evaluator-type argo)"
        log_dim "Run dir: $RUN_DIR"
        echo ""

        python3 "$SCRIPT_DIR/evaluate.py" \
            "--run-dir" "$RUN_DIR" \
            "--evaluator-type" "argo"

        echo ""
        log_ok "Tasks generated → $RUN_DIR/argo_eval_tasks.json"
        echo ""
        echo -e "  ${YELLOW}After Argo fills in scores, run finalization:${NC}"
        echo -e "  ${CYAN}  python3 $SCRIPT_DIR/evaluate.py \\${NC}"
        echo -e "  ${CYAN}      --run-dir $RUN_DIR \\${NC}"
        echo -e "  ${CYAN}      --evaluator-type argo --finalize${NC}"
        echo ""
        echo -e "  ${DIM}Then: python3 $SCRIPT_DIR/report.py --run-dir $RUN_DIR${NC}"
        return 0
    fi

    log_section "Step 2/3 — Evaluation (evaluate.py)"
    log_dim "Run dir: $RUN_DIR"
    log_dim "Evaluator: $EVALUATOR_TYPE ($EVALUATOR_MODEL)"
    echo ""

    if [[ "$JUDGE_PANEL" == "true" ]]; then
        if [[ "$EVALUATOR_TYPE" != "local" ]]; then
            log_error "--judge-panel requires --evaluator local"
            return 1
        fi
        if [[ -n "$EVALUATOR_MODEL_MAP" ]]; then
            log_error "--judge-panel cannot be combined with --evaluator-model-map"
            return 1
        fi
        if [[ -n "$EVALUATOR_MODEL" ]]; then
            log_error "--judge-panel uses the fixed calibrated Qwen/Muse pair and cannot be combined with --evaluator-model"
            return 1
        fi

        local panel_common=("--run-dir" "$RUN_DIR"
                            "--evaluator-type" "local"
                            "--allow-self-evaluation")
        [[ -n "$EVALUATOR_MAX_TOKENS" ]] && panel_common+=("--evaluator-max-tokens" "$EVALUATOR_MAX_TOKENS")
        [[ "$FORCE" == "true" ]] && panel_common+=("--force")

        python3 "$SCRIPT_DIR/evaluate.py" "${panel_common[@]}" \
            --evaluator-model "qwen/qwen3.6-27b" \
            --output-file "scores.json"
        python3 "$SCRIPT_DIR/evaluate.py" "${panel_common[@]}" \
            --evaluator-model "lmstudio-community/muse-glimmer-30b" \
            --output-file "scores-judge-muse-glimmer-30b.json"
        log_ok "Judge panel completed → scores.json (Qwen primary) + Muse dissent file"
        return 0
    fi

    local args=("--run-dir" "$RUN_DIR"
                 "--evaluator-type" "$EVALUATOR_TYPE")
    [[ -n "$EVALUATOR_MODEL" ]] && args+=("--evaluator-model" "$EVALUATOR_MODEL")
    [[ -n "$EVALUATOR_MODEL_MAP" ]] && args+=("--evaluator-model-map" "$EVALUATOR_MODEL_MAP")
    [[ -n "$EVALUATOR_MAX_TOKENS" ]] && args+=("--evaluator-max-tokens" "$EVALUATOR_MAX_TOKENS")
    [[ "$FORCE" == "true" ]] && args+=("--force")

    python3 "$SCRIPT_DIR/evaluate.py" "${args[@]}"

    log_ok "Evaluation completed → $RUN_DIR/scores.json"
}

# ── Step 3: Report ───────────────────────────────────────────────────────────

run_report() {
    log_section "Step 3/3 — Report (report.py)"
    log_dim "Run dir: $RUN_DIR"
    echo ""

    python3 "$SCRIPT_DIR/report.py" "--run-dir" "$RUN_DIR"

    log_ok "Report ready → $RUN_DIR/REPORT.md"
    log_ok "Leaderboard updated → $BENCH_DIR/leaderboard.csv"
}

# ── Final summary ──────────────────────────────────────────────────

show_summary() {
    echo ""
    log_section "Summary"

    local scores_file="$RUN_DIR/scores.json"
    if [[ -f "$scores_file" && "$JQ_AVAILABLE" == "true" ]]; then
        echo ""
        echo -e "  ${BOLD}Model:${NC} $MODEL"
        echo -e "  ${BOLD}Run:${NC}   $(basename "$RUN_DIR")"
        echo ""

        # Results table per category
        echo -e "  ${BOLD}Category        Score   Bar${NC}"
        sep
        local cat_map='{"CAT-01":"Extraction   ","CAT-02":"Distillation ","CAT-03":"Reasoning    ","CAT-04":"Code         ","CAT-05":"Classification"}'
        jq -r --argjson cmap "$cat_map" '
            .summary |
            to_entries[] |
            select(.key | startswith("CAT")) |
            .key as $k |
            .value.mean as $m |
            ($cmap[$k] // $k) as $name |
            ($m | . * 10 | round / 10) as $score |
            ($score / 5 * 10 | floor) as $filled |
            (10 - $filled) as $empty |
            "\($name)  \($score)/5   " + ("█" * $filled) + ("░" * $empty)
        ' "$scores_file" | while IFS= read -r line; do
            echo -e "  $line"
        done

        sep
        local overall
        overall=$(jq -r '.summary.overall.mean // "—"' "$scores_file")
        local labels
        labels=$(jq -r '.summary | to_entries[] | select(.key | startswith("CAT")) | .key as $k | .value.mean | if . >= 4.0 then $k else empty end' "$scores_file" | tr '\n' ' ')
        echo -e "  ${BOLD}Overall:${NC} $overall/5"
        [[ -n "$labels" ]] && echo -e "  ${BOLD}≥4.0 in:${NC}  $labels"
    fi

    echo ""
    echo -e "  ${DIM}Full report:  cat $RUN_DIR/REPORT.md${NC}"
    echo -e "  ${DIM}Leaderboard:  cat $BENCH_DIR/leaderboard.csv${NC}"
    echo ""
}

# ── Routing preflight (A1, opt-in) ─────────────────────────────────────────────
#
# When --route-preflight <file> is passed, run the verifiability router over the
# file's {question, field} items and print the tier/route report before the run.
# Non-blocking by design: a bad file or a router error warns and continues.
run_route_preflight() {
    [[ -z "$ROUTE_PREFLIGHT_FILE" ]] && return 0

    log_section "Verifiability routing preflight (A1)"
    if [[ ! -f "$ROUTE_PREFLIGHT_FILE" ]]; then
        log_warn "Route-preflight file not found: $ROUTE_PREFLIGHT_FILE — skipping"
        return 0
    fi

    if ! python3 "$SCRIPT_DIR/route_preflight.py" "$ROUTE_PREFLIGHT_FILE"; then
        log_warn "Route preflight reported an error — continuing (non-blocking)"
    fi
    return 0
}

# ── MAIN ──────────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║        Inwestomat — LLM Benchmark            ║${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════╝${NC}"
    echo ""

    # Check dependencies (python3, jq, curl)
    check_deps

    # Routing preflight (A1) — opt-in via --route-preflight; prints before the run, never blocks
    run_route_preflight

    # Check LM Studio — if unavailable, script exits with error
    check_lm_studio

    # --list-models mode: show models and exit
    [[ "$LIST_MODELS" == "true" ]] && list_models_and_exit

    # Mutual exclusion --model and --all-models
    if [[ "$ALL_MODELS" == "true" && -n "$MODEL" ]]; then
        log_error "--model and --all-models are mutually exclusive"
        exit 1
    fi

    # ── Mode --all-models ─────────────────────────────────────────────────────
    if [[ "$ALL_MODELS" == "true" ]]; then

        # Filter chat models (remove embeddings)
        local chat_models
        chat_models=$(filter_chat_models "$AVAILABLE_MODELS")

        if [[ -z "$chat_models" ]]; then
            log_error "No chat models in LM Studio (all are embeddings?)"
            exit 1
        fi

        local model_count
        model_count=$(echo "$chat_models" | grep -c .)

        log_section "Mode: All models"
        echo ""
        echo -e "  ${BOLD}Models to test (${model_count}):${NC}"
        local i=1
        while IFS= read -r m; do
            echo -e "  ${CYAN}  ${i}) $m${NC}"
            i=$((i+1))
        done <<< "$chat_models"

        # Show skipped embeddings
        local skipped
        skipped=$(comm -23 <(echo "$AVAILABLE_MODELS" | sort) <(echo "$chat_models" | sort) 2>/dev/null || true)
        if [[ -n "$skipped" ]]; then
            echo ""
            echo -e "  ${DIM}Skipped (embedding/non-chat):${NC}"
            while IFS= read -r m; do
                echo -e "  ${DIM}    • $m${NC}"
            done <<< "$skipped"
        fi
        echo ""

        # Select evaluator once for all models
        [[ "$SKIP_EVAL" == "false" ]] && choose_evaluator

        # Show tests
        show_tests_info

        # Estimated time
        local n_prompts
        n_prompts=$(python3 -c "
import json, os, glob
tests_dir = '$TESTS_DIR'
cats = '$CATEGORIES'.split() if '$CATEGORIES' else []
total = 0
for f in sorted(glob.glob(os.path.join(tests_dir, 'cat-*.json'))):
    cat_num = os.path.basename(f)[4:6]
    if cats and cat_num not in cats:
        continue
    d = json.load(open(f))
    total += sum(len(t['prompts']) for t in d['tests'])
print(total)
" 2>/dev/null || echo "0")

        local n_prompts_num
        n_prompts_num=$(printf '%s' "$n_prompts" | grep -E '^[0-9]+$' || echo 0)
        echo -e "  ${DIM}Total API calls: ~${model_count} models × ${n_prompts} prompts = ~$((model_count * n_prompts_num)) inference + evaluation${NC}"
        echo ""

        # Confirm start
        if [[ "$AUTO_YES" == "true" ]]; then
            log_dim "Auto-yes: starting benchmark for all ${model_count} models"
        else
            read -rp "$(echo -e "${BOLD}Start benchmark for all ${model_count} models? [Y/n]:${NC} ")" confirm
            confirm="${confirm:-Y}"
            [[ "$confirm" =~ ^[nN]$ ]] && { echo "Cancelled."; exit 0; }
        fi

        # ── Loop through all models ──────────────────────────────────────
        local succeeded=()
        local failed=()
        local run_ids=()
        local idx=1

        while IFS= read -r current_model; do
            if run_pipeline_for_model "$current_model" "$idx" "$model_count"; then
                succeeded+=("$current_model")
                # RUN_DIR set by run_pipeline_for_model() — use it directly
                [[ -n "$RUN_DIR" ]] && run_ids+=("$(basename "$RUN_DIR")")
            else
                failed+=("$current_model")
            fi
            idx=$((idx + 1))
        done <<< "$chat_models"

        # ── Summary ──────────────────────────────────────────────────────
        echo ""
        sep
        log_ok "Completed: ${#succeeded[@]}/${model_count} models OK"
        [[ ${#failed[@]} -gt 0 ]] && log_warn "Errors: ${failed[*]}"

        [[ "$SKIP_EVAL" == "false" && "$SKIP_REPORT" == "false" ]] && \
            show_all_models_summary "${run_ids[@]}"

        return 0
    fi

    # ── Normal mode — single model ───────────────────────────────────────────

    # Required --model
    validate_model

    # Select evaluator (interactively or from --evaluator)
    # Skip if --skip-eval (evaluation will not be run)
    [[ "$SKIP_EVAL" == "false" ]] && choose_evaluator

    # Display information about tests that will be run
    show_tests_info

    # Confirm start
    echo ""
    if [[ "$AUTO_YES" == "true" ]]; then
        log_dim "Auto-yes: starting pipeline"
    else
        read -rp "$(echo -e "${BOLD}Start pipeline? [Y/n]:${NC} ")" confirm
        confirm="${confirm:-Y}"
        [[ "$confirm" =~ ^[nN]$ ]] && { echo "Cancelled."; exit 0; }
    fi

    # ── Pipeline steps ────────────────────────────────────────────────────────

    # Step 1: Inference
    run_inference

    if [[ "$DRY_RUN" == "true" ]]; then
        log_ok "Dry run completed; evaluation and report were not executed"
        return 0
    fi

    # Step 2: Evaluation (optionally skipped)
    if [[ "$SKIP_EVAL" == "true" ]]; then
        log_warn "Skipped evaluation (--skip-eval)"
        log_dim "Run manually: python3 scripts/evaluate.py --run-dir $RUN_DIR"
    else
        run_evaluation
    fi

    # Step 3: Report (optionally skipped; also skip when evaluator = argo)
    if [[ "$SKIP_REPORT" == "true" || "$SKIP_EVAL" == "true" || "$EVALUATOR_TYPE" == "argo" ]]; then
        if [[ "$EVALUATOR_TYPE" == "argo" && "$SKIP_EVAL" == "false" ]]; then
            : # message already displayed by run_evaluation()
        elif [[ "$SKIP_EVAL" == "false" && "$SKIP_REPORT" == "false" ]]; then
            log_warn "Skipped report (--skip-report)"
        fi
    else
        run_report
        show_summary
    fi
}

# Run
main "$@"
