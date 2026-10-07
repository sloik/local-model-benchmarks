#!/usr/bin/env bash
# =============================================================================
# run_mac.sh — Runs benchmark directly on Mac (without Cowork session)
#
# Difference from run_all.sh:
#   • LM Studio available through localhost:1234 (not through Cowork tunnel)
#   • Default evaluator: local (Qwen3.6 27B) — does not require API key
#   • Python run from ~/.venv-benchmark/bin/python or system Python
#   • Does not require active Cowork session / VM
#   • Always runs non-interactively (passes --yes to run_all.sh)
#
# BEFORE FIRST USE:
#   bash run_mac.sh --setup     # installs Python dependencies (openai)
#
# USAGE:
#   bash run_mac.sh --model <model_id>
#   bash run_mac.sh --model <model_id> --evaluator local
#   bash run_mac.sh --model <model_id> --evaluator claude  # requires ANTHROPIC_API_KEY
#   bash run_mac.sh --model <model_id> --categories 01 03  # only selected categories
#   bash run_mac.sh --all-models                           # all models sequentially
#
# EXAMPLES:
#   bash run_mac.sh --model gemma-4-31b-it
#   bash run_mac.sh --model qwen/qwen3.6-27b --evaluator local
#   ANTHROPIC_API_KEY=sk-ant-... bash run_mac.sh --model ornith-1.5-35b-a3b-mlx --evaluator claude
#
# MODELS: fetched dynamically from LM Studio API (only LLMs, excludes embeddings)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"

# ── Mac-specific defaults ─────────────────────────────────────────────────────
export LM_STUDIO_BASE="${LM_STUDIO_BASE:-http://localhost:1234/v1}"
# Same token as in evaluate.py — configured in LM Studio on Mac.
# If your LM Studio uses a different token, set: export LM_STUDIO_TOKEN=<token>
export LM_STUDIO_TOKEN="${LM_STUDIO_TOKEN:-}"

# Detect LM Studio port (default 1234 on Mac; in VM it was 11234 through tunnel)
# If you use a non-standard port, set LM_STUDIO_BASE manually.

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_ok()   { echo -e "${GREEN}✓${NC} $*"; }
log_warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
log_err()  { echo -e "${RED}✗${NC}  $*"; }
log_info() { echo -e "${CYAN}ℹ${NC}  $*"; }

# ── Detect Python ─────────────────────────────────────────────────────────────
VENV_DIR="$HOME/.venv-benchmark"

detect_python() {
    # 1. Dedicated venv for benchmark
    if [[ -x "$VENV_DIR/bin/python" ]]; then
        echo "$VENV_DIR/bin/python"
        return
    fi
    # 2. python3 with check for openai
    for py in python3 python3.11 python3.12 python3.10; do
        if command -v "$py" &>/dev/null; then
            if "$py" -c "import openai" 2>/dev/null; then
                echo "$py"
                return
            fi
        fi
    done
    echo ""
}

PYTHON="$(detect_python)"

# ── Parse arguments ───────────────────────────────────────────────────────────
MODEL=""
EVALUATOR="local"
CATEGORIES=()
ALL_MODELS=false
SETUP=false
EVALUATOR_MAX_TOKENS=""
EVALUATOR_MODEL_MAP=""
FORCE=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)             MODEL="$2"; shift 2 ;;
        --evaluator)         EVALUATOR="$2"; shift 2 ;;
        --categories)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CATEGORIES+=("$1"); shift
            done
            ;;
        --all-models)        ALL_MODELS=true; shift ;;
        --setup)             SETUP=true; shift ;;
        --evaluator-max-tokens) EVALUATOR_MAX_TOKENS="$2"; shift 2 ;;
        --evaluator-model-map) EVALUATOR_MODEL_MAP="$2"; shift 2 ;;
        --force)             FORCE=true; shift ;;
        *)                   EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Setup: install dependencies ───────────────────────────────────────────────
if [[ "$SETUP" == "true" ]]; then
    echo -e "\n${BOLD}=== Setup: Python environment for benchmark ===${NC}\n"

    if ! command -v python3 &>/dev/null; then
        log_err "python3 not found. Install Python 3.10+ from python.org"
        exit 1
    fi

    PY_VERSION=$(python3 --version 2>&1)
    log_info "Python: $PY_VERSION"

    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating venv: $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi

    log_info "Installing openai (required)..."
    "$VENV_DIR/bin/pip" install openai --quiet --upgrade

    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        log_info "Installing anthropic (found ANTHROPIC_API_KEY)..."
        "$VENV_DIR/bin/pip" install anthropic --quiet --upgrade
    fi

    log_ok "Setup complete."
    log_info "Venv: $VENV_DIR/bin/python"
    echo ""
    echo -e "Now you can run the benchmark:"
    echo -e "  ${CYAN}bash run_mac.sh --model gemma-4-31b-it${NC}"
    echo ""
    exit 0
fi

# ── Check Python ──────────────────────────────────────────────────────────────
if [[ -z "$PYTHON" ]]; then
    log_err "Python with 'openai' library not found."
    echo ""
    echo "Run setup:"
    echo "  ${CYAN}bash run_mac.sh --setup${NC}"
    exit 1
fi

log_info "Python: $PYTHON"

# ── Check LM Studio ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Checking LM Studio ($LM_STUDIO_BASE)...${NC}"

LM_STATUS=$("$PYTHON" -c "
import urllib.request, json, sys
try:
    req = urllib.request.Request(
        '${LM_STUDIO_BASE}/models',
        headers={'Authorization': 'Bearer ${LM_STUDIO_TOKEN}'}
    )
    resp = urllib.request.urlopen(req, timeout=5)
    data = json.loads(resp.read())
    models = [m['id'] for m in data.get('data', [])]
    print('OK:' + ','.join(models[:5]))
except Exception as e:
    print('ERR:' + str(e))
" 2>/dev/null)

if [[ "$LM_STATUS" == ERR:* ]]; then
    log_err "LM Studio unavailable: ${LM_STATUS#ERR:}"
    echo ""
    echo "Make sure LM Studio is running on Mac."
    echo "Default port: 1234. If different, set:"
    echo "  ${CYAN}export LM_STUDIO_BASE=http://localhost:PORT/v1${NC}"
    echo ""
    # Try VM port (in case we're running in Cowork anyway)
    log_warn "Trying VM port (11234)..."
    export LM_STUDIO_BASE="http://192.168.64.1:11234/v1"
    LM_STATUS2=$("$PYTHON" -c "
import urllib.request, json, sys
try:
    req = urllib.request.Request(
        '${LM_STUDIO_BASE}/models',
        headers={'Authorization': f'Bearer {os.getenv("LM_STUDIO_TOKEN","")}'}
    )
    resp = urllib.request.urlopen(req, timeout=5)
    data = json.loads(resp.read())
    models = [m['id'] for m in data.get('data', [])]
    print('OK:' + ','.join(models[:5]))
except Exception as e:
    print('ERR:' + str(e))
" 2>/dev/null)
    if [[ "$LM_STATUS2" == OK:* ]]; then
        export LM_STUDIO_TOKEN="${LM_STUDIO_TOKEN:-}"
        log_ok "Connected through VM port (tunnel active): ${LM_STATUS2#OK:}"
    else
        log_err "LM Studio unavailable on both ports. Start LM Studio and try again."
        exit 1
    fi
else
    log_ok "LM Studio online: ${LM_STATUS#OK:}"
fi

# ── Check evaluator ───────────────────────────────────────────────────────────
if [[ "$EVALUATOR" == "claude" ]]; then
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        log_err "Missing ANTHROPIC_API_KEY (required for --evaluator claude)"
        echo ""
        echo "Set before running:"
        echo "  ${CYAN}export ANTHROPIC_API_KEY=sk-ant-...${NC}"
        echo "  ${CYAN}bash run_mac.sh --model ... --evaluator claude${NC}"
        echo ""
        echo "Or use local evaluator (default, no key required):"
        echo "  ${CYAN}bash run_mac.sh --model ... --evaluator local${NC}"
        exit 1
    fi
    log_ok "Evaluator: claude (Anthropic API)"
elif [[ "$EVALUATOR" == "local" ]]; then
    log_ok "Evaluator: local (Qwen3.6 primary in LM Studio)"
fi

# ── Fetch models from LM Studio API ──────────────────────────────────────────
# Queries the native REST API (/api/v1/models) to get all downloaded LLMs.
# Filters out non-LLM types (embeddings, etc.).
fetch_models_from_lmstudio() {
    local base_url="${LM_STUDIO_BASE%/v1}"  # strip /v1 suffix to get base URL
    local token="$LM_STUDIO_TOKEN"
    "$PYTHON" - "$base_url" "$token" <<'PYEOF'
import urllib.request, json, sys
base_url, token = sys.argv[1], sys.argv[2]
try:
    req = urllib.request.Request(
        f'{base_url}/api/v1/models',
        headers={'Authorization': f'Bearer {token}'}
    )
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read())
    models = data.get('models', data.get('data', []))
    # Filter: only LLMs (skip embeddings, etc.)
    llm_keys = [m.get('key', m.get('id', '')) for m in models if m.get('type', 'llm') == 'llm']
    for k in sorted(llm_keys):
        print(k)
except Exception as e:
    print('ERR:' + str(e), file=sys.stderr)
    sys.exit(1)
PYEOF
}

ALL_MODELS_LIST=()
if [[ "$ALL_MODELS" == "true" ]] || [[ -z "$MODEL" ]]; then
    log_info "Fetching model list from LM Studio..."
    while IFS= read -r line; do
        ALL_MODELS_LIST+=("$line")
    done < <(fetch_models_from_lmstudio)

    if [[ ${#ALL_MODELS_LIST[@]} -eq 0 ]]; then
        log_err "No LLM models found in LM Studio. Download models first."
        exit 1
    fi
    log_ok "Found ${#ALL_MODELS_LIST[@]} LLM models in LM Studio"
fi

# ── Run ────────────────────────────────────────────────────────────────────────
echo ""

run_for_model() {
    local m="$1"
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Model: $m${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"

    # Build args as array — safe for models with / and spaces
    local run_args=("--model" "$m" "--evaluator" "$EVALUATOR" "--yes")

    if [[ ${#CATEGORIES[@]} -gt 0 ]]; then
        run_args+=("--categories" "${CATEGORIES[@]}")
    fi

    if [[ -n "$EVALUATOR_MAX_TOKENS" ]]; then
        run_args+=("--evaluator-max-tokens" "$EVALUATOR_MAX_TOKENS")
    fi

    if [[ -n "$EVALUATOR_MODEL_MAP" ]]; then
        run_args+=("--evaluator-model-map" "$EVALUATOR_MODEL_MAP")
    fi

    if [[ "$FORCE" == "true" ]]; then
        run_args+=("--force")
    fi

    # EXTRA_ARGS may be empty — safe expansion by checking size
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        run_args+=("${EXTRA_ARGS[@]}")
    fi

    PYTHONPATH="$SCRIPT_DIR" \
    LM_STUDIO_BASE="$LM_STUDIO_BASE" \
    LM_STUDIO_TOKEN="$LM_STUDIO_TOKEN" \
    bash "$SCRIPT_DIR/run_all.sh" "${run_args[@]}"
}

if [[ "$ALL_MODELS" == "true" ]]; then
    log_info "--all-models mode: ${#ALL_MODELS_LIST[@]} models"
    for m in "${ALL_MODELS_LIST[@]}"; do
        run_for_model "$m" || log_warn "Model $m finished with error — continuing"
        echo ""
    done
    log_ok "All models complete."
elif [[ -n "$MODEL" ]]; then
    run_for_model "$MODEL"
else
    echo -e "${BOLD}USAGE:${NC}"
    echo "  bash run_mac.sh --model <model_id>"
    echo "  bash run_mac.sh --all-models"
    echo "  bash run_mac.sh --setup"
    echo ""
    echo -e "${BOLD}AVAILABLE MODELS (from LM Studio):${NC}"
    if [[ ${#ALL_MODELS_LIST[@]} -gt 0 ]]; then
        for m in "${ALL_MODELS_LIST[@]}"; do
            echo "  • $m"
        done
    else
        echo "  (could not fetch — is LM Studio running?)"
    fi
    echo ""
    echo -e "${BOLD}OPTIONS:${NC}"
    echo "  --evaluator local|claude          (default: local)"
    echo "  --categories 01 02 03             (default: all)"
    echo "  --evaluator-max-tokens N          (default: 8192 for local)"
    echo "  --judge-panel                    Qwen primary + separate Muse dissent pass"
    echo "  --evaluator-model-map MAP         Per-category evaluator override"
    echo "                                    (default: none; Qwen judges every category)"
    echo "  --force                           (overwrite existing results)"
    exit 0
fi
