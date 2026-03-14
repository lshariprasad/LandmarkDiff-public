#!/usr/bin/env bash
# transition_phaseA_to_phaseB.sh - Automate Phase A -> Phase B training handoff
#
# Checks Phase A checkpoint, runs quick eval, patches Phase B config with the
# correct resume path, and submits the Phase B SLURM job.
#
# Usage:
#   ./scripts/transition_phaseA_to_phaseB.sh             # full transition
#   ./scripts/transition_phaseA_to_phaseB.sh --dry-run   # check everything, skip submission
#   ./scripts/transition_phaseA_to_phaseB.sh --checkpoint outputs/phaseA_v3_curriculum/checkpoint-50000

set -euo pipefail

WORKDIR="${LANDMARKDIFF_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
DRY_RUN=false
CKPT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --checkpoint) CKPT_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cd "$WORKDIR"
log "Working directory: $WORKDIR"
log "Mode: $(if $DRY_RUN; then echo 'DRY RUN'; else echo 'LIVE'; fi)"
echo ""

# -- Step 1: Locate Phase A checkpoint ----------------------------------------
log "Step 1: Locating Phase A checkpoint..."

PHASEA_OUTPUT="checkpoints_phaseA"

if [ -n "$CKPT_OVERRIDE" ]; then
    PHASEA_CKPT="$CKPT_OVERRIDE"
    log "  Using override checkpoint: $PHASEA_CKPT"
else
    # Prefer final/ if it exists, otherwise latest numbered checkpoint
    if [ -d "$PHASEA_OUTPUT/final" ]; then
        PHASEA_CKPT="$PHASEA_OUTPUT/final"
    else
        PHASEA_CKPT=$(find "$PHASEA_OUTPUT" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
            | sort -t- -k2 -n | tail -1 || true)
    fi
fi

if [ -z "$PHASEA_CKPT" ] || [ ! -d "$PHASEA_CKPT" ]; then
    log "ERROR: No Phase A checkpoint found in $PHASEA_OUTPUT"
    log "       Training may still be running. Check: squeue -u \$USER"
    exit 1
fi

CKPT_SIZE=$(du -sh "$PHASEA_CKPT" 2>/dev/null | cut -f1)
log "  Found: $PHASEA_CKPT ($CKPT_SIZE)"

if [ -d "$PHASEA_CKPT/controlnet_ema" ]; then
    log "  EMA weights present"
else
    log "  WARNING: No controlnet_ema/ subdirectory (non-EMA checkpoint)"
fi

# -- Step 2: Quick evaluation -------------------------------------------------
log ""
log "Step 2: Running quick Phase A evaluation (max 100 pairs)..."

TEST_DIR="data/splits/test"
EVAL_SCRIPT="scripts/run_evaluation.py"

if [ ! -d "$TEST_DIR" ]; then
    log "  WARNING: Test directory $TEST_DIR not found, skipping evaluation"
elif [ ! -f "$EVAL_SCRIPT" ]; then
    log "  WARNING: Eval script $EVAL_SCRIPT not found, skipping evaluation"
else
    EVAL_OUTPUT="$PHASEA_OUTPUT/transition_eval.json"
    if $DRY_RUN; then
        log "  [DRY RUN] Would run: python $EVAL_SCRIPT --checkpoint $PHASEA_CKPT --test_dir $TEST_DIR --output $EVAL_OUTPUT --max_pairs 100"
    else
        python "$EVAL_SCRIPT" \
            --checkpoint_dir "$PHASEA_CKPT" \
            --test_dir "$TEST_DIR" \
            --max_pairs 100 && {
            log "  Evaluation saved to $EVAL_OUTPUT"
        } || {
            log "  WARNING: Evaluation failed (non-fatal), continuing transition"
        }
    fi
fi

# -- Step 3: Verify ArcFace weights for Phase B identity loss ------------------
log ""
log "Step 3: Checking ArcFace weights for Phase B identity loss..."

ARCFACE_FOUND=false
for ARCFACE_PATH in \
    "$HOME/.insightface/models/buffalo_l/backbone.pth" \
    "$HOME/.cache/arcface/backbone.pth" \
    "$HOME/.insightface/models/buffalo_l/w600k_r50.onnx"; do
    if [ -f "$ARCFACE_PATH" ]; then
        ARCFACE_SIZE=$(du -sh "$ARCFACE_PATH" | cut -f1)
        log "  Found: $ARCFACE_PATH ($ARCFACE_SIZE)"
        ARCFACE_FOUND=true
        break
    fi
done

if ! $ARCFACE_FOUND; then
    log "  WARNING: No ArcFace weights found in known locations"
    log "  Phase B will attempt auto-download or fall back to SSIM identity proxy"
fi

# -- Step 4: Update Phase B config with correct resume path --------------------
log ""
log "Step 4: Updating Phase B config with resume path..."

PHASEB_CONFIG="configs/phaseB_identity.yaml"

if [ ! -f "$PHASEB_CONFIG" ]; then
    log "ERROR: Phase B config not found at $PHASEB_CONFIG"
    exit 1
fi

CURRENT_RESUME=$(grep 'resume_phaseA:' "$PHASEB_CONFIG" | sed 's/.*resume_phaseA:\s*//')

if [ "$CURRENT_RESUME" = "$PHASEA_CKPT" ]; then
    log "  Config already points to $PHASEA_CKPT (no change needed)"
else
    log "  Updating resume_phaseA: $CURRENT_RESUME -> $PHASEA_CKPT"
    if $DRY_RUN; then
        log "  [DRY RUN] Would update $PHASEB_CONFIG"
    else
        sed -i "s|resume_phaseA:.*|resume_phaseA: $PHASEA_CKPT|" "$PHASEB_CONFIG"
        log "  Updated $PHASEB_CONFIG"
    fi
fi

# -- Step 5: Submit Phase B training ------------------------------------------
log ""
log "Step 5: Submitting Phase B training job..."

PHASEB_SCRIPT="slurm/train_phaseB_multigpu.sh"
if [ ! -f "$PHASEB_SCRIPT" ]; then
    log "ERROR: Phase B SLURM script not found at $PHASEB_SCRIPT"
    exit 1
fi

if $DRY_RUN; then
    log "  [DRY RUN] Would run: sbatch $PHASEB_SCRIPT"
    JOBID="<dry-run>"
else
    mkdir -p logs outputs/phaseB_identity
    SUBMIT_OUTPUT=$(sbatch "$PHASEB_SCRIPT" 2>&1)
    JOBID=$(echo "$SUBMIT_OUTPUT" | grep -oP '\d+$' || echo "unknown")
    log "  $SUBMIT_OUTPUT"
fi

# -- Summary -------------------------------------------------------------------
log ""
log "========== Transition Summary =========="
log "  Phase A checkpoint: $PHASEA_CKPT ($CKPT_SIZE)"
log "  ArcFace weights:    $(if $ARCFACE_FOUND; then echo 'found'; else echo 'NOT FOUND (will auto-download)'; fi)"
log "  Phase B config:     $PHASEB_CONFIG"
log "  Phase B job ID:     $JOBID"
log "  Monitor with:       squeue -u \$USER"
if [ "$JOBID" != "<dry-run>" ] && [ "$JOBID" != "unknown" ]; then
    log "  Logs:               logs/phaseB_4gpu_${JOBID}.out"
fi
log "========================================"
