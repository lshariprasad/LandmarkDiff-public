#!/bin/bash
set -euo pipefail
# ============================================================================
# End-to-end LandmarkDiff training pipeline orchestrator
#
# Stages (with SLURM job dependencies):
#   1. Build combined dataset (all 3 waves + real pairs)
#   2. Create train/test split
#   3. Phase A training (diffusion loss only, ~12h on A6000)
#   4. Evaluation of Phase A checkpoint
#   5. Phase B training (4-term loss, initialized from Phase A)
#   6. Final evaluation + paper table generation
#
# Usage:
#   bash scripts/pipeline_orchestrator.sh [--dry-run]
# ============================================================================

WORK_DIR="${LANDMARKDIFF_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORK_DIR"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Will print SLURM commands without submitting"
fi

source "${CONDA_PREFIX:-$HOME/miniconda3}"/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "=============================================="
echo "  LandmarkDiff Pipeline Orchestrator"
echo "  $(date)"
echo "=============================================="

# --- Stage 0: Validate configs ---
echo ""
echo "--- Stage 0: Config Validation ---"
CONFIGS_OK=true
for cfg in configs/phaseA_production.yaml configs/phaseB_production.yaml; do
    if [ -f "$cfg" ]; then
        echo "  Validating $cfg ..."
        if ! python scripts/validate_config.py "$cfg"; then
            echo "  FAIL: $cfg has validation errors"
            CONFIGS_OK=false
        fi
    fi
done

if ! $CONFIGS_OK; then
    echo "  ERROR: Config validation failed. Fix errors before running pipeline."
    exit 1
fi
echo "  All configs valid."

# --- Stage 1: Build combined dataset ---
cat > /tmp/surgery_build_dataset.sh << 'BUILDEOF'
#!/bin/bash
#SBATCH --job-name=surgery_build
#SBATCH --partition=batch
#SBATCH --account=${SLURM_ACCOUNT:-your_account}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=slurm-build-%j.out
set -euo pipefail

WORK_DIR="${LANDMARKDIFF_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORK_DIR"
source "${CONDA_PREFIX:-$HOME/miniconda3}"/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "=== Stage 1: Building combined dataset ==="
echo "Start: $(date)"

python scripts/build_training_dataset.py \
    --output data/training_combined \
    --synthetic_dirs data/synthetic_surgery_pairs data/synthetic_surgery_pairs_v2 \
    --v3_dir data/synthetic_surgery_pairs_v3 \
    --real_dir data/real_surgery_pairs/pairs \
    --augment_factor 5

echo "=== Stage 1.5: Creating train/test split ==="
python scripts/create_test_split.py \
    --data_dir data/training_combined \
    --test_dir data/test_pairs \
    --test_fraction 0.05 \
    --seed 42

echo "Dataset build complete: $(date)"
BUILDEOF

echo ""
echo "--- Stage 1: Build Dataset ---"
if $DRY_RUN; then
    echo "  [DRY] sbatch /tmp/surgery_build_dataset.sh"
    BUILD_JOB="12345"
else
    BUILD_JOB=$(sbatch /tmp/surgery_build_dataset.sh | awk '{print $4}')
    echo "  Submitted: job $BUILD_JOB"
fi

# --- Stage 2: Phase A Training ---
cat > /tmp/surgery_phaseA.sh << 'PHASEAEOF'
#!/bin/bash
#SBATCH --job-name=surgery_phaseA
#SBATCH --partition=batch_gpu
#SBATCH --account=${SLURM_ACCOUNT:-your_account}
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=slurm-phaseA-%j.out
set -euo pipefail

WORK_DIR="${LANDMARKDIFF_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORK_DIR"
source "${CONDA_PREFIX:-$HOME/miniconda3}"/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "=== Stage 2: Phase A Training ==="
echo "Start: $(date)"
nvidia-smi

PAIR_COUNT=$(find data/training_combined/ -name "*_input.png" 2>/dev/null | wc -l)
echo "Training pairs: $PAIR_COUNT"

if [ "$PAIR_COUNT" -lt 1000 ]; then
    echo "ERROR: Not enough training pairs ($PAIR_COUNT < 1000)"
    exit 1
fi

python scripts/train_controlnet.py \
    --data_dir data/training_combined \
    --output_dir checkpoints_phaseA \
    --num_train_steps 100000 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --checkpoint_every 10000 \
    --sample_every 5000 \
    --log_every 100 \
    --ema_decay 0.9999 \
    --phase A \
    --seed 42 \
    --wandb_dir "$WORK_DIR"

# Record in experiment lineage
python scripts/experiment_lineage.py record-training \
    --config configs/phaseA_production.yaml \
    --checkpoint checkpoints_phaseA/final \
    --steps 100000 \
    --job-id "${SLURM_JOB_ID:-unknown}" \
    2>/dev/null || echo "Lineage recording skipped"

echo "Phase A complete: $(date)"
PHASEAEOF

echo ""
echo "--- Stage 2: Phase A Training ---"
if $DRY_RUN; then
    echo "  [DRY] sbatch --dependency=afterok:$BUILD_JOB /tmp/surgery_phaseA.sh"
    PHASEA_JOB="12346"
else
    PHASEA_JOB=$(sbatch --dependency=afterok:$BUILD_JOB /tmp/surgery_phaseA.sh | awk '{print $4}')
    echo "  Submitted: job $PHASEA_JOB (depends on $BUILD_JOB)"
fi

# --- Stage 3: Phase A Evaluation ---
cat > /tmp/surgery_eval_phaseA.sh << 'EVALAEOF'
#!/bin/bash
#SBATCH --job-name=surgery_evalA
#SBATCH --partition=batch_gpu
#SBATCH --account=${SLURM_ACCOUNT:-your_account}
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=slurm-evalA-%j.out
set -euo pipefail

WORK_DIR="${LANDMARKDIFF_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORK_DIR"
source "${CONDA_PREFIX:-$HOME/miniconda3}"/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "=== Stage 3: Phase A Evaluation ==="
echo "Start: $(date)"

# Find best checkpoint (use final if exists, otherwise latest)
CKPT="checkpoints_phaseA/final"
if [ ! -d "$CKPT" ]; then
    CKPT=$(ls -d checkpoints_phaseA/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi

echo "Evaluating checkpoint: $CKPT"

python scripts/evaluate_checkpoint.py \
    --checkpoint "$CKPT" \
    --test_dir data/test_pairs \
    --output results/phaseA_eval.json \
    --save_images \
    --images_dir results/phaseA_images \
    --num_steps 25

# Record evaluation in lineage
python scripts/experiment_lineage.py record-eval \
    --checkpoint "$CKPT" \
    --results results/phaseA_eval.json \
    2>/dev/null || echo "Lineage recording skipped"

# Generate paper tables (preliminary)
python scripts/generate_paper_tables.py \
    --results results/phaseA_eval.json \
    --output paper/tables/

echo "Phase A eval complete: $(date)"
EVALAEOF

echo ""
echo "--- Stage 3: Phase A Evaluation ---"
if $DRY_RUN; then
    echo "  [DRY] sbatch --dependency=afterok:$PHASEA_JOB /tmp/surgery_eval_phaseA.sh"
    EVALA_JOB="12347"
else
    EVALA_JOB=$(sbatch --dependency=afterok:$PHASEA_JOB /tmp/surgery_eval_phaseA.sh | awk '{print $4}')
    echo "  Submitted: job $EVALA_JOB (depends on $PHASEA_JOB)"
fi

# --- Stage 4: Phase B Training ---
cat > /tmp/surgery_phaseB.sh << 'PHASEBEOF'
#!/bin/bash
#SBATCH --job-name=surgery_phaseB
#SBATCH --partition=batch_gpu
#SBATCH --account=${SLURM_ACCOUNT:-your_account}
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=slurm-phaseB-%j.out
set -euo pipefail

WORK_DIR="${LANDMARKDIFF_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORK_DIR"
source "${CONDA_PREFIX:-$HOME/miniconda3}"/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "=== Stage 4: Phase B Training ==="
echo "Start: $(date)"
nvidia-smi

# Initialize from Phase A final checkpoint
PHASEA_CKPT="checkpoints_phaseA/final"
if [ ! -d "$PHASEA_CKPT" ]; then
    PHASEA_CKPT=$(ls -d checkpoints_phaseA/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi

echo "Initializing from Phase A: $PHASEA_CKPT"

python scripts/train_controlnet.py \
    --data_dir data/training_combined \
    --output_dir checkpoints_phaseB \
    --num_train_steps 50000 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --checkpoint_every 5000 \
    --sample_every 2500 \
    --log_every 50 \
    --ema_decay 0.9999 \
    --phase B \
    --resume_phaseA "$PHASEA_CKPT" \
    --clinical_augment \
    --seed 42 \
    --wandb_dir "$WORK_DIR"

# Record in experiment lineage
python scripts/experiment_lineage.py record-training \
    --config configs/phaseB_production.yaml \
    --checkpoint checkpoints_phaseB/final \
    --steps 50000 \
    --job-id "${SLURM_JOB_ID:-unknown}" \
    2>/dev/null || echo "Lineage recording skipped"

echo "Phase B complete: $(date)"
PHASEBEOF

echo ""
echo "--- Stage 4: Phase B Training ---"
if $DRY_RUN; then
    echo "  [DRY] sbatch --dependency=afterok:$EVALA_JOB /tmp/surgery_phaseB.sh"
    PHASEB_JOB="12348"
else
    PHASEB_JOB=$(sbatch --dependency=afterok:$EVALA_JOB /tmp/surgery_phaseB.sh | awk '{print $4}')
    echo "  Submitted: job $PHASEB_JOB (depends on $EVALA_JOB)"
fi

# --- Stage 5: Final Evaluation ---
cat > /tmp/surgery_eval_final.sh << 'EVALFEOF'
#!/bin/bash
#SBATCH --job-name=surgery_evalF
#SBATCH --partition=batch_gpu
#SBATCH --account=${SLURM_ACCOUNT:-your_account}
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=slurm-evalF-%j.out
set -euo pipefail

WORK_DIR="${LANDMARKDIFF_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORK_DIR"
source "${CONDA_PREFIX:-$HOME/miniconda3}"/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "=== Stage 5: Final Evaluation ==="
echo "Start: $(date)"

# Evaluate Phase B final
CKPT="checkpoints_phaseB/final"
if [ ! -d "$CKPT" ]; then
    CKPT=$(ls -d checkpoints_phaseB/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi

echo "Evaluating final checkpoint: $CKPT"

python scripts/evaluate_checkpoint.py \
    --checkpoint "$CKPT" \
    --test_dir data/test_pairs \
    --output results/final_eval.json \
    --save_images \
    --images_dir results/final_images \
    --num_steps 25

# Record final evaluation in lineage
python scripts/experiment_lineage.py record-eval \
    --checkpoint "$CKPT" \
    --results results/final_eval.json \
    2>/dev/null || echo "Lineage recording skipped"

# Compute baselines
python scripts/compute_baselines.py \
    --test_dir data/test_pairs \
    --output results/baseline_results.json \
    2>/dev/null || echo "Baseline computation skipped"

# Regenerate displacement report from current data
python scripts/displacement_analysis.py report \
    --pairs_dir data/real_surgery_pairs/pairs \
    --output data/displacement_report.json \
    2>/dev/null || echo "Displacement report generation skipped (using existing)"

# Generate ablation template if not exists
if [ ! -f results/ablation_results.json ]; then
    python scripts/displacement_analysis.py ablation-template \
        --output results/ablation_results.json \
        2>/dev/null || echo "Ablation template skipped"
fi

# Generate final paper tables
python scripts/generate_paper_tables.py \
    --results results/final_eval.json \
    --baselines results/baseline_results.json \
    --ablation results/ablation_results.json \
    --displacement_report data/displacement_report.json \
    --output paper/tables/

# Link paper tables to evaluations
python scripts/experiment_lineage.py link-paper \
    --table "Table1_main_results" \
    --eval-id "$(python -c "
import json, sys
try:
    from scripts.experiment_lineage import LineageDB
    db = LineageDB.load()
    ev = db.get_latest_eval()
    print(ev['id'] if ev else 'unknown')
except: print('unknown')
")" 2>/dev/null || echo "Paper link skipped"

# Check for stale results
python scripts/experiment_lineage.py check-stale \
    2>/dev/null || echo "WARNING: Stale results detected -- review lineage report"

# Generate lineage report
python scripts/experiment_lineage.py report > results/lineage_report.txt \
    2>/dev/null || echo "Lineage report skipped"

echo "=== Pipeline Complete ==="
echo "Results: results/"
echo "Tables:  paper/tables/"
echo "Lineage: results/lineage_report.txt"
echo "End: $(date)"
EVALFEOF

echo ""
echo "--- Stage 5: Final Evaluation ---"
if $DRY_RUN; then
    echo "  [DRY] sbatch --dependency=afterok:$PHASEB_JOB /tmp/surgery_eval_final.sh"
    EVALF_JOB="12349"
else
    EVALF_JOB=$(sbatch --dependency=afterok:$PHASEB_JOB /tmp/surgery_eval_final.sh | awk '{print $4}')
    echo "  Submitted: job $EVALF_JOB (depends on $PHASEB_JOB)"
fi

echo ""
echo "=============================================="
echo "  Pipeline Submitted!"
echo "  Build:    $BUILD_JOB"
echo "  Phase A:  $PHASEA_JOB (after $BUILD_JOB)"
echo "  Eval A:   $EVALA_JOB (after $PHASEA_JOB)"
echo "  Phase B:  $PHASEB_JOB (after $EVALA_JOB)"
echo "  Eval F:   $EVALF_JOB (after $PHASEB_JOB)"
echo ""
echo "  Monitor: squeue -u $USER -n surgery_build,surgery_phaseA,surgery_evalA,surgery_phaseB,surgery_evalF"
echo "=============================================="
