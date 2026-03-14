#!/bin/bash
# Monitor dataset build progress and submit Phase A training when ready.
#
# Run on the head node (not via sbatch):
#   nohup bash scripts/submit_when_ready.sh > submit_monitor.log 2>&1 &
#
# This script:
# 1. Waits for build_training_dataset.py to finish
# 2. Verifies the dataset
# 3. Creates the test split
# 4. Submits Phase A training via sbatch

set -euo pipefail

WORK_DIR="${LANDMARKDIFF_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORK_DIR"

source "${CONDA_PREFIX:-$HOME/miniconda3}"/etc/profile.d/conda.sh
conda activate landmarkdiff

MIN_PAIRS=50000  # Minimum pairs before training is worthwhile

echo "$(date): Monitoring dataset build..."
echo "  Minimum pairs required: $MIN_PAIRS"

# Wait for build process to finish
while pgrep -f "build_training_dataset.py" > /dev/null 2>&1; do
    CURRENT=$(find data/training_combined/ -name "*_input.png" 2>/dev/null | wc -l)
    echo "$(date): Build in progress... $CURRENT pairs"
    sleep 300  # Check every 5 minutes
done

FINAL_COUNT=$(find data/training_combined/ -name "*_input.png" 2>/dev/null | wc -l)
echo "$(date): Build complete! $FINAL_COUNT pairs"

if [ "$FINAL_COUNT" -lt "$MIN_PAIRS" ]; then
    echo "ERROR: Only $FINAL_COUNT pairs (need at least $MIN_PAIRS)"
    exit 1
fi

# Verify dataset
echo "$(date): Verifying dataset..."
python scripts/verify_dataset.py --data_dir data/training_combined --sample 500

# Create test split
echo "$(date): Creating test split..."
python scripts/create_test_split.py \
    --data_dir data/training_combined \
    --test_dir data/test_pairs \
    --test_fraction 0.05 \
    --seed 42

# Submit Phase A training
echo "$(date): Submitting Phase A training..."
JOB_ID=$(sbatch --parsable scripts/train_phaseA_slurm.sh)
echo "$(date): Phase A training submitted as job $JOB_ID"

# Submit evaluation (depends on training)
echo "$(date): Submitting evaluation (dependency: afterok:$JOB_ID)..."
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$JOB_ID scripts/eval_slurm.sh)
echo "$(date): Evaluation submitted as job $EVAL_JOB"

echo ""
echo "=============================================="
echo "  Pipeline Submitted Successfully"
echo "=============================================="
echo "  Dataset: $FINAL_COUNT pairs"
echo "  Train job: $JOB_ID"
echo "  Eval job: $EVAL_JOB (runs after training)"
echo "=============================================="
