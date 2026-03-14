#!/bin/bash
# Submit all ablation experiments
# Usage: bash scripts/ablation/submit_all.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting ablation experiments..."
JOB_DIFFUSION_ONLY=$(sbatch --parsable "$SCRIPT_DIR/ablation_diffusion_only.sh")
echo "  diffusion_only: $JOB_DIFFUSION_ONLY"
JOB_DIFF_IDENTITY=$(sbatch --parsable "$SCRIPT_DIR/ablation_diff_identity.sh")
echo "  diff_identity: $JOB_DIFF_IDENTITY"
JOB_DIFF_PERCEPTUAL=$(sbatch --parsable "$SCRIPT_DIR/ablation_diff_perceptual.sh")
echo "  diff_perceptual: $JOB_DIFF_PERCEPTUAL"
JOB_FULL=$(sbatch --parsable "$SCRIPT_DIR/ablation_full.sh")
echo "  full: $JOB_FULL"

echo "All ablation jobs submitted."
