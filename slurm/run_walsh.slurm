#!/bin/bash
#SBATCH --job-name=walshflow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=walshflow-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Replace this with the activation command for your HPC.
# source "$HOME/miniforge3/etc/profile.d/conda.sh"
# conda activate walshflow

CONFIG_PATH="${1:-examples/water/water_angle.toml}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
walshflow run "$CONFIG_PATH"
