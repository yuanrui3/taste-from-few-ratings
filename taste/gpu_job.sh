#!/bin/bash
# One GPU job for the taste model on a Slurm cluster.
#
# Usage, from your project folder on the cluster (it holds code/, photos/, cut/, venv/ ...):
#   sbatch -A <your allocation> -p <a GPU partition> code/taste/gpu_job.sh code/taste/<script>.py [args...]
# Set TASTE_ROOT to run elsewhere than the submit folder, TASTE_VENV for another Python env.
# Small on purpose: one GPU, a few hours. Pass your allocation (-A) and partition (-p) to sbatch.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J taste
#SBATCH -o logs/%x-%j.log
set -euo pipefail
R="${TASTE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$R"
mkdir -p logs
export HF_HOME="${HF_HOME:-$R/models/hf_home}" TORCH_HOME="${TORCH_HOME:-$R/models/torch_home}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1  # fetch weights on the login node first (--prefetch); compute nodes may lack internet
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${TASTE_VENV:-$R/venv}/bin/python" "$@"
