#!/bin/bash
#SBATCH --job-name=wind_train
#SBATCH --time=04:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --output=ml/logs/train_%j.out
#SBATCH --error=ml/logs/train_%j.err

set -euo pipefail

# Modules (MATCH TF CUDA build = 12.2)
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0

export CUDA_HOME="${EBROOTCUDA}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"

# Help the dynamic loader find CUDA/cuDNN libs
export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${EBROOTCUDNN}/lib64:${LD_LIBRARY_PATH:-}"

# Go to repo root (venv is here now)
REPO="/home2/s5549329/ForecatsingWindPower"
cd "${REPO}" || exit 1

# Ensure logs dir exists (path is relative to repo root)
mkdir -p ml/logs

echo "=============================="
echo "Module list:"
module list || true
echo "which python (before venv):"
which python || true
python -V || true
echo "=============================="

echo "=============================="
echo "GPU ENV:"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "nvidia-smi:"
nvidia-smi || true
echo "=============================="

# Activate venv from repo root
source .venv/bin/activate

echo "which python (after venv):"
which python || true
python -V || true

# (Optional) keep tooling updated; do NOT reinstall TF each job unless needed
python -m pip install --upgrade pip setuptools wheel

# TF / GPU diagnostics
python - <<'PY'
import tensorflow as tf
info = tf.sysconfig.get_build_info()
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("CUDA build:", info.get("cuda_version"))
print("cuDNN build:", info.get("cudnn_version"))
PY

# Run from ml/ so imports like "src...." resolve cleanly
cd ml

# Train
python -m src.models.train_all_arhitectures

# Rank
python -m src.models.rank_runs

deactivate
