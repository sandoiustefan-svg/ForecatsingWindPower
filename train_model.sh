#!/bin/bash
#SBATCH --job-name=wind_train
#SBATCH --time=04:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --output=ml/logs/train_%j.out
#SBATCH --error=ml/logs/train_%j.err

# Modules
module --force purge

# This usually enables the correct module tree
module load StdEnv

# If cache is stale
module --ignore_cache load Python/3.10.4-GCCcore-11.3.0
module --ignore_cache load CUDA/11.8.0
module --ignore_cache load cuDNN/8.7.0.84-CUDA-11.8.0

export CUDA_HOME=$EBROOTCUDA
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

# Go to repo -> ml
cd /home2/s5549329/ForecatsingWindPower || exit 1
cd ml || exit 1
mkdir -p logs

echo "=============================="
echo "Module list:"
module list || true
echo "which python:"
which python || true
python -V || true
echo "=============================="

# venv in ml/
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate

# Always ensure pip belongs to this venv
python -m pip install --upgrade pip setuptools wheel

# Install deps
python -m pip install -r requirements.txt

# TF/GPU info
python -c "
import tensorflow as tf
print('TF:', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
"

# Train
python -m src.models.train_all_arhitectures

deactivate
