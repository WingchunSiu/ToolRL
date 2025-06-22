#!/bin/bash
#SBATCH --job-name=simple-sft-rlla
#SBATCH --account=beaa-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/simple_sft_%j.out
#SBATCH --error=logs/simple_sft_%j.err

echo "Starting Simple SFT training on RLLA dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Load modules
module purge
module load anaconda3_gpu
module list

# Navigate to project directory
cd /u/siu1/ToolRL

# Activate virtual environment
source venv/bin/activate

# Verify activation
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Set environment variables
export PYTHONPATH=/u/siu1/ToolRL:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

# Show GPU info before training
echo "GPU information:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Run simple SFT training
echo "Starting simple SFT training..."
python simple_sft_trainer.py

echo "Simple SFT training completed!"

# Show final GPU usage
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# List the saved checkpoint
echo "Saved checkpoint:"
ls -la checkpoints/simple_sft_rlla/ 