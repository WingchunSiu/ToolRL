#!/bin/bash

#SBATCH --job-name=grpo-qwen-training  # Give your job a name
#SBATCH --account=beaa-delta-gpu      # Specify your account (REQUIRED for Delta)
#SBATCH --nodes=1                      # Request one node
#SBATCH --ntasks-per-node=1            # Request one task per node
#SBATCH --cpus-per-task=8              # Adjust as needed for data processing
#SBATCH --mem=64G                      # Adjust as needed
#SBATCH --gres=gpu:2                   # Request 2 GPUs
#SBATCH --time=04:00:00                # Set a time limit for the job (e.g., 4 hours)
#SBATCH --partition=gpuA100x4          # Or the specific GPU partition you have access to on Delta
#SBATCH --output=grpo_%j.out           # Standard output log
#SBATCH --error=grpo_%j.err            # Standard error log

# Load any necessary modules (check Delta's documentation for what you might need)
# module load cuda/ ... 
# module load python/ ...

# Set your environment variables
export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS=2
export ROLLOUT_TP_SIZE=1
export VLLM_ATTENTION_BACKEND=PYTORCH_DEFAULT

# All the env variables below are set to 0 by default
export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0

export DATA_DIR="./dataset/rlla_4k"
export BASE_MODEL="Qwen/Qwen2.5-3b-Instruct"
export EXPERIMENT_NAME="grpo-qwen2.5-3b"

# Execute your training script
bash ./examples/grpo_trainer/run_grpo.sh