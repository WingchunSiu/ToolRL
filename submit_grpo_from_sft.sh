#!/bin/bash
#SBATCH --job-name=grpo-from-sft-qwen-1.5b
#SBATCH --account=beaa-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/grpo_from_sft_%j.out
#SBATCH --error=logs/grpo_from_sft_%j.err

echo "Starting GRPO training from SFT checkpoint"
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

# Set experiment name for reward function
export EXPERIMENT_NAME=grpo-from-sft-qwen2.5-1.5b-instruct-rlla4k

# Enable contribution-enhanced reward design
export ENABLE_CONTRIBUTION=1
export CONTRIBUTION_BETA=0.5  # Weight for contribution reward
export WITHLENGTH=0          # Disable length reward for now
export CORRECTMAX1=0         # Use full correctness reward range (-3 to 3)

# Set SFT checkpoint path (from simple SFT trainer)
SFT_CHECKPOINT_DIR="/u/siu1/ToolRL/checkpoints/simple_sft_rlla"
if [ -d "$SFT_CHECKPOINT_DIR" ]; then
    echo "✅ Found SFT checkpoint: $SFT_CHECKPOINT_DIR"
    MODEL_PATH="$SFT_CHECKPOINT_DIR"
else
    echo "❌ SFT checkpoint directory not found: $SFT_CHECKPOINT_DIR"
    echo "Please run SFT training first!"
    exit 1
fi

# Show GPU info before training
echo "GPU information:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Run GRPO training from SFT checkpoint
echo "Starting GRPO training from SFT checkpoint..."
echo "Using model: $MODEL_PATH"

python -m verl.trainer.main_ppo \
    --config-path=/u/siu1/ToolRL/verl/trainer/config \
    --config-name=ppo_trainer \
    data.train_files=/u/siu1/ToolRL/dataset/rlla_4k/train.parquet \
    data.val_files=/u/siu1/ToolRL/dataset/rlla_4k/test.parquet \
    data.prompt_key=prompt \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.train_batch_size=256 \
    data.val_batch_size=64 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    critic.model.path="$MODEL_PATH" \
    trainer.total_epochs=1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.project_name=verl_examples \
    trainer.experiment_name=grpo-from-sft-qwen2.5-1.5b-instruct-rlla4k \
    trainer.logger='[console]' \
    trainer.save_freq=5 \
    trainer.test_freq=1

echo "GRPO training from SFT completed!"

# Show final GPU usage
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 