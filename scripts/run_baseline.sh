#!/bin/bash
#SBATCH --job-name=echonet_base
#SBATCH --account=project_2018481
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-baseline-%j.out

module purge
module load pytorch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export ECHO_DATA_ROOT=/scratch/project_2018481/relbouaz/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/relbouaz/thesis_project6

# Make the src package visible to Python
export PYTHONPATH=$ECHO_PROJECT_ROOT

cd $ECHO_PROJECT_ROOT

python -m src.train \
    --model baseline \
    --num_frames 64 \
    --img_size 112 \
    --batch_size 8 \
    --epochs 15 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --temporal_sampling clip \
    --clip_period 1 \
    --use_wandb