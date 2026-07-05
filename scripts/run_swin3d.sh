#!/bin/bash
#SBATCH --job-name=swin3d
#SBATCH --account=project_2018481
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-swin3d-%j.out

module purge
module load pytorch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export ECHO_DATA_ROOT=/scratch/project_2018481/relbouaz/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/relbouaz/thesis_project6
export PYTHONPATH=$ECHO_PROJECT_ROOT

cd $ECHO_PROJECT_ROOT

python -m src.train \
    --model swin3d \
    --num_frames 32 \
    --img_size 224 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --epochs 15 \
    --lr 1e-3 \
    --backbone_lr 1e-5 \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --freeze_epochs 2 \
    --warmup_frac 0.1 \
    --grad_clip 1.0 \
    --temporal_sampling clip \
    --clip_period 2 \
    --use_wandb