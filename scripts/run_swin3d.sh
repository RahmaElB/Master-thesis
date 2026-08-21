#!/bin/bash
#SBATCH --job-name=swin3d
#SBATCH --account=project_2018481
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-swin3d-%j.out

module --force purge

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export ECHO_DATA_ROOT=/scratch/project_2018481/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/thesis_project6
export PYTHONPATH=$ECHO_PROJECT_ROOT
export APPTAINER_CACHEDIR=$TMPDIR

SIF=$ECHO_PROJECT_ROOT/containers/pytorch_2.10_cuda13_roihu.sif

cd $ECHO_PROJECT_ROOT

apptainer exec --nv --bind /scratch/project_2018481:/scratch/project_2018481 \
    $SIF \
    python3 -m src.train \
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
