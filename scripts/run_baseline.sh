#!/bin/bash
#SBATCH --job-name=echonet_base
#SBATCH --account=project_2018481
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-baseline-%j.out

# Roihu: no plain "pytorch" module exists (confirmed via module spider on a
# GPU node) - PyTorch is only available as CSC's Apptainer container. Also,
# CPU-node (x86_64) and GPU-node (ARM/aarch64) Python environments are not
# interchangeable, so we force-purge any modules that may have leaked in
# from the submitting shell before doing anything else.
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
