#!/bin/bash
#SBATCH --job-name=cnn_lstm
#SBATCH --account=project_2018481
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-cnn-lstm-%j.out

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
        --model cnn_lstm \
        --num_frames 64 \
        --img_size 112 \
        --batch_size 4 \
        --epochs 15 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --dropout 0.3 \
        --lstm_hidden_size 256 \
        --temporal_sampling clip \
        --clip_period 1 \
        --use_wandb
