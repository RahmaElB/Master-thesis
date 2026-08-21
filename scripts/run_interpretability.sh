#!/bin/bash
#SBATCH --job-name=echo_interp
#SBATCH --account=project_2018481
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-interp-%j.out

module --force purge

export ECHO_DATA_ROOT=/scratch/project_2018481/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/thesis_project6
export PYTHONPATH=$ECHO_PROJECT_ROOT
export APPTAINER_CACHEDIR=$TMPDIR

SIF=$ECHO_PROJECT_ROOT/containers/pytorch_2.10_cuda13_roihu.sif

cd $ECHO_PROJECT_ROOT

MODEL=swin3d
CHECKPOINT=$ECHO_PROJECT_ROOT/checkpoints/swin3d_32f_224px_clipp2_pretrained_best.pt

apptainer exec --nv --bind /scratch/project_2018481:/scratch/project_2018481 \
    $SIF \
    python3 -m src.interpret.run_interpretability \
        --model $MODEL \
        --checkpoint $CHECKPOINT \
        --num_frames 32 \
        --img_size 224 \
        --temporal_sampling clip \
        --clip_period 2 \
        --num_examples 8 \
        --near_boundary_width 5.0 \
        --out_dir results/interpretability
