#!/bin/bash
#SBATCH --job-name=r3d
#SBATCH --account=project_2018481
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-r3d-%j.out

module --force purge

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export ECHO_DATA_ROOT=/scratch/project_2018481/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/thesis_project6
export PYTHONPATH=$ECHO_PROJECT_ROOT
export APPTAINER_CACHEDIR=$TMPDIR

SIF=$ECHO_PROJECT_ROOT/containers/pytorch_2.10_cuda13_roihu.sif

cd $ECHO_PROJECT_ROOT

# NOTE: dropped --backbone_lr here, that differential LR was the Swin3D
# collapse fix specifically. R3D was never collapsing, so splitting its LR
# just let the head overfit faster (val_auc peaked at epoch 6 both times I
# ran it with backbone_lr=1e-5, then just got worse for 9 more epochs).
# Also cut epochs 15 -> 8 since the best checkpoint has shown up around
# epoch 5-6 in every run so far, no point burning GPU hours past that.
apptainer exec --nv --bind /scratch/project_2018481:/scratch/project_2018481 \
    $SIF \
    python3 -m src.train \
        --model r3d \
        --num_frames 64 \
        --img_size 112 \
        --batch_size 4 \
        --epochs 12 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --dropout 0.3 \
        --temporal_sampling clip \
        --clip_period 1 \
        --grad_clip 1.0 \
        --use_wandb
