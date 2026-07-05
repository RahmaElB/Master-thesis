#!/bin/bash
#SBATCH --job-name=r3d
#SBATCH --account=project_2018481
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-r3d-%j.out

module purge
module load pytorch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export ECHO_DATA_ROOT=/scratch/project_2018481/relbouaz/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/relbouaz/thesis_project6

# Tell Python where the project root is
export PYTHONPATH=$ECHO_PROJECT_ROOT

cd $ECHO_PROJECT_ROOT

# NOTE: dropped --backbone_lr here, that differential LR was the Swin3D
# collapse fix specifically. R3D was never collapsing, so splitting its LR
# just let the head overfit faster (val_auc peaked at epoch 6 both times I
# ran it with backbone_lr=1e-5, then just got worse for 9 more epochs).
# Also cut epochs 15 -> 8 since the best checkpoint has shown up around
# epoch 5-6 in every run so far, no point burning GPU hours past that.
python -m src.train \
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