#!/bin/bash
#SBATCH --job-name=echo_interp
#SBATCH --account=project_2018481
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-interp-%j.out

module purge
module load pytorch

export ECHO_DATA_ROOT=/scratch/project_2018481/relbouaz/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/relbouaz/thesis_project6
export PYTHONPATH=$ECHO_PROJECT_ROOT

cd $ECHO_PROJECT_ROOT

MODEL=swin3d
CHECKPOINT=$ECHO_PROJECT_ROOT/checkpoints/swin3d_32f_224px_clipp2_pretrained_best.pt

python -m src.interpret.run_interpretability \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --num_frames 32 \
    --img_size 224 \
    --temporal_sampling clip \
    --clip_period 2 \
    --num_examples 8 \
    --near_boundary_width 5.0 \
    --out_dir results/interpretability