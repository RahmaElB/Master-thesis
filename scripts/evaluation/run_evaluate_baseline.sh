#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --account=project_2018481
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-eval-baseline-%j.out

module --force purge

export ECHO_DATA_ROOT=/scratch/project_2018481/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/thesis_project6
export PYTHONPATH=$ECHO_PROJECT_ROOT
export APPTAINER_CACHEDIR=$TMPDIR

SIF=$ECHO_PROJECT_ROOT/containers/pytorch_2.10_cuda13_roihu.sif

cd $ECHO_PROJECT_ROOT

apptainer exec --nv --bind /scratch/project_2018481:/scratch/project_2018481 \
    $SIF \
    python3 -m src.evaluate --model baseline \
        --checkpoint checkpoints/baseline_64f_112px_clipp1_pretrained_best.pt \
        --num_frames 64 --img_size 112 --temporal_sampling clip --clip_period 1 \
        --batch_size 8 --epochs 15 --lr 1e-4 --weight_decay 1e-4 --dropout 0.3 \
        --split test --out_dir results/evaluation
