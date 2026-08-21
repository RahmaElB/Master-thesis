#!/bin/bash
#SBATCH --job-name=echo_eda
#SBATCH --account=project_2018481
#SBATCH --partition=small
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-eda-%j.out

# EDA only needs pandas/numpy/matplotlib - no GPU, no torch - so this runs
# on Roihu's CPU (x86_64) partition using the plain python-data module
# (confirmed working: pandas 2.3.3), rather than paying for a GPU allocation
# or dealing with the arm64 Apptainer container for a job that never
# touches the GPU.
module --force purge
module load python-data/3.12-31.03

export ECHO_DATA_ROOT=/scratch/project_2018481/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/thesis_project6
export PYTHONPATH=$ECHO_PROJECT_ROOT

cd $ECHO_PROJECT_ROOT

python3 -m src.data.eda \
    --csv_path $ECHO_DATA_ROOT/FileList.csv \
    --out_dir results/eda
