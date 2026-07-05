#!/bin/bash
#SBATCH --job-name=echo_eda
#SBATCH --account=project_2018481
#SBATCH --partition=small
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-eda-%j.out

module purge
module load pytorch

export ECHO_DATA_ROOT=/scratch/project_2018481/relbouaz/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/relbouaz/thesis_project6

cd $ECHO_PROJECT_ROOT

python -m src.data.eda \
    --csv_path $ECHO_DATA_ROOT/FileList.csv \
    --out_dir results/eda
