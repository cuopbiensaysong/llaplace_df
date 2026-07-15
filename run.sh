#!/bin/bash
#SBATCH --job-name=physonet       # Job name
#SBATCH --output=slurm/naao_uk.txt      # Output file
#SBATCH --error=slurm/naao_uk_e.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=80G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node


python finetuning/tune.py --dataset-key noaa_uk --preds 168 --arms d a --run-tag v1