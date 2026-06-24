#!/bin/bash
#SBATCH --job-name=physonet       # Job name
#SBATCH --output=slurm/log.txt      # Output file
#SBATCH --error=slurm/physonet_e.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=80G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

llapdiff-train --dataset-key physionet --preds 12 --verbose --summary-json ldt/results/physionet_pred12.json

llapdiff-train --dataset-key bms_air --preds 168 --verbose --summary-json ldt/results/bms_air_pred168.json
llapdiff-train --dataset-key uci_air --preds 168 --verbose --summary-json ldt/results/uci_air_pred168.json
llapdiff-train --dataset-key noaa_us --preds 168 --verbose --summary-json ldt/results/noaa_us_pred168.json
llapdiff-train --dataset-key noaa_uk --preds 168 --verbose --summary-json ldt/results/noaa_uk_pred168.json
llapdiff-train --dataset-key us_equity --preds 100 --verbose --summary-json ldt/results/us_equity_pred100.json
llapdiff-train --dataset-key crypto --preds 100 --verbose --summary-json ldt/results/crypto_pred100.json
