#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --mem=8GB
#SBATCH --time=100:00:00
#SBATCH --partition=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Nicholas.Li@uhnresearch.ca
#SBATCH --job-name=SRTD3
#SBATCH --output=SRTD3.txt

source ~/anaconda3/bin/activate
conda activate TranSynergy
cd ..

python Code/SRTD.py 3 ${SLURM_ARRAY_TASK_ID}
