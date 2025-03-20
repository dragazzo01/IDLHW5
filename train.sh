#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH -t 05:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --output=/ocean/projects/cis250019p/ramireza/IDLHW5/slurm_out/train-%j.out


hostname
echo "job starting"
module load AI/pytorch_23.02-1.13.1-py3
cd /ocean/projects/cis250019p/ramireza/IDLHW5
pip install -r requirements.txt

python train.py

squeue -j $SLURM_JOBID
module unload AI/pytorch_23.02-1.13.1-py3
echo "job finished"