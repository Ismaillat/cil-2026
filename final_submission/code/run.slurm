#!/bin/bash
#SBATCH --account=cil_jobs
#SBATCH --job-name=cil
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodelist=studgpu-node[05,13,21,29]
#SBATCH --mem=32G

module load cuda/12.6.0
export PYTHONUNBUFFERED=1

cd ~/cil
/cluster/courses/cil/envs/envs/text-classification/bin/python -u "$@"
