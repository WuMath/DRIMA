#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=GRaNIE
#SBATCH --output=/home/liyang/BioWuYan/dygmamba_project/model/cell_line/0log/print_%j.out
#SBATCH --nodes=1                     ##指定使用1个节点数
#SBATCH -n 1                          ##指定总任务数1
#SBATCH --qos=a100g1                  ##指定qos为a100g1
#SBATCH --gres=gpu:a100:1             ##指定使用gpu资源,a100数量1
#SBATCH -p a100                       

set -e

CELL_TYPE="HCT116"

printf '*%.0s' {1..50}; echo

echo "run GRaNIE $CELL_TYPE job"

printf '*%.0s' {1..50}; echo

source /home/liyang/miniconda3/etc/profile.d/conda.sh

conda activate /home/liyang/BioWuYan/conda_env/GRaNIE

cd /home/liyang/BioWuYan/dygmamba_project/model/cell_line/$CELL_TYPE/GRaNIE

Rscript granie.R