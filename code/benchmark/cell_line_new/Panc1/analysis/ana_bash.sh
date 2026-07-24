#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=benchmark       ##作业名称
#SBATCH --output=/home/liyang/BioWuYan/dygmamba_project/model/0log/print_%j.out
#SBATCH  --nodes=1                     ##指定使用1个节点数
#SBATCH  -n 1                          ##指定总任务数1
#SBATCH  --qos=a100g1                  ##指定qos为a100g1
#SBATCH  -p a100                       
#SBATCH --mem=200G 

set -e

CELL_TYPE="Panc1"

printf '*%.0s' {1..50}; echo

echo "run CellOracle $CELL_TYPE job"

printf '*%.0s' {1..50}; echo

source /home/liyang/miniconda3/etc/profile.d/conda.sh

conda activate /home/liyang/BioWuYan/conda_env/dygmamba39

cd "/home/liyang/BioWuYan/dygmamba_project/model/cell_line/$CELL_TYPE/analysis"

python -u Panc1_benchmark.py