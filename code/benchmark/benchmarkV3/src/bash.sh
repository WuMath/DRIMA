#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=dataread
#SBATCH --output=/home/wuyan/dygmamba_project/model/0log/print_%j.out
#SBATCH --nodes=1                     ##指定使用1个节点数
#SBATCH -n 1                          ##指定总任务数1
#SBATCH --qos=a100g1                  ##指定qos为a100g1

#SBATCH -p a100                       

set -e

source /home/wuyan/miniconda3/etc/profile.d/conda.sh

conda activate /home/wuyan/conda_env/dygmamba39

cd "/home/wuyan/dygmamba_project/model/benchmarkV3/src"

python -u analysis_step1.py

printf '*%.0s' {1..50}; echo
echo "Step 2"

# python -u analysis_step2.py