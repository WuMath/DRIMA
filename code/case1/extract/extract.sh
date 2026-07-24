#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH  --job-name=extract    ##任务名
#SBATCH --output=/home/wuyan/dygmamba_project/NewRealPlan/log/print_%j.out
#SBATCH  --nodes=1                     ##指定使用1个节点数
#SBATCH  -n 1                          ##指定总任务数1
#SBATCH  --qos=a100g2                  ##指定qos为a100g1
#SBATCH  -p a100      

set -eo pipefail

source /home/wuyan/miniconda3/etc/profile.d/conda.sh

conda activate /home/wuyan/conda_env/dygmamba39

cd "/home/wuyan/dygmamba_project/NewRealPlan/Case1/code/ananlysis/extract"

python -u tractory_extract_V1.py