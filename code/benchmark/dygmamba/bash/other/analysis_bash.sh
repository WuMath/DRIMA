#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=traindygmamba
#SBATCH --output=/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/log/print_%j.out
#SBATCH --nodes=1                     ##指定使用1个节点数
#SBATCH -n 1                          ##指定总任务数1
#SBATCH --qos=a100g1                  ##指定qos为a100g1
#SBATCH --gres=gpu:a100:1             ##指定使用gpu资源,a100数量1
#SBATCH -p a100                       

set -e

source /home/liyang/miniconda3/etc/profile.d/conda.sh

# conda activate /home/liyang/BioWuYan/conda_env/singlecellpreprocess

conda activate /home/liyang/BioWuYan/conda_env/dygmamba39

echo "run job"

# python -u main_data_preprocess.py

# python -u analysis/assess.py

# python -u benchmark/benchmark_process.py

cd /home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src/models

python -u main_train_hpc.py





