#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=dygData
#SBATCH --output=/home/liyang/BioWuYan/dygmamba_project/model/cell_line/0log/print_%j.out
#SBATCH --nodes=1                     ##指定使用1个节点数
#SBATCH -n 1                          ##指定总任务数1
#SBATCH --qos=a100g1                  ##指定qos为a100g1
#SBATCH --gres=gpu:a100:1             ##指定使用gpu资源,a100数量1
#SBATCH -p a100                       

set -e


source /home/liyang/miniconda3/etc/profile.d/conda.sh

# conda activate /home/liyang/BioWuYan/conda_env/singlecellpreprocess
conda activate /home/liyang/BioWuYan/conda_env/dygmamba39

export LD_LIBRARY_PATH=/home/liyang/BioWuYan/conda_env/dygmamba39/lib:$LD_LIBRARY_PATH

# 同时屏蔽 cuda-11.7 的路径
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v 'cuda-11.7' | tr '\n' ':')


CELL_TYPE="HCT116"

printf '*%.0s' {1..50}; echo

echo "run DYGMAMBA $CELL_TYPE job"

printf '*%.0s' {1..50}; echo

cd /home/liyang/BioWuYan/dygmamba_project/model/cell_line/$CELL_TYPE/dygmamba/src

# python -u dygmamba_preprocess.py

python -u models/main_train_hpc.py





