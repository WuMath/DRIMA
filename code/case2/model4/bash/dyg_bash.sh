#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=dygTrain
#SBATCH --output=/home/wuyan/dygmamba_project/NewRealPlan/log/print_%j.out
#SBATCH --nodes=1                     ##指定使用1个节点数
#SBATCH -n 1                          ##指定总任务数1
#SBATCH --qos=a100g2                  ##指定qos为a100g1
#SBATCH --gres=gpu:a100:1             ##指定使用gpu资源,a100数量1
#SBATCH -p a100                       
#SBATCH --mem=32G   ## 为该节点申请 32GB 的物理内存
set -e


source /home/wuyan/miniconda3/etc/profile.d/conda.sh

conda activate /home/liyang/BioWuYan/conda_env/dygmamba39

CELL_TYPE="model4"

CODE_PATH="/home/wuyan/dygmamba_project/NewRealPlan/case2/code/$CELL_TYPE/src"


printf '*%.0s' {1..50}; echo

echo "run DYGMAMBA $CELL_TYPE job"

printf '*%.0s' {1..50}; echo



cd $CODE_PATH

python -u main_train_hpc.py





