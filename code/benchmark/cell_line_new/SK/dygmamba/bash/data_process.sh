#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=dataread
#SBATCH --output=/home/wuyan/dygmamba_project/model/0log/print_%j.out
#SBATCH --nodes=1                     ##指定使用1个节点数
#SBATCH -n 1                          ##指定总任务数1
#SBATCH --qos=a100g1                  ##指定qos为a100g1

#SBATCH -p a100                       

set -e

CELL_TYPE="SK"


echo "Process cell $CELL_TYPE"

source /home/wuyan/miniconda3/etc/profile.d/conda.sh

conda activate /home/wuyan/conda_env/singlecellpreprocess

CODE_PATH="/home/wuyan/dygmamba_project/model/cell_line/$CELL_TYPE/dygmamba/src"

#########################################################################
#==================== Data Read
printf '*%.0s' {1..50}; echo
echo "Data read"

cd $CODE_PATH

python -u preprocess_data.py

#########################################################################
#==================== Data Process

printf '*%.0s' {1..50}; echo
echo "Trajectory inference"

conda activate /home/wuyan/conda_env/R43_env

cd $CODE_PATH

Rscript trajectory_inference.R

echo "run finished $CELL_TYPE data process job"

printf '*%.0s' {1..50}; echo

#########################################################################
#==================== DyGMamba Process

echo "run DYGMAMBA $CELL_TYPE data job"

printf '*%.0s' {1..50}; echo

conda activate /home/wuyan/conda_env/singlecellpreprocess

cd $CODE_PATH

python -u dygmamba_preprocess.py

