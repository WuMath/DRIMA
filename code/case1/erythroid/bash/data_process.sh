#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=erythroid
#SBATCH --output=/home/wuyan/dygmamba_project/NewRealPlan/log/print_%j.out
#SBATCH --nodes=1                     ##指定使用1个节点数
#SBATCH -n 1                          ##指定总任务数1
#SBATCH --qos=a100g2                  ##指定qos为a100g1

#SBATCH -p a100                      

set -e

CELL_TYPE="erythroid"


echo "Process cell $CELL_TYPE"

source /home/wuyan/miniconda3/etc/profile.d/conda.sh

CODE_PATH="/home/wuyan/dygmamba_project/NewRealPlan/Case1/code/ananlysis/$CELL_TYPE/src"


#########################################################################
#==================== Data Process
# Step : 第一次数据处理
#########################################################################

conda activate /home/wuyan/conda_env/singlecellpreprocess

cd $CODE_PATH

printf '*%.0s' {1..50}; echo
echo "preprocess data"

python -u preprocess_data.py

#########################################################################
#==================== Trajectory inference

# Step : 第一次轨迹推断
#########################################################################
printf '*%.0s' {1..50}; echo
echo "Trajectory inference"

conda activate /home/wuyan/conda_env/R43_env

cd $CODE_PATH

Rscript trajectory_inference.R

echo "run finished $CELL_TYPE data process job"

printf '*%.0s' {1..50}; echo

#########################################################################
# Step : 按伪时序等间隔采样 + 重新 UMAP + 重新导出 R
#########################################################################
printf '*%.0s' {1..50}; echo
echo "按伪时序等间隔采样 "

conda activate /home/wuyan/conda_env/singlecellpreprocess
cd $CODE_PATH

python -u downsample.py

#########################################################################
# Step : 第二次轨迹推断（在采样子集上重新跑 Slingshot）
#########################################################################
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


