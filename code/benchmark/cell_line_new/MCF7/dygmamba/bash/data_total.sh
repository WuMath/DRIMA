#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=dataread
#SBATCH --output=/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/log/print_%j.out
#SBATCH --nodes=1                     ##指定使用1个节点数
#SBATCH -n 1                          ##指定总任务数1
#SBATCH --qos=a100g1                  ##指定qos为a100g1
#SBATCH --gres=gpu:a100:1             ##指定使用gpu资源,a100数量1
#SBATCH -p a100                       
#SBATCH --mem=64G

set -e

CELL_TYPE="MCF7"

printf '*%.0s' {1..50}; echo

echo "run DYGMAMBA $CELL_TYPE job"

printf '*%.0s' {1..50}; echo

source /home/liyang/miniconda3/etc/profile.d/conda.sh

conda activate /home/liyang/BioWuYan/conda_env/singlecellpreprocess

CODE_PATH="/home/liyang/BioWuYan/dygmamba_project/model/cell_line/$CELL_TYPE/dygmamba/src"

COMMON_DIR="/home/liyang/BioWuYan/dygmamba_project/data/jaspar"

OUTPUT_DIR="/home/liyang/BioWuYan/dygmamba_project/data/cell_line/$CELL_TYPE/process"

mkdir -p $OUTPUT_DIR
# #########################################################################
# #==================== Data Read
# printf '*%.0s' {1..50}; echo
# echo "Data read"

# cd $CODE_PATH

# python -u main_data_read.py

# #########################################################################
# #===================== JASPAR

# printf '*%.0s' {1..50}; echo
# echo "JASPAR process"

# cd $OUTPUT_DIR

# bedtools getfasta -fi "$COMMON_DIR/hg38.fa" -bed "$OUTPUT_DIR/peaks.bed" -fo "$OUTPUT_DIR/peaks.fa"


# printf '*%.0s' {1..50}; echo
# echo "FIMO scan"

# fimo --thresh 1e-4 --oc fimo_out "$COMMON_DIR/JASPAR2026_CORE_vertebrates_non-redundant_pfms.meme" "$OUTPUT_DIR/peaks.fa"

# #########################################################################
# #==================== HIC & JASPAR

# cd $CODE_PATH

# printf '*%.0s' {1..50}; echo
# echo "process hic and jaspar data"
# echo "******************************"

# python -u hic_jaspar_main.py

# #########################################################################
# #==================== Data Process

printf '*%.0s' {1..50}; echo
echo "Data process"

cd $CODE_PATH

python -u main_data_process.py

# #########################################################################


# cd $CODE_PATH

# printf '*%.0s' {1..50}; echo
# echo "Trajectory inference"

# conda activate /home/liyang/BioWuYan/conda_env/R43_env

# Rscript trajectory_inference.R

