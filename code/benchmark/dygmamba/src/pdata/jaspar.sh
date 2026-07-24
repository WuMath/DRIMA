#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=fimo_jaspar          ##作业名称
#SBATCH --output=/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/log/print_%j.out
#SBATCH --nodes=1                     ##指定使用1个节点数
#SBATCH -n 1                          ##指定总任务数1
#SBATCH --qos=a100g1                  ##指定qos为a100g1
#SBATCH --gres=gpu:a100:1             ##指定使用gpu资源,a100数量1
#SBATCH -p a100                       

set -e

source /home/liyang/miniconda3/etc/profile.d/conda.sh

conda activate /home/liyang/BioWuYan/conda_env/singlecellpreprocess

cd "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/process"

COMMON_DIR="/home/liyang/BioWuYan/dygmamba_project/data/jaspar"

OUTPUT_DIR="/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/process"

# bedtools getfasta -fi "$COMMON_DIR/hg38.fa" -bed "$OUTPUT_DIR/peaks.bed" -fo "$OUTPUT_DIR/peaks.fa"

echo "FIMO scan"

fimo --thresh 1e-4 --oc fimo_out "$COMMON_DIR/JASPAR2026_CORE_vertebrates_non-redundant_pfms.meme" "$OUTPUT_DIR/peaks.fa"