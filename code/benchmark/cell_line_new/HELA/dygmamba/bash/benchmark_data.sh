#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=benchdata
#SBATCH --output=/home/wuyan/dygmamba_project/model/0log/print_%j.out
#SBATCH --nodes=1                     ##指定使用1个节点数
#SBATCH -n 1                          ##指定总任务数1
#SBATCH --qos=a100g2                  ##指定qos为a100g1

#SBATCH -p a100                       

set -e

CELL_TYPE="HELA"


COMMON_DIR="/home/wuyan/dygmamba_project"

CODE_PATH="$COMMON_DIR/model/cell_line/$CELL_TYPE/dygmamba/src"

JASPAR_DIR="$COMMON_DIR/data/jaspar"

OUTPUT_DIR="$COMMON_DIR/data/cell_line/$CELL_TYPE/process"

mkdir -p $OUTPUT_DIR


echo "Process benchmark cell $CELL_TYPE"

source /home/wuyan/miniconda3/etc/profile.d/conda.sh

conda activate /home/wuyan/conda_env/singlecellpreprocess

#########################################################################
#===================== JASPAR

# printf '*%.0s' {1..50}; echo
# echo "JASPAR process"

# cd $OUTPUT_DIR

# bedtools getfasta -fi "$JASPAR_DIR/hg38.fa" -bed "$OUTPUT_DIR/peaks.bed" -fo "$OUTPUT_DIR/peaks.fa"


# printf '*%.0s' {1..50}; echo
# echo "FIMO scan"

# fimo --thresh 1e-4 --oc fimo_out "$JASPAR_DIR/JASPAR2026_CORE_vertebrates_non-redundant_pfms.meme" "$OUTPUT_DIR/peaks.fa"

#########################################################################
#==================== Benchmark process

cd $CODE_PATH

python -u benchmark_data_read.py

echo "Finidhed process"

