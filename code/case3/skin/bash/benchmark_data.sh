#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=benchdata
#SBATCH --output=/home/wuyan/dygmamba_project/NewRealPlan/log/print_%j.out
#SBATCH --nodes=1                     ##指定使用1个节点数
#SBATCH -n 1                          ##指定总任务数1
#SBATCH --qos=a100g2                  ##指定qos为a100g1

#SBATCH -p a100                       

set -e

CELL_TYPE="skin"



COMMON_DIR="/home/wuyan/dygmamba_project"

CODE_PATH="/home/wuyan/dygmamba_project/NewRealPlan/case3/code/$CELL_TYPE/src"

OUTPUT_DIR="/home/wuyan/dygmamba_project/NewRealPlan/case3/data/process/$CELL_TYPE/process"


JASPAR_FILE_FA="/home/wuyan/dygmamba_project/NewRealPlan/case3/data/mm10/fasta/genome.fa"

JASPAR_FILE_ME="/home/wuyan/dygmamba_project/data/jaspar/JASPAR2026_CORE_vertebrates_non-redundant_pfms.meme"


mkdir -p $OUTPUT_DIR


echo "Process benchmark cell $CELL_TYPE"

source /home/wuyan/miniconda3/etc/profile.d/conda.sh

conda activate /home/wuyan/conda_env/singlecellpreprocess

######################################################################
cd $CODE_PATH

python -u peak.py

#########################################################################
#===================== JASPAR

printf '*%.0s' {1..50}; echo
echo "JASPAR process"

cd $OUTPUT_DIR

bedtools getfasta -fi $JASPAR_FILE_FA -bed "$OUTPUT_DIR/peaks.bed" -fo "$OUTPUT_DIR/peaks.fa"


printf '*%.0s' {1..50}; echo
echo "FIMO scan"

fimo --thresh 1e-4 --oc fimo_out $JASPAR_FILE_ME "$OUTPUT_DIR/peaks.fa"

#########################################################################
#==================== Benchmark process

cd $CODE_PATH

python -u benchmark_data_read.py

echo "Finidhed process"

