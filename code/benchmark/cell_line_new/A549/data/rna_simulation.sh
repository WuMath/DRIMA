#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH  --job-name=rna_sim    ##任务名
#SBATCH --output=/home/wuyan/dygmamba_project/data/scdata/A549/log/print_%j.out
#SBATCH  --nodes=1                     ##指定使用1个节点数
#SBATCH  -n 1                          ##指定总任务数1
#SBATCH  --qos=a100g1                  ##指定qos为a100g1
#SBATCH  -p a100                       

source /home/wuyan/miniconda3/etc/profile.d/conda.sh

conda activate /home/wuyan/conda_env/singlecellpreprocess

# --- configuration ---

cd "/home/wuyan/dygmamba_project/data/scdata/A549/rna"

INPUT_BAM="A549_ENCFF394BGT.bam"

SAMPLE_SIZE=50000

OUTPUT_DIR="rna_bams"

mkdir -p "$OUTPUT_DIR"

set -eo pipefail

# --- configuration---

echo "========================================================"
echo "Starting Slurm Array Task"
echo "Job ID: $SLURM_JOB_ID"
echo "Input BAM: $INPUT_BAM"
echo "========================================================"


echo "Calculating sampling fraction..."

TOTAL_LINES=$(samtools view -c "$INPUT_BAM")

FRACTION=$(awk -v size="$SAMPLE_SIZE" -v total="$TOTAL_LINES" 'BEGIN {print size/total}')

FRACTION=${FRACTION#0.}

echo "Total reads: $TOTAL_LINES, Fraction: $FRACTION"

NUM_JOBS=500

for i in $(seq 1 $NUM_JOBS)
do
    SEED=$(( i * 100 + RANDOM ))
    TASK_ID=$i
    OUTPUT_BAM="${OUTPUT_DIR}/rna_cell_${TASK_ID}.bam"
    SORT_BAM="${OUTPUT_DIR}/rna_cell_${TASK_ID}.sorted.bam"

    echo "Output file: $OUTPUT_BAM"

    echo "Random seed: $SEED"

    echo "Running samtools view..."

    samtools view -h -s ${SEED}.${FRACTION} -b -o "$OUTPUT_BAM" "$INPUT_BAM"

    if [ $? -eq 0 ]; then
    
        echo "Successfully created sample file: $OUTPUT_BAM"

        mkdir -p "cells_gene_count"

        set -eo pipefail

        echo "process bam file"

        samtools sort "$OUTPUT_BAM" -o "$SORT_BAM"

        samtools index "$SORT_BAM"

        echo "start featureCounts"

        featureCounts -a "/home/wuyan/dygmamba_project/data/scdata/gencode.V49.annotation.gtf" \
                    -o "cells_gene_count/rna_cell_${TASK_ID}_counts.txt" \
                    -T 4 \
                    -t exon \
                    -g gene_name \
                    -p \
                    --countReadPairs \
                    $SORT_BAM

        echo "rm garbage"

        rm "$OUTPUT_DIR/rna_cell_${TASK_ID}.sorted.bam.bai" "$SORT_BAM"
    else
        echo "Error creating sample file for task $TASK_ID"
        exit 1 
    fi

done

echo "Array Task $TASK_ID finished at $(date)"