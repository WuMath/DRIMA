#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH  --job-name=atac_sim    ##任务名
#SBATCH --output=/home/wuyan/dygmamba_project/data/scdata/H1/log/print_%j.out
#SBATCH  --nodes=1                     ##指定使用1个节点数
#SBATCH  -n 1                          ##指定总任务数1
#SBATCH  --qos=a100g2                  ##指定qos为a100g1
#SBATCH  -p a100                       

source /home/wuyan/miniconda3/etc/profile.d/conda.sh

conda activate /home/wuyan/conda_env/singlecellpreprocess

CELLTYPE="H1"
# --- configuration ---

cd "/home/wuyan/dygmamba_project/data/scdata/$CELLTYPE/atac"

INPUT_BAM="H1_ENCFF801RHD.bam"

SAMPLE_SIZE=20000

OUTPUT_DIR="atac_bam"

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

# FRACTION=$(awk -v size="$SAMPLE_SIZE" -v total="$TOTAL_LINES" 'BEGIN {print size/total}')

FRACTION=$(awk -v size="$SAMPLE_SIZE" -v total="$TOTAL_LINES" \
    'BEGIN { printf "%.6f\n", size/total }' | sed 's/^0\.//')

FRACTION=${FRACTION#0.}

echo "Total reads: $TOTAL_LINES, Fraction: $FRACTION"

NUM_JOBS=500

for i in $(seq 1 $NUM_JOBS);
do
    SEED=$(( i * 100 + RANDOM ))
    TASK_ID=$i
    OUTPUT_BAM="${OUTPUT_DIR}/atac_cell_${TASK_ID}.bam"
    SORT_BAM="${OUTPUT_DIR}/atac_cell_${TASK_ID}.sorted.bam"

    echo "Output file: $OUTPUT_BAM"
    echo "Random seed: $SEED"
    echo "Running samtools view..."

    samtools view -h -s ${SEED}.${FRACTION} -b -f 2 -o "$OUTPUT_BAM" "$INPUT_BAM"

    if [ $? -eq 0 ]; then

        echo "Successfully created sample file: $OUTPUT_BAM"

        set -eo pipefail

        echo "sort bam file"

        samtools sort "$OUTPUT_BAM" -o "$SORT_BAM"

        samtools index "$SORT_BAM"

    else
        echo "Error creating sample file for task $TASK_ID"
        exit 1 
    fi
done

SC_BAM_FILES="$OUTPUT_DIR/*.sorted.bam"

mkdir -p ./peaks

macs2 callpeak -t "$INPUT_BAM" \
    -f BAMPE \
    -n Consensus_All \
    -g hs \
    --outdir ./peaks \
    --nomodel --shift -100 --extsize 200 \
    --keep-dup all

echo "Step 2: Creating SAF format..."

awk 'BEGIN{OFS="\t"; print "GeneID\tChr\tStart\tEnd\tStrand"} {print "Peak_"NR, $1, $2, $3, "+"}' ./peaks/Consensus_All_peaks.narrowPeak > consensus.saf

echo "Step 3: FeatureCounts on single cells..."
# 使用高质量的 Peak 列表对模拟的单细胞进行定量
featureCounts -a consensus.saf \
              -F SAF \
              -o counts_matrix.txt \
              -T 20 \
              -p \
              $SC_BAM_FILES

echo "Done. Skipped merging step by using original bulk BAM."
