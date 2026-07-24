#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=scglue
#SBATCH --output=/home/liyang/BioWuYan/dygmamba_project/model/cell_line/0log/print_%j.out
#SBATCH  --nodes=1                     ##指定使用1个节点数
#SBATCH  -n 1                          ##指定总任务数1
#SBATCH  --qos=a100g1                  ##指定qos为a100g1
#SBATCH  -p a100                       


set -e

source /home/liyang/miniconda3/etc/profile.d/conda.sh

CELL_TYPE="IMR90"

printf '*%.0s' {1..50}; echo

echo "run GLUE $CELL_TYPE job"

printf '*%.0s' {1..50}; echo

FLAG_TARGET="Step3"


if [ "$FLAG_TARGET" == "All" ]; then

    conda activate /home/liyang/BioWuYan/conda_env/new_scglue

    cd /home/liyang/BioWuYan/dygmamba_project/model/cell_line/$CELL_TYPE/glue

    echo "***********************************"
    echo "Start glue"

    python -u glue_main.py

    echo "***************************** End ***************************"

    echo "***********************************"
    echo "Start draft grn"

    conda activate /home/liyang/BioWuYan/conda_env/pyscenic_env

    cd /home/liyang/BioWuYan/dygmamba_project/data/cell_line/$CELL_TYPE/data_glue

    pyscenic grn rna.loom tfs.txt \
        -o draft_grn.csv --seed 0 --num_workers 1 \
        --cell_id_attribute obs_names --gene_attribute name
    echo "***************************** End ***************************"

    echo "***********************************"
    echo "Start fine grn"
    
    conda activate /home/liyang/BioWuYan/conda_env/pyscenic_env

    cd /home/liyang/BioWuYan/dygmamba_project/data/cell_line/$CELL_TYPE/data_glue

    pyscenic ctx draft_grn.csv \
        glue.genes_vs_tracks.rankings.feather \
        supp.genes_vs_tracks.rankings.feather \
        --annotations_fname ctx_annotation.tsv \
        --expression_mtx_fname rna.loom \
        --output pruned_grn.csv \
        --rank_threshold 500 --min_genes 1 \
        --num_workers 4 \
        --cell_id_attribute obs_names --gene_attribute name 2> /dev/null

    echo "***************************** End ***************************"

fi







if [ "$FLAG_TARGET" == "Step1" ]; then

    conda activate /home/liyang/BioWuYan/conda_env/new_scglue

    cd /home/liyang/BioWuYan/dygmamba_project/model/cell_line/$CELL_TYPE/glue

    echo "***********************************"
    echo "Start glue"

    python -u glue_main.py

    echo "***************************** End ***************************"

fi


if [ "$FLAG_TARGET" == "Step2" ]; then

    echo "***********************************"
    echo "Start draft grn"

    conda activate /home/liyang/BioWuYan/conda_env/pyscenic_env

    cd /home/liyang/BioWuYan/dygmamba_project/data/cell_line/$CELL_TYPE/data_glue

    pyscenic grn rna.loom tfs.txt \
        -o draft_grn.csv --seed 0 --num_workers 1 \
        --cell_id_attribute obs_names --gene_attribute name
    echo "***************************** End ***************************"

fi

if [ "$FLAG_TARGET" == "Step3" ]; then

    echo "***********************************"
    echo "Start fine grn"
    
    conda activate /home/liyang/BioWuYan/conda_env/pyscenic_env

    cd /home/liyang/BioWuYan/dygmamba_project/data/cell_line/$CELL_TYPE/data_glue

    pyscenic ctx draft_grn.csv \
        glue.genes_vs_tracks.rankings.feather \
        supp.genes_vs_tracks.rankings.feather \
        --annotations_fname ctx_annotation.tsv \
        --expression_mtx_fname rna.loom \
        --output pruned_grn.csv \
        --rank_threshold 500 --min_genes 1 \
        --num_workers 4 \
        --cell_id_attribute obs_names --gene_attribute name 2> /dev/null

    echo "***************************** End ***************************"

fi


