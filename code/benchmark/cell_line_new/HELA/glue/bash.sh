echo "***********************************"
echo "Start fine grn"

CELL_TYPE="HELA"

source /home/wuyan/miniconda3/etc/profile.d/conda.sh

conda activate /home/wuyan/conda_env/pyscenic_env

cd /home/wuyan/dygmamba_project/data/cell_line/$CELL_TYPE/data_glue

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