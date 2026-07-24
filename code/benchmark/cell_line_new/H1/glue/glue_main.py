import os
import code
import sys
import pickle
import dill
import anndata as ad
import networkx as nx
import scanpy as sc
import scglue


import numpy as np
import pandas as pd

from GLUE import data_preprocess_glue
from GLUE import GLUE_model_train
from GLUE import GRN_inference, GRN_Analysis



if __name__ =="__main__":
    
    cell_type = "H1"

    output_path = '/home/wuyan/dygmamba_project/data/cell_line/' + cell_type + '/data_glue/'

    os.makedirs(output_path, exist_ok=True)

    data_path = "/home/wuyan/dygmamba_project/data/cell_line/" + cell_type + "/process/"

    # glue_data_path = "/home/wuyan/MethodTest/new_scglue/data/"

    gtf_file = "/home/wuyan/dygmamba_project/data/annotation/gencode.v48.chr_patch_hapl_scaff.annotation.gtf.gz"

    motif_bed_file = "/home/wuyan/dygmamba_project/data/jaspar/glue/JASPAR2022-hg38.bed.gz"
    
    ##########################################################################################################
    # Data read
    adata_rna = ad.read_h5ad(data_path + "rna_processed.h5ad") 

    adata_atac = ad.read_h5ad(data_path + "atac_processed.h5ad")

    adata_rna.obs["cell_type"] ='cluster1'
    adata_atac.obs["cell_type"] = 'cluster1'

    ##########################################################################################################
    # For **glue**, 
    # - the **rna** should have the following preprocess: filter of genes and cells, 
    #           save raw counts in layers["counts"], highly variable gene, normalize, log1p, scale, and pca.
    # - the **atac** should have the following preprocess: lsi, neighbor, umap
    # 
    ##########################################################################################################

    scglue.data.lsi(adata_atac, n_components=100, n_iter=15)
    sc.pp.neighbors(adata_atac, use_rep="X_lsi", metric="cosine")
    sc.tl.umap(adata_atac)
    
    data_preprocess_glue(adata_rna, adata_atac, output_path, gtf_file)
    
    ##########################################################################################################
    # ****************************************** Model Train ***********************************************

    rna = ad.read_h5ad(output_path + "rna-pp.h5ad")
    atac = ad.read_h5ad(output_path + "atac-pp.h5ad")
    guidance = nx.read_graphml(output_path + "guidance.graphml.gz")

    GLUE_model_train(rna, atac,guidance, output_path)

    #########################################################################################################
    # ****************************************** Model Inference ***********************************************

    rna = ad.read_h5ad(output_path + "rna-emb.h5ad")
    atac = ad.read_h5ad(output_path + "atac-emb.h5ad")
    guidance_hvf = nx.read_graphml(output_path + "guidance-hvf.graphml.gz")

    motif_bed = scglue.genomics.read_bed( motif_bed_file)

    genes = rna.var.query("highly_variable").index
    peaks = atac.var.query("highly_variable").index
    tfs = pd.Index(motif_bed["name"]).intersection(rna.var_names)

    print(f"****************** tf size : {tfs.size} ******************")

    rna[:, np.union1d(genes, tfs)].write_loom(output_path + "rna.loom")
    np.savetxt( output_path + "tfs.txt", tfs, fmt="%s")


    gene2peak = GRN_inference(rna, atac, guidance_hvf, motif_bed, output_path)

    gene2tf_rank_glue, gene2tf_rank_supp, peak2tf = GRN_Analysis(rna, atac, gene2peak, motif_bed, tfs)

    with open(output_path + "gene2peak.pkl", "wb") as f:
        dill.dump(gene2peak, f)
    
    with open(output_path + "peak2tf.pkl", "wb") as f:
        dill.dump(peak2tf, f)
        
    ##########################################################################################################
    # ****************************************** Analysis ***********************************************
    
    # get tfs
    genes = rna.var.query("highly_variable").index
    peaks = atac.var.query("highly_variable").index
    tfs = pd.Index(motif_bed["name"]).intersection(rna.var_names)

    # # GRN fine 
    gene2tf_rank_glue.columns = gene2tf_rank_glue.columns + "_glue"
    gene2tf_rank_supp.columns = gene2tf_rank_supp.columns + "_supp"

    scglue.genomics.write_scenic_feather(gene2tf_rank_glue, output_path + "glue.genes_vs_tracks.rankings.feather")
    scglue.genomics.write_scenic_feather(gene2tf_rank_supp, output_path + "supp.genes_vs_tracks.rankings.feather")

    TF_data = pd.concat([
        pd.DataFrame({
            "#motif_id": tfs + "_glue",
            "gene_name": tfs
        }),
        pd.DataFrame({
            "#motif_id": tfs + "_supp",
            "gene_name": tfs
        })
    ]).assign(
        motif_similarity_qvalue=0.0,
        orthologous_identity=1.0,
        description="placeholder"
    )

    TF_data.to_csv(output_path + "ctx_annotation.tsv", sep="\t", index=False)

    print("***************** END **********************")
    
    os._exit(0)