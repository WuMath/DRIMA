import code
import pickle
import anndata as ad
import networkx as nx
import scanpy as sc
import scglue

import numpy as np
import pandas as pd
import seaborn as sns

from IPython import display
from matplotlib import rcParams
from networkx.algorithms.bipartite import biadjacency_matrix
from networkx.drawing.nx_agraph import graphviz_layout
from itertools import chain



def data_preprocess_glue(adata_rna, adata_atac, data_path, gtf_file):
    ######################
    # rna analysis

    scglue.data.get_gene_annotation(
        adata_rna, gtf=gtf_file,
        gtf_by="gene_name"
    )
    adata_rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()

    invalid_genes = adata_rna.var[adata_rna.var['chromStart'].isna() | adata_rna.var['chromEnd'].isna()]
    valid_genes_mask = adata_rna.var['chromStart'].notna() & adata_rna.var['chromEnd'].notna()
    adata_rna_clean = adata_rna[:, valid_genes_mask].copy()

    ######################
    # atac analysis

    
    split = adata_atac.var_names.str.split(r"[:-]")
    adata_atac.var["chrom"] = split.map(lambda x: x[0])
    adata_atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
    adata_atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
    adata_atac.var.head()

    ######################
    # construct and check graph

    guidance = scglue.genomics.rna_anchored_guidance_graph(adata_rna_clean, adata_atac)

    # 获取两个数据集中所有特征的名称，并放入集合中以便快速查找
    rna_feature_names = set(adata_rna.var_names)
    atac_feature_names = set(adata_atac.var_names)

    # 遍历指导图中的所有节点，将它们分别归入两个列表
    graph_rna_nodes = [
        node for node in guidance.nodes
        if node in rna_feature_names
    ]
    graph_atac_nodes = [
        node for node in guidance.nodes
        if node in atac_feature_names
    ]

    # 使用明确分离好的特征列表来过滤 AnnData 对象
    adata_rna = adata_rna[:, graph_rna_nodes].copy()
    adata_atac = adata_atac[:, graph_atac_nodes].copy()

    # 现在再进行检查，应该就不会报错了
    scglue.graph.check_graph(guidance, [adata_rna, adata_atac])

    #############################
    # 1. 找出所有 'object' 类型的列，这些是潜在的混合类型列
    object_cols = adata_rna.var.select_dtypes(include=['object']).columns

    # 2. 遍历这些列，并将它们的类型强制转换为字符串
    for col in object_cols:
        # 这会将列中的所有值（包括 NaN）都变成字符串形式（例如 'nan'）
        adata_rna.var[col] = adata_rna.var[col].astype(str)

    adata_rna.write(data_path + "rna-pp.h5ad", compression="gzip")

    adata_atac.write(data_path + "atac-pp.h5ad", compression="gzip")

    nx.write_graphml(guidance, data_path + "guidance.graphml.gz")



def GLUE_model_train(rna, atac, guidance, data_path_stage):

    rna.obs["domain"] = "scRNAseq"
    atac.obs["domain"] = "scATACseq"

    scglue.models.configure_dataset(
        rna, "NB", use_highly_variable=True,
        use_layer="counts", use_rep="X_pca"
    )

    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=True,
        use_rep="X_lsi"
    )

    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )).copy()

    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac}, guidance_hvf,
        fit_kws={"directory": data_path_stage + "glue_fit", "max_epochs": 400}
    )

    glue.save(data_path_stage + "glue.dill")
    # glue = scglue.models.load_model("glue.dill")

    dx = scglue.models.integration_consistency(
        glue, {"rna": rna, "atac": atac}, guidance_hvf
    )
    _ = sns.lineplot(x="n_meta", y="consistency", data=dx).axhline(y=0.05, c="darkred", ls="--")

    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)

    combined = ad.concat([rna, atac])

    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
    sc.tl.umap(combined)

    sc.settings.figdir = data_path_stage
    sc.pl.umap(combined, color=["cell_type", "domain"], wspace=0.65, save="combined_umap.pdf")

    feature_embeddings = glue.encode_graph(guidance_hvf)
    feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)
    feature_embeddings.iloc[:5, :5]

    rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
    atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()

    rna.write(data_path_stage + "rna-emb.h5ad", compression="gzip")
    atac.write(data_path_stage + "atac-emb.h5ad", compression="gzip")
    nx.write_graphml(guidance_hvf, data_path_stage + "guidance-hvf.graphml.gz")






def GRN_inference(rna, atac, guidance_hvf, motif_bed, output_path):

    rna.var["name"] = rna.var_names
    atac.var["name"] = atac.var_names

    features = pd.Index(np.concatenate([rna.var_names, atac.var_names]))
    feature_embeddings = np.concatenate([rna.varm["X_glue"], atac.varm["X_glue"]])

    skeleton = guidance_hvf.edge_subgraph(
        e for e, attr in dict(guidance_hvf.edges).items()
        if attr["type"] == "fwd"
    ).copy()

    reginf = scglue.genomics.regulatory_inference(
        features, feature_embeddings,
        skeleton=skeleton, random_state=0
    )

    gene2peak = reginf.edge_subgraph(
        e for e, attr in dict(reginf.edges).items()
        if attr["qval"] < 0.05
    )

    #########################################################################

    scglue.genomics.Bed(atac.var).write_bed(output_path + "peaks.bed", ncols=3)

    scglue.genomics.write_links(
        gene2peak,
        scglue.genomics.Bed(rna.var).strand_specific_start_site(),
        scglue.genomics.Bed(atac.var),
        output_path + "gene2peak.links", keep_attrs=["score"]
    )

    
    # #########################################################################
    # # ***************************** peak-tf glue  **************************


    # peak_bed = scglue.genomics.Bed(atac.var.loc[peaks])
    
    # peak2tf = scglue.genomics.window_graph(peak_bed, motif_bed, 0, right_sorted=True)

    # peak2tf = peak2tf.edge_subgraph(e for e in peak2tf.edges if e[1] in tfs)

    # ###################################################
    # # gene-peak-tf glue
    # gene2tf_rank_glue = scglue.genomics.cis_regulatory_ranking(
    #     gene2peak, peak2tf, genes, peaks, tfs,
    #     region_lens=atac.var.loc[peaks, "chromEnd"] - atac.var.loc[peaks, "chromStart"],
    #     random_state=0
    # )
 
    # #########################################################################
    # # ***************************** peak-tf supplementary  ****************** 

    # flank_bed = scglue.genomics.Bed(rna.var.loc[genes]).strand_specific_start_site().expand(500, 500)
    
    # flank2tf = scglue.genomics.window_graph(flank_bed, motif_bed, 0, right_sorted=True)

    # gene2flank = nx.Graph([(g, g) for g in genes])
    
    # ###################################################
    # # gene-peak-tf supplementary
    # gene2tf_rank_supp = scglue.genomics.cis_regulatory_ranking(
    #     gene2flank, flank2tf, genes, genes, tfs,
    #     n_samples=0
    # )
    # gene2tf_rank_supp.iloc[:5, :5]

    return gene2peak 




def GRN_Analysis(rna, atac, gene2peak, motif_bed, tfs):
    #########################################################################
    # ***************************** peak-tf glue  **************************
    genes = rna.var.query("highly_variable").index
    peaks = atac.var.query("highly_variable").index

    peak_bed = scglue.genomics.Bed(atac.var.loc[peaks])
    
    peak2tf = scglue.genomics.window_graph(peak_bed, motif_bed, 0, right_sorted=True)

    peak2tf = peak2tf.edge_subgraph(e for e in peak2tf.edges if e[1] in tfs)

    ###################################################
    # gene-peak-tf glue
    gene2tf_rank_glue = scglue.genomics.cis_regulatory_ranking(
        gene2peak, peak2tf, genes, peaks, tfs,
        region_lens=atac.var.loc[peaks, "chromEnd"] - atac.var.loc[peaks, "chromStart"],
        random_state=0
    )
 
    #########################################################################
    # ***************************** peak-tf supplementary  ****************** 

    flank_bed = scglue.genomics.Bed(rna.var.loc[genes]).strand_specific_start_site().expand(500, 500)
    
    flank2tf = scglue.genomics.window_graph(flank_bed, motif_bed, 0, right_sorted=True)

    gene2flank = nx.Graph([(g, g) for g in genes])
    
    ###################################################
    # gene-peak-tf supplementary
    gene2tf_rank_supp = scglue.genomics.cis_regulatory_ranking(
        gene2flank, flank2tf, genes, genes, tfs,
        n_samples=0
    )

    gene2tf_rank_supp.iloc[:5, :5]

    return gene2tf_rank_glue, gene2tf_rank_supp, peak2tf


##########################################################################################################
# ****************************************** Main Function ***********************************************

def main():

    output_path = "/home/liyang/BioWuYan/Compara/Data/selfexperiment/scglue3/"

    #---------- option 1----------------------
    # data_path = "./Data/"
    # adata_rna, adata_atac = data_reader(data_path)

    #---------- option 2-----------------------

    data_path = '/home/liyang/BioWuYan/BioProject/Data/scenicplus_data/GM12878/'

    gtf_file = data_path + 'gencode.v48.chr_patch_hapl_scaff.annotation.gtf.gz'

    adata_rna = ad.read_h5ad(data_path + "preprocess_adata_rna.h5ad") 

    adata_atac = ad.read_h5ad(data_path + "atac_bams/adata_atac.h5ad")

    
    # there need a cell type for umap
    adata_rna.obs["cell_type"] ='cluster1'
    adata_atac.obs["cell_type"] = 'cluster1'

    #########################################################################
    # ***************************** data preprocess  ******************

    data_preprocess_glue(adata_rna, adata_atac, output_path, gtf_file)
    

    #########################################################################
    # ***************************** model train  ******************
    rna = ad.read_h5ad(data_path + "rna-pp.h5ad")
    atac = ad.read_h5ad(data_path + "atac-pp.h5ad")
    guidance = nx.read_graphml(data_path + "guidance.graphml.gz")

    GLUE_model_train(rna, atac,guidance, output_path)

    #########################################################################
    # ***************************** peak-gene inference  ******************
    rna = ad.read_h5ad(output_path + "rna-emb.h5ad")
    atac = ad.read_h5ad(output_path + "atac-emb.h5ad")
    guidance_hvf = nx.read_graphml(output_path + "guidance-hvf.graphml.gz")

    motif_bed = scglue.genomics.read_bed(data_path + "JASPAR2022-mm10.bed.gz")

    genes = rna.var.query("highly_variable").index
    peaks = atac.var.query("highly_variable").index
    tfs = pd.Index(motif_bed["name"]).intersection(rna.var_names)

    print(f"****************** tf size : {tfs.size} ******************")

    rna[:, np.union1d(genes, tfs)].write_loom(output_path + "rna.loom")
    np.savetxt( output_path + "tfs.txt", tfs, fmt="%s")

    gene2peak = GRN_inference(rna, atac, guidance_hvf, motif_bed, output_path)

    gene2tf_rank_glue, gene2tf_rank_supp = GRN_Analysis(rna, atac, gene2peak, motif_bed, tfs)

    ##########################################################################################################
    # ****************************************** Analysis ***********************************************

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


    grn = scglue.genomics.read_ctx_grn( output_path + "pruned_grn.csv")

    # breakpoint()




if __name__ == "__main__":
    main()
    # code.interact(local=locals())



