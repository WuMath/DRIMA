import scanpy as sc
import celloracle as co
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import numpy as np
import code
import anndata as ad
from scipy.io import mmread

from scipy import sparse
import scipy.sparse as sp_sparse
from scipy.sparse import issparse


def filter_jaspar_tf(jaspar_tf_peak):
    '''
    input data: 
        jaspar_tf_peak: is a anndata with tf as vars, peak as obs
    '''
    if sp_sparse.issparse(jaspar_tf_peak.X):
        nan_mask = np.isnan(jaspar_tf_peak.X.data)
        if np.any(nan_mask):
            print(f"  > 发现了 {np.sum(nan_mask)} 个 NaN 值, 将其设为 0 ...")
            jaspar_tf_peak.X.data[nan_mask] = 0.0
    else:
        if np.any(np.isnan(jaspar_tf_peak.X)):
            print(f"  > 发现了 NaN 值, 将其设为 0 ...")
            jaspar_tf_peak.X = np.nan_to_num(jaspar_tf_peak.X, nan=0.0)


    score_threshold = 0
    jaspar_tf_peak.X = (jaspar_tf_peak.X > score_threshold).astype(int)
    jaspar_tf_peak.X = sp_sparse.csr_matrix(jaspar_tf_peak.X)

    # ************* 1. Peak Filter (Rows) **************************

    peak_counts = jaspar_tf_peak.X.getnnz(axis=1)
    keep_peaks_mask = peak_counts > 0

    print(f"\n步骤 1: 过滤 Peaks (行)")
    print(f"  > 找到 {np.sum(keep_peaks_mask)} / {jaspar_tf_peak.n_obs} 个 peaks 至少有 1 个 TF 结合。")
    jaspar_tf_peak = jaspar_tf_peak[keep_peaks_mask, :].copy()

    # ************* 2. TF Filter (Columns) **************************

    tf_counts = jaspar_tf_peak.X.getnnz(axis=0)
    keep_tfs_mask = tf_counts > 0

    print(f"\n步骤 2: 过滤 TFs (列)")
    print(f"  > 找到 {np.sum(keep_tfs_mask)} / {jaspar_tf_peak.n_vars} 个 TFs 至少结合 1 个 peak。")

    jaspar_tf_peak = jaspar_tf_peak[:, keep_tfs_mask].copy()

    print(f"  > 最终形状: {jaspar_tf_peak.shape}")

    return jaspar_tf_peak



def Anndata_read(data_path, feature_file1, feature_file2, feature1, feature2):

    X = mmread(data_path + "counts.mtx").tocsr()
    obs = pd.read_feather(data_path + "metadata.feather")
    obs.index = pd.read_csv(data_path + feature_file1, header=0)[feature1]
    var = pd.DataFrame(index=pd.read_csv(data_path + feature_file2, header=0)[feature2])
    
    adata = ad.AnnData(X=X.T, obs=obs, var=var)

    return adata

def get_oracle_data(data_path):
    data_path = "/home/liyang/BioWuYan/BioProject/Prepocess/Simulation/scmultisim/main_ds/tree1_500_cells110_genes_sigma0.1_1/"
    print(data_path)

    adata_rna = Anndata_read(data_path + "RNAData/",
                                    feature_file1="cells.csv",
                                    feature_file2="genes.csv",
                                    feature1="cells",
                                    feature2="genes")

    adata_atac = Anndata_read(data_path + "ATACData/",
                                    feature_file1="cells.csv",
                                    feature_file2="peaks.csv",
                                    feature1="cells",
                                    feature2="peaks")

    adata_rna_obs = Anndata_read(data_path + "RNADataObs/",
                                    feature_file1="cells.csv",
                                    feature_file2="genes.csv",
                                    feature1="cells",
                                    feature2="genes")

    X = mmread(data_path + "RP/regulation.mtx").tocsr()
    obs = pd.DataFrame(index=pd.read_csv(data_path + "RP/genes.tsv", header=None)[0])
    var = pd.DataFrame(index=pd.read_csv(data_path + "RP/peaks.tsv", header=None)[0])

    RP_adata = ad.AnnData(X=X.T, obs=obs, var=var)

    RT_X = mmread(data_path + "RT/regulation.mtx").tocsr()
    RT_obs = pd.DataFrame(index=pd.read_csv(data_path + "RT/tfs.tsv", header=None)[0])
    RT_var = pd.DataFrame(index=pd.read_csv(data_path + "RT/peaks.tsv", header=None)[0])

    RT_adata = ad.AnnData(X=RT_X.T, obs=RT_obs, var=RT_var)

    grn = pd.read_csv(data_path + "grn.csv")
    
    grn = grn.set_index(grn.columns[0])


    return adata_atac, adata_rna, adata_rna_obs, RT_adata, RP_adata, grn

# code.interact(local=locals())


if __name__ == "__main__":
    
    cell_type = "MCF7"
    
    data_path = '/home/liyang/BioWuYan/dygmamba_project/data/cell_line/' + cell_type + '/process/'

    output_path = '/home/liyang/BioWuYan/dygmamba_project/data/cell_line/' + cell_type + '/data_celloracle/'

    os.makedirs(output_path, exist_ok=True)
    
    ############################################################################################
    # ***************************************** data read *************************************

    adata_atac = ad.read_h5ad(data_path + "atac_processed.h5ad")

    adata_rna = ad.read_h5ad(data_path + "rna_processed.h5ad")

    adata_rp_gene_peak = ad.read_h5ad(data_path + "peak_gene_rp_network.h5ad")

    jaspar_tf_peak = ad.read_h5ad(data_path + "jaspar_data_processed.h5ad")

    ############################################################################################
    # ***************************************** data processed ********************************

    jaspar_tf_peak = filter_jaspar_tf(jaspar_tf_peak)

    common_peaks = set(adata_atac.var_names) & set(jaspar_tf_peak.obs_names) & set(adata_rp_gene_peak.var_names)
    common_peaks_list = list(common_peaks)

    adata_atac = adata_atac[:, common_peaks_list].copy()

    jaspar_filtered = jaspar_tf_peak[common_peaks_list, :].copy()

    jaspar_filtered.var_names_make_unique()

    adata_rp_gene_peak = adata_rp_gene_peak[:, common_peaks_list].copy()


    common_genes = set(adata_rna.var_names) & set(adata_rp_gene_peak.obs_names)

    common_genes_list = list(common_genes)

    adata_rna = adata_rna[:, common_genes_list].copy()

    adata_rp_gene_peak = adata_rp_gene_peak[common_genes_list, :].copy()

    # print("Preprocess the count matrix...")
    # sc.pp.normalize_per_cell(adata_rna, counts_per_cell_after=100)
    # adata_rna.raw = adata_rna
    
    # adata_rna.layers["raw_count"] = adata_rna.raw.X.copy()
   
    # sc.pp.log1p(adata_rna)
    # sc.pp.scale(adata_rna)
    # sc.tl.pca(adata_rna, svd_solver='arpack')
    # sc.pp.neighbors(adata_rna, n_neighbors=30, n_pcs=30)
    # sc.tl.umap(adata_rna)
   
    # adata_rna.obs["leiden"] = "cluster0"
   
    # sc.pl.umap(adata_rna, color="leiden")

    ############################################################################################
    # ***************************************** construct grn  ********************************
    
    if issparse(jaspar_filtered.X):
        data_mat = jaspar_filtered.X.toarray()
    else:
        data_mat = jaspar_filtered.X

    # 3. 安全地创建 DataFrame
    base_grn = pd.DataFrame(
        data=data_mat, 
        index=jaspar_filtered.obs_names, 
        columns=jaspar_filtered.var_names
    )

    adata_rna.obs["leiden"] = "cluster0"

    oracle = co.Oracle()
    adata_rna.X = adata_rna.layers["counts"].copy()

    # sc.pp.highly_variable_genes(adata_rna, n_top_genes=500)
    # adata_rna = adata_rna[:, adata_rna.var.highly_variable].copy()

    if issparse(adata_rna.X):
        print("Detected sparse matrix (AnnData View). Converting to dense array...")
        adata_rna.X = adata_rna.X.toarray()
        
        print(f"Conversion complete. New data type: {type(adata_rna.X)}")
        
    else:
        print("Data is already dense (or not sparse). No conversion needed.")


    TF_TG_mat = (adata_rp_gene_peak.X @ jaspar_filtered.X > 0).astype(int).toarray()

    regulators = list(jaspar_filtered.var_names)

    TF_TG_grn = pd.DataFrame(data = TF_TG_mat, index = adata_rp_gene_peak.obs_names, columns = regulators)
    TF_to_TG_dictionary = {}

    for tf in TF_TG_grn.columns:
        TF_to_TG_dictionary[tf] = TF_TG_grn.index[TF_TG_grn[tf] > 0].tolist()

    ############################################################################################
    # ***************************************** run celloracle  ********************************

    print("Running cell oracle....")

    oracle.import_anndata_as_raw_count(adata=adata_rna, 
                                    cluster_column_name="leiden", 
                                    embedding_name="X_pca", 
                                    transform="log2")


    TG_to_TF_dictionary = co.utility.inverse_dictionary(TF_to_TG_dictionary)
    oracle.import_TF_data(TFdict=TG_to_TF_dictionary)

    # --------------------------------------------------------------------
    #
    # Data imputation
    #
    # --------------------------------------------------------------------
    # knn imputation
    # Perform PCA
    oracle.perform_PCA()

    # Select important PCs
    plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
    
    n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
    plt.axvline(n_comps, c="k")
    plt.show()
    print(n_comps)
    
    n_comps = min(n_comps, 50)

    n_cell = oracle.adata.shape[0]
    k = int(0.025*n_cell)
    oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8, b_maxl=k*4, n_jobs=4)

    # --------------------------------------------------------------------
    #
    # Infer GRNs
    #
    # --------------------------------------------------------------------
    
    links = oracle.get_links(cluster_name_for_GRN_unit="leiden", alpha=10, verbose_level=10)

    grn_pred_true = links.links_dict["cluster0"]

    ############################################################################################
    # ***************************************** save result  ********************************

    # --------------------------------------------------------------------
    print("Save results...")
    if not os.path.exists(output_path + "celloracle_results"):
        os.makedirs(output_path + "celloracle_results")

    # save cluster assignment result
    cluster_assign = adata_rna.obs[["leiden"]]
    cluster_assign.to_csv(output_path + "celloracle_results/cluster_assignment.csv")

    # save grn inference result
    for cluster_id in links.links_dict.keys():
        # direct output of celloracle
        grn_cluster = links.links_dict[cluster_id]
        grn_cluster.to_csv(output_path + "celloracle_results/grn_df_" + str(cluster_id) + ".csv")

        # the coeff mean can be treated as the edge weight of the grn
        grn_coef_mean = pd.DataFrame(data = 0, index = adata_rna.var.index.values, columns = adata_rna.var.index.values)
        for i in range(grn_cluster.shape[0]):
            source = grn_cluster.loc[i, "source"]
            target = grn_cluster.loc[i, "target"]
            # row source, column target
            grn_coef_mean.loc[source, target] = grn_cluster.loc[i, "coef_mean"]
        # save results
        np.savetxt(fname = output_path + "celloracle_results/grn_coef_mean_" + str(cluster_id) + ".txt", X = grn_coef_mean.values)

    print("********************** Finished *******************************************")


