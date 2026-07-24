import os
import glob
import re
import time
import code
import numpy as np
import pandas as pd
import anndata
import pyfaidx
import pybedtools
from pybedtools import BedTool
import anndata as ad
import scipy.sparse as sp_sparse
from scipy import sparse
import pickle
import scanpy as sc
import networkx as nx
from matplotlib import rcParams
import sys
import scipy.io as sio


sys.path.append("/home/wuyan/dygmamba_project/model/dygmamba/src")


from pdata.data_read import read_genenotation


from pdata.data_read import read_rna_from_txt, read_atac_featurecounts, anndata_to_bed
from pdata.data_preprocess import filter_atac_base

from pdata.data_preprocess import calculate_rp_distance, calculate_peak_peak_rp_distance
from pdata.data_preprocess import analyze_score_distribution
from pdata.data_preprocess import rna_preprocess, atac_preprocess
from pdata.data_preprocess import dataframe_to_anndata_sparse
from pdata.data_preprocess import find_zero_sum_elements





def trajectory_filter(output_path, trajectory):
    # 全量数据路径
    full_rna_path  = output_path +  "rna_origin.h5ad"
    full_atac_path = output_path +  "atac_origin.h5ad"


    # 细胞类型列名
    cell_type_col = "cell_type"

    # 是否清洗离群细胞
    remove_outliers = True
    outlier_percentile = 90  # 保留 90%，去掉最远的 10%

    ######################################################################

    print("=" * 60)
    print(f"轨迹筛选: {' → '.join(trajectory)}")
    print("=" * 60)

    # ---- 1. 加载全量数据 ----
    print("\n[1/4] 加载全量数据...")
    adata_rna = ad.read_h5ad(full_rna_path)
    adata_atac = ad.read_h5ad(full_atac_path)
    print(f"  全量 RNA:  {adata_rna.shape}")
    print(f"  全量 ATAC: {adata_atac.shape}")

    # ---- 2. 筛选轨迹细胞 ----
    print(f"\n[2/4] 筛选细胞类型: {trajectory}")
    mask = adata_rna.obs[cell_type_col].isin(trajectory)
    adata_rna = adata_rna[mask].copy()
    adata_atac = adata_atac[mask].copy()

    print(f"  筛选后: {adata_rna.shape[0]} cells")
    for ct in trajectory:
        n = (adata_rna.obs[cell_type_col] == ct).sum()
        print(f"    {ct}: {n}")

    # ---- 3. 清洗离群细胞 ----
    if remove_outliers:
        print(f"\n[3/4] 清洗离群细胞 (保留 {outlier_percentile}%)...")
        
        # 在筛选后的子集上重新做 PCA
        adata_tmp = adata_rna.copy()
        sc.pp.pca(adata_tmp, n_comps=min(30, adata_tmp.shape[0]-1, adata_tmp.shape[1]-1))
        pca = adata_tmp.obsm['X_pca'][:, :20]
        
        keep_mask = np.ones(adata_rna.shape[0], dtype=bool)
        
        for ct in trajectory:
            ct_mask = adata_rna.obs[cell_type_col].values == ct
            ct_pca = pca[ct_mask]
            center = ct_pca.mean(axis=0)
            dists = np.linalg.norm(ct_pca - center, axis=1)
            threshold = np.percentile(dists, outlier_percentile)
            outlier = dists > threshold
            
            # 在全局 mask 中标记
            ct_indices = np.where(ct_mask)[0]
            keep_mask[ct_indices[outlier]] = False
            
            print(f"    {ct}: 去除 {outlier.sum()} / {ct_mask.sum()} 个离群细胞")
        
        adata_rna = adata_rna[keep_mask].copy()
        adata_atac = adata_atac[keep_mask].copy()
        print(f"  清洗后: {adata_rna.shape[0]} cells")
    else:
        print("\n[3/4] 跳过离群清洗")

    adata_rna.write_h5ad(output_path + "rna_origin2.h5ad")
    adata_atac.write_h5ad(output_path + "atac_origin2.h5ad")

    print(f"\n  已保存: {output_path}rna_origin.h5ad  ({adata_rna.shape})")
    print(f"  已保存: {output_path}atac_origin.h5ad ({adata_atac.shape})")






################################################################################################
##### ******************************************************************************************
################################################################################################


def main_process(output_path):
    
    ##### ******************************************************************************************
    print("\n ********************---read gene info---***************************** \n")

    gtf_file_path = "/home/wuyan/dygmamba_project/Real/Claude/Other/case_study/gencode.V49.annotation.gtf" 

    gtf_df = read_genenotation(gtf_file_path)

    gtf_df.to_pickle(output_path + "gene_info_data.pkl")

    print("Successfully svae Gene info data in ", output_path + "gene_info_data.pkl")
    

    ################################################################################################
    ##### ******************************************************************************************
    ################################################################################################
    
    print("\n ********************---process rna data---***************************** \n")

    adata_rna = ad.read_h5ad(output_path + "rna_origin2.h5ad")
    
    sc.pp.pca(adata_rna, n_comps=min(30, adata_rna.shape[0]-1, adata_rna.shape[1]-1))
    sc.pp.neighbors(adata_rna, n_pcs=20)
    sc.tl.umap(adata_rna)

    adata_rna.obsm['X_umap_full'] = adata_rna.obsm['X_umap'].copy()
        # 同时导出 CSV 给 R 读取
    umap_df = pd.DataFrame(
        adata_rna.obsm['X_umap'],
        columns=['UMAP_1', 'UMAP_2'],
        index=adata_rna.obs_names
    )
    umap_df.to_csv(output_path + "umap_coords.csv")
    print(f"  UMAP 已导出: {output_path}umap_coords.csv")

    gtf_df = pd.read_pickle(output_path + "gene_info_data.pkl")

    adata_rna_file = output_path + "rna_processed.h5ad"

    gene_info_file = output_path + "gene_info_filtered.pkl"

    adata_rna, gene_info = rna_preprocess(adata_rna, gtf_df, adata_rna_file, gene_info_file)

    ##### ******************************************************************************************

    print("\n ********************---process atac data---***************************** \n")

    adata_atac = ad.read_h5ad(output_path + "atac_origin2.h5ad")
    
    adata_atac = filter_atac_base(adata_atac)
    
    adata_atac_file = output_path + "atac_processed.h5ad" 
    
    adata_atac.write_h5ad(adata_atac_file)
    
    ##### ******************************************************************************************
    print("\n ********************---process prior network---***************************** \n")    

    adata_rp, adata_rp_dist = calculate_rp_distance(
        adata_atac=adata_atac, 
        adata_rna=adata_rna, 
        gene_info_df=gene_info, # 您的 DataFrame
        decay_dist=50000,       # 衰减参数：50kb (推荐)
        max_range=250000        # 硬性限制：250kb
    )    

    adata_peak_rp, adata_peak_rp_dist = calculate_peak_peak_rp_distance(adata_atac, 
                            decay_distance=50000, # 推荐 50kb
                            max_range=250000)


    rp_stat = analyze_score_distribution(adata_rp)
    rp_threshold = rp_stat["q0.75"]

    rp_peak_stat = analyze_score_distribution(adata_peak_rp)
    rp_peak_threshold = rp_peak_stat["q0.75"]
    
    flag = 1
    if flag==1:
        rp_threshold = 0
        rp_peak_threshold = 0
    
    adata_rp_binary = adata_rp.copy()
    adata_peak_rp_binary = adata_peak_rp.copy()
    
    binary_mask = adata_rp_binary.X > rp_threshold
    adata_rp_binary.X = binary_mask.astype(int)

    peak_binary_mask = adata_peak_rp_binary.X > rp_peak_threshold
    adata_peak_rp_binary.X = peak_binary_mask.astype(int)

    ##### ******************************************************************************************
    zero_peaks, zero_genes = find_zero_sum_elements(adata_rp_binary)
    if len(zero_peaks) > 0:
        print(f"Removing {len(zero_peaks)} zero-sum peaks from adata_rp")
        adata_rp = adata_rp[:, ~adata_rp.var_names.isin(zero_peaks)]
        adata_rp_dist = adata_rp_dist[:, ~adata_rp_dist.var_names.isin(zero_peaks)]
        adata_rp_binary = adata_rp_binary[:, ~adata_rp_binary.var_names.isin(zero_peaks)]
        adata_atac = adata_atac[:, ~adata_atac.var_names.isin(zero_peaks)]
        
    if len(zero_genes) > 0:
        print(f"Removing {len(zero_genes)} zero-sum genes from adata_rp")
        adata_rp = adata_rp[ ~adata_rp.obs_names.isin(zero_genes), :]
        adata_rp_dist = adata_rp_dist[ ~adata_rp_dist.obs_names.isin(zero_genes), :]
        adata_rp_binary = adata_rp_binary[ ~adata_rp_binary.obs_names.isin(zero_genes), :]
        adata_rna = adata_rna[:, ~adata_rna.var_names.isin(zero_genes)]

    adata_rp.write_h5ad(output_path + "peak_gene_rp_network.h5ad")
    adata_rp_dist.write_h5ad(output_path + "peak_gene_rp_dist_network.h5ad")
    adata_peak_rp.write_h5ad(output_path + "peak_peak_rp_network.h5ad")
    adata_rp_binary.write_h5ad(output_path + "binary_peak_gene_rp_network.h5ad")
    adata_peak_rp_binary.write_h5ad(output_path + "binary_peak_peak_rp_network.h5ad")
    adata_atac.write_h5ad(adata_atac_file)
    adata_rna.write_h5ad(adata_rna_file) 
    
    ##### ******************************************************************************************
    print("\n ********************---Convert to R---***************************** \n")

    adata_rna = ad.read_h5ad(output_path + "rna_processed.h5ad")
    adata_atac = ad.read_h5ad(output_path + "atac_processed.h5ad")


    atac_dir = output_path + "R/atac/"
    os.makedirs(atac_dir, exist_ok=True)
    cellinfo = adata_atac.obs
    atacinfo = adata_atac.var
    mtx = adata_atac.X
    cellinfo.to_csv(atac_dir + "cellinfo.csv")
    atacinfo.to_csv(atac_dir + "atacinfo.csv")
    sio.mmwrite(atac_dir + "sparse.mtx", mtx)



    rna_dir = output_path + "R/rna/"
    os.makedirs(rna_dir, exist_ok=True)
    cellinfo = adata_rna.obs
    rnainfo = adata_rna.var
    mtx = adata_rna.layers['counts']
    cellinfo.to_csv(rna_dir + "cellinfo.csv")
    rnainfo.to_csv(rna_dir + "rnainfo.csv")
    sio.mmwrite(rna_dir + "sparse.mtx", mtx)
    

if __name__ == "__main__":
        
    data_root = "/home/wuyan/dygmamba_project/NewRealPlan/Case1/process/"
    output_path =  data_root + 'erythroid' + '/process/'
    
    print("*"*60)
    print("Trajectory filter")
    
    trajectory = ["HSC", "MK/E prog", "Erythroblast"]
    
    trajectory_filter(output_path, trajectory)
    
    print("*"*60)
    print("Data Process")
    
    main_process(output_path)
    
    print("*"*60)
    print("Finished")
    
    os._exit(0)
