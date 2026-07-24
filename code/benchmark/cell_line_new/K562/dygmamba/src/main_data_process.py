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
import muon.atac as ma  # 用于 TF-IDF
import networkx as nx
from matplotlib import rcParams
import sys
import scipy.io as sio

sys.path.append("/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src")


from pdata.data_preprocess import build_tf_peak_network, build_hic_peak_gene_network, build_tf_gene_network_from_anndata
from pdata.data_preprocess import filter_jaspar_tf
from pdata.data_preprocess import calculate_rp_250kb, calculate_peak_peak_rp
from pdata.data_preprocess import analyze_score_distribution
from pdata.data_preprocess import rna_preprocess, atac_preprocess
from pdata.data_preprocess import dataframe_to_anndata_sparse
from pdata.data_preprocess import find_zero_sum_elements



def regulation_stat():
    
    output_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/process/"
    
    adata_rp = ad.read_h5ad(output_path + "peak_gene_rp_network.h5ad")

    adata_peak_rp = ad.read_h5ad(output_path + "peak_peak_rp_network.h5ad")

    binary_adata_rp = ad.read_h5ad(output_path + "binary_peak_gene_rp_network.h5ad")

    binary_adata_peak_rp = ad.read_h5ad(output_path + "binary_peak_peak_rp_network.h5ad")
    
    # 1. 把所有数据放入字典，方便批量处理
    adatas = {
        "adata_rp": adata_rp,
        "adata_peak_rp": adata_peak_rp,
        "binary_adata_rp": binary_adata_rp,
        "binary_adata_peak_rp": binary_adata_peak_rp
    }

    summary_list = []

    print(f"{'Dataset':<25} | {'Shape (Rows x Cols)':<20} | {'Avg Row Non-Zeros':<20} | {'Avg Col Non-Zeros':<20}")
    print("-" * 95)

    for name, adata in adatas.items():
        
        if sparse.issparse(adata.X):
            row_counts = adata.X.getnnz(axis=1)
        else:
            row_counts = np.count_nonzero(adata.X, axis=1)
            
        # 找出非零的行索引 (即 count > 0 的行)
        # 这就是您要的“记录哪些行的和是非零的”
        active_row_mask = row_counts > 0
        active_rows = adata.obs_names[active_row_mask].tolist()
        
        print(f"  - 行总数: {adata.n_obs}")
        print(f"  - 非零行数 (有连接的行): {len(active_rows)}")
        print(f"  - 全零行数 (无连接的行): {adata.n_obs - len(active_rows)}")

        # ---------------------------
        # 2. 分析列 (Var / Cols)
        # ---------------------------
        # 计算每一列的非零个数
        if sparse.issparse(adata.X):
            col_counts = adata.X.getnnz(axis=0)
        else:
            col_counts = np.count_nonzero(adata.X, axis=0)
            
        # 找出非零的列索引
        active_col_mask = col_counts > 0
        active_cols = adata.var_names[active_col_mask].tolist()
        
        
        # 记录详细统计供后续查看
        summary_list.append({
            "Dataset": name,
            "Row_Total": adata.n_obs,
            "Row_Num": len(active_rows),
            "Col_Total": adata.n_vars,
            "Col_NNZ": len(active_cols),
            "Num links": adata.X.nnz, 
            "Sparsity": 100 * adata.X.nnz / (adata.X.shape[0] * adata.shape[1])
        })

    # 如果需要详细表格
    df_summary = pd.DataFrame(summary_list)

    print(df_summary)


################################################################################################
##### ******************************************************************************************
################################################################################################
    
def main_process_V1(): 

    cell_type = "GM12878"
    output_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/" + cell_type + "/process/"


    ################################################################################################
    ##### ******************************************************************************************
    ################################################################################################

    print("\n ********************---process rna data---***************************** \n")

    adata_rna = ad.read_h5ad(output_path + "rna_origin.h5ad")

    gtf_df = pd.read_pickle(output_path + "gene_info_data.pkl")

    adata_rna_file = output_path + "rna_processed.h5ad"

    gene_info_file = output_path + "gene_info_filtered.pkl"

    adata_rna, gene_info = rna_preprocess(adata_rna, gtf_df, adata_rna_file, gene_info_file)

    ##### ******************************************************************************************

    print("\n ********************---process atac data---***************************** \n")

    adata_atac = ad.read_h5ad(output_path + "atac_origin.h5ad")

    tf_chip_seq_scenic = pd.read_parquet(output_path + "combined_chip_seq.parquet")

    hic_data_df = pd.read_pickle(output_path + 'hic_data_new.pkl')

    gene_info = pd.read_pickle(output_path + "gene_info_filtered.pkl")

    peak_filter_file = output_path + "atac_peak_filtering_results.pkl"

    adata_atac_file = output_path + "atac_processed.h5ad" 

    adata_atac = atac_preprocess(adata_atac, tf_chip_seq_scenic, 
                                 hic_data_df, gene_info, 
                                 peak_filter_file, adata_atac_file)

    ##### ******************************************************************************************
    print("\n ********************---process jaspar---***************************** \n")

    # # 读取整个文件
    jaspar_data = pd.read_pickle(output_path + "jaspar_df.pkl")
    
    jaspar_anndata = dataframe_to_anndata_sparse(jaspar_data, obs_col="sequence_name", var_col="TF_Symbol")
    
    jaspar_anndata.obs_names = jaspar_anndata.obs_names.str.replace(':', '-', regex=False)
    
    jaspar_data_processed = filter_jaspar_tf(jaspar_anndata)

    jaspar_data_processed.write_h5ad(output_path + "jaspar_data_processed.h5ad")

    ##### ******************************************************************************************
    print("\n ********************---process chip-seq---***************************** \n")

    adata_atac = ad.read_h5ad(output_path + "atac_processed.h5ad")
    
    # 读取整个文件
    tf_chip_df = tf_chip_seq_scenic[["chrom", "start", "end", "tf_name"]].copy()

    tf_peak_network = build_tf_peak_network(adata_atac, tf_chip_df) # obs: peak, var: TF

    tf_peak_network.write_h5ad(output_path + "tf_peak_network.h5ad")

    ##### ******************************************************************************************
    print("\n ********************---process hi-c---***************************** \n")

    hic_data_df = pd.read_pickle(output_path + 'hic_data_new.pkl')

    gene_info = pd.read_pickle(output_path + "gene_info_filtered.pkl")

    peak_gene_df, peak_gene_grn = build_hic_peak_gene_network(adata_atac, hic_data_df, 
                                                              gene_info, score_col="contact_count") 
    # obs: peaks, var: gene 
    
    peak_gene_grn.write_h5ad(output_path + "peak_gene_network.h5ad")
    peak_gene_df.to_pickle(output_path + "peak_gene_df.pkl")
    
    ##### ******************************************************************************************
    adata_tf_gene = build_tf_gene_network_from_anndata(tf_peak_network, peak_gene_grn)


    adata_tf_gene.write_h5ad(output_path + "tf_gene_network.h5ad")


    ##### ******************************************************************************************
    print("\n ********************---process prior network---***************************** \n")    

    adata_rp = calculate_rp_250kb(
        adata_atac=adata_atac, 
        adata_rna=adata_rna, 
        gene_info_df=gene_info, # 您的 DataFrame
        decay_dist=50000,       # 衰减参数：50kb (推荐)
        max_range=250000        # 硬性限制：250kb
    )    

    adata_peak_rp = calculate_peak_peak_rp(adata_atac, 
                            decay_distance=50000, # 推荐 50kb
                            max_range=250000)

    adata_rp.write_h5ad(output_path + "peak_gene_rp_network.h5ad")
    adata_peak_rp.write_h5ad(output_path + "peak_peak_rp_network.h5ad")

    rp_stat = analyze_score_distribution(adata_rp)
    rp_threshold = rp_stat["q0.75"]

    rp_peak_stat = analyze_score_distribution(adata_peak_rp)
    rp_peak_threshold = rp_peak_stat["q0.75"]
    
    flag = 1
    if flag==1:
        rp_threshold = 0
        rp_peak_threshold = 0
    
    binary_mask = adata_rp.X > rp_threshold
    adata_rp.X = binary_mask.astype(int)

    peak_binary_mask = adata_peak_rp.X > rp_peak_threshold
    adata_peak_rp.X = peak_binary_mask.astype(int)

    adata_rp.write_h5ad(output_path + "binary_peak_gene_rp_network.h5ad")
    adata_peak_rp.write_h5ad(output_path + "binary_peak_peak_rp_network.h5ad")

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



################################################################################################
##### ******************************************************************************************
################################################################################################
from pdata.data_preprocess import filter_atac_base


def main_process(): 
    
    chip_seq_cell_cluster = ['GM12878', 'HepG2', 'IMR90', 'K562', 'MCF7', 'PC3', 'Panc1']
    hi_cluster = ['GM12878', 'HepG2', 'IMR90', 'K562', 'HCT116']

    cell_type = "K562"
    output_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/" + cell_type + "/process/"


    ################################################################################################
    ##### ******************************************************************************************
    ################################################################################################

    print("\n ********************---process rna data---***************************** \n")

    adata_rna = ad.read_h5ad(output_path + "rna_origin.h5ad")

    gtf_df = pd.read_pickle(output_path + "gene_info_data.pkl")

    adata_rna_file = output_path + "rna_processed.h5ad"

    gene_info_file = output_path + "gene_info_filtered.pkl"

    adata_rna, gene_info = rna_preprocess(adata_rna, gtf_df, adata_rna_file, gene_info_file)

    ##### ******************************************************************************************

    print("\n ********************---process atac data---***************************** \n")

    adata_atac = ad.read_h5ad(output_path + "atac_origin.h5ad")
    
    adata_atac = filter_atac_base(adata_atac)
    
    adata_atac_file = output_path + "atac_processed.h5ad" 
    
    adata_atac.write_h5ad(adata_atac_file)
    
    ##### ******************************************************************************************
    print("\n ********************---process prior network---***************************** \n")    

    adata_rp = calculate_rp_250kb(
        adata_atac=adata_atac, 
        adata_rna=adata_rna, 
        gene_info_df=gene_info, # 您的 DataFrame
        decay_dist=50000,       # 衰减参数：50kb (推荐)
        max_range=250000        # 硬性限制：250kb
    )    

    adata_peak_rp = calculate_peak_peak_rp(adata_atac, 
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
        adata_rp_binary = adata_rp_binary[:, ~adata_rp_binary.var_names.isin(zero_peaks)]
        adata_atac = adata_atac[:, ~adata_atac.var_names.isin(zero_peaks)]
        
    if len(zero_genes) > 0:
        print(f"Removing {len(zero_genes)} zero-sum genes from adata_rp")
        adata_rp = adata_rp[ ~adata_rp.obs_names.isin(zero_genes), :]
        adata_rp_binary = adata_rp_binary[ ~adata_rp_binary.obs_names.isin(zero_genes), :]
        adata_rna = adata_rna[:, ~adata_rna.var_names.isin(zero_genes)]

    adata_rp.write_h5ad(output_path + "peak_gene_rp_network.h5ad")
    adata_peak_rp.write_h5ad(output_path + "peak_peak_rp_network.h5ad")
    adata_rp_binary.write_h5ad(output_path + "binary_peak_gene_rp_network.h5ad")
    adata_peak_rp_binary.write_h5ad(output_path + "binary_peak_peak_rp_network.h5ad")
    adata_atac.write_h5ad(adata_atac_file)
    adata_rna.write_h5ad(adata_rna_file)

    ##### ******************************************************************************************
    print("\n ********************---process jaspar---***************************** \n")

    # # 读取整个文件
    jaspar_data = pd.read_pickle(output_path + "jaspar_df.pkl")
    
    jaspar_anndata = dataframe_to_anndata_sparse(jaspar_data, obs_col="sequence_name", var_col="TF_Symbol")
    
    jaspar_anndata.obs_names = jaspar_anndata.obs_names.str.replace(':', '-', regex=False)
    
    jaspar_data_processed = filter_jaspar_tf(jaspar_anndata)

    jaspar_data_processed.write_h5ad(output_path + "jaspar_data_processed.h5ad")
    
    print("Jaspar TF data saved in: " + output_path + "jaspar_data_processed.h5ad")

    ##### ******************************************************************************************
    print("\n ********************---process chip-seq---***************************** \n")

    if cell_type in chip_seq_cell_cluster:
        
        tf_chip_seq_scenic = pd.read_parquet(output_path + "combined_chip_seq.parquet")

        tf_chip_seq_scenic = pd.read_parquet(output_path + "combined_chip_seq.parquet")
        
        adata_atac = ad.read_h5ad(output_path + "atac_processed.h5ad")
        
        # 读取整个文件
        tf_chip_df = tf_chip_seq_scenic[["chrom", "start", "end", "tf_name"]].copy()

        tf_peak_network = build_tf_peak_network(adata_atac, tf_chip_df) # obs: peak, var: TF

        tf_peak_network.write_h5ad(output_path + "tf_peak_network.h5ad")
        
        print("TF-peak network saved in: " + output_path + "tf_peak_network.h5ad")
        
    else:
        print("*"*60)
        print("Cell type not in chip-seq data")

    ##### ******************************************************************************************
    print("\n ********************---process hi-c---***************************** \n")

    if cell_type in hi_cluster:
        
        hic_data_df = pd.read_pickle(output_path + 'hic_data_new.pkl')

        gene_info = pd.read_pickle(output_path + "gene_info_filtered.pkl")

        peak_gene_df, peak_gene_grn = build_hic_peak_gene_network(adata_atac, hic_data_df, 
                                                                gene_info, score_col="contact_count") 
        # obs: peaks, var: gene 
        
        peak_gene_grn.write_h5ad(output_path + "peak_gene_network.h5ad")
        peak_gene_df.to_pickle(output_path + "peak_gene_df.pkl")
        
        print("Hi-C peak-gene network saved in: " + output_path + "peak_gene_network.h5ad")
    
    ##### ******************************************************************************************
    
    if cell_type in chip_seq_cell_cluster and cell_type in hi_cluster:
        adata_tf_gene = build_tf_gene_network_from_anndata(tf_peak_network, peak_gene_grn)

        adata_tf_gene.write_h5ad(output_path + "tf_gene_network.h5ad")
        print("TF-gene network saved in: " + output_path + "tf_gene_network.h5ad")


    
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

    main_process()