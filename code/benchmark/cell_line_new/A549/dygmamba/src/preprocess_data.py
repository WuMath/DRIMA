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

################################################################################################
##### ******************************************************************************************
################################################################################################

def main_read(data_root, cell_type):

    data_path = data_root + "data/cell_line/" + cell_type + "/original/"

    output_path = data_root + "data/cell_line/" + cell_type + "/process/"

    ##### ******************************************************************************************
    print("\n ****************** read ATAC from txt ****************** \n")

    count_file = data_path + "atac/counts_matrix.txt"

    atac_output_file = output_path + "atac_origin.h5ad"

    adata_atac = read_atac_featurecounts(count_file, atac_output_file)

    adata_atac = filter_atac_base(adata_atac)
    
    peak_bed_df = anndata_to_bed(adata_atac, output_path + "peaks.bed")
    
    print("\n ****************** Finished ****************** \n")


    ##### ******************************************************************************************
    print("\n ********************* read RNA from txt ********************* \n")

    rna_count_directory = data_path + 'rna/cells_gene_count/'

    adata_rna_output_file = output_path + "rna_origin.h5ad"

    adata_rna = read_rna_from_txt(rna_count_directory, adata_rna_output_file)

    print("\n ****************** Finished ****************** \n")


    ##### ******************************************************************************************
    print("\n ********************---read gene info---***************************** \n")

    gtf_file_path = data_path + "gencode.V49.annotation.gtf" 

    gtf_df = read_genenotation(gtf_file_path)

    gtf_df.to_pickle(output_path + "gene_info_data.pkl")

    print("Successfully svae Gene info data in ", output_path + "gene_info_data.pkl")


    
    ################################################################################################
    ##### ******************************************************************************************
    ################################################################################################
 

def main_process(data_root, cell_type):
    ################################################################################################
    ##### ******************************************************************************************
    ################################################################################################
    
    output_path = data_root + "data/cell_line/" + cell_type + "/process/"
    
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

    adata_rp, adata_rp_dist = calculate_rp_distance(
        adata_atac=adata_atac, 
        adata_rna=adata_rna, 
        gene_info_df=gene_info, # 您的 DataFrame
        decay_dist=50000,       # 衰减参数：50kb (推荐)
        max_range=50000        # 硬性限制：250kb
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
        
    data_root= "/home/wuyan/dygmamba_project/"
    
    cell_type = "A549"
    
    print("*"*60)
    print("Data Read")
    
    main_read(data_root, cell_type)
    
    print("*"*60)
    print("Data Process")
    main_process(data_root, cell_type)
    
    print("*"*60)
    print("Finished")
