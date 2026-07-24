    
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

sys.path.append("/home/wuyan/dygmamba_project/model/dygmamba/src")

from pdata.data_preprocess import build_tf_peak_network, build_hic_peak_gene_network, build_tf_gene_network_from_anndata
from pdata.data_preprocess import filter_jaspar_tf
from pdata.data_preprocess import calculate_rp_250kb, calculate_peak_peak_rp
from pdata.data_preprocess import analyze_score_distribution
from pdata.data_preprocess import rna_preprocess, atac_preprocess
from pdata.data_preprocess import dataframe_to_anndata_sparse
from pdata.data_preprocess import find_zero_sum_elements
from pdata.hic_data_process import read_hic_data
from pdata.data_read import read_fimo_jaspar, load_scenhancer_gold_standard
    
if __name__ == "__main__":
    
    data_root= "/home/wuyan/dygmamba_project/"
    
    cell_type = "HELA"
    
    hic_file = data_root + "data/hic/scenhancer/Hela-S3_EP.txt"

    data_path = data_root + "data/cell_line/" + cell_type + "/original/"

    output_path = data_root + "data/cell_line/" + cell_type + "/process/"
    
    
    adata_atac_file = output_path + "atac_processed.h5ad"
    adata_rna_file = output_path + "rna_processed.h5ad"
    
    adata_rna = ad.read_h5ad(adata_rna_file)
    adata_atac = ad.read_h5ad(adata_atac_file)
    
    gene_info = pd.read_pickle(output_path + "gene_info_filtered.pkl")
    
    
    
      
    ##### ******************************************************************************************
    print("\n ********************---read jaspar---***************************** \n")
    
    meme_file = data_root + "data/jaspar/JASPAR2026_CORE_vertebrates_non-redundant_pfms.meme" 
    
    jaspar_fimo = output_path + "fimo_out/fimo.txt"

    jaspar_df = read_fimo_jaspar(jaspar_fimo, meme_file)  

    jaspar_df.to_pickle(output_path + "jaspar_df.pkl")

    print("\n ****************** Finished ****************** \n")

    print("Data save in: " + output_path + "jaspar_df.pkl")
    
    
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
    print("\n ********************---read chip-seq---***************************** \n")
    from pdata.data_read import read_cell_type_chipseq
    
    chip_seq_dir = data_root + "data/TF_ChIP_seq/" + cell_type + "/"
    

        
    tf_chip_seq_scenic = read_cell_type_chipseq(cell_type, chip_seq_dir, output_path)
        

    
    ##### ******************************************************************************************
    print("\n ********************---process chip-seq---***************************** \n")



    tf_chip_seq_scenic = pd.read_parquet(output_path + "combined_chip_seq.parquet")
    
    adata_atac = ad.read_h5ad(output_path + "atac_processed.h5ad")
    
    # 读取整个文件
    tf_chip_df = tf_chip_seq_scenic[["chrom", "start", "end", "tf_name"]].copy()

    tf_peak_network = build_tf_peak_network(adata_atac, tf_chip_df) # obs: peak, var: TF

    tf_peak_network.write_h5ad(output_path + "tf_peak_network.h5ad")
    
    print("TF-peak network saved in: " + output_path + "tf_peak_network.h5ad")
  
    
    ##### ******************************************************************************************
    print("\n ********************---read hic---***************************** \n")
    
    hic_data_df = load_scenhancer_gold_standard(hic_file, score_threshold=0.5)

    hic_data_df.to_pickle(output_path + 'hic_data_new.pkl')
    
    print("Data save in: " + output_path + 'hic_data_new.pkl')
    
    ##### ******************************************************************************************
    print("\n ********************---process hi-c---***************************** \n")

    hic_data_df = pd.read_pickle(output_path + 'hic_data_new.pkl')

    hic_data_df['PeakID'] = hic_data_df['chrom'].astype(str) + '-' + \
    hic_data_df['region_start'].astype(str) + '-' + hic_data_df['region_end'].astype(str)
    hic_data_df.rename(columns={'chrom':'chr', 'region_start':'start', 
                                'region_end':'end', 'gene_name':'gene',
                                'abc_score':'hic_score'}, inplace= True)

    hic_data_df = hic_data_df[hic_data_df['gene'].isin(set(adata_rna.var_names))]

    from pdata.data_preprocess import build_hic_peak_gene_network_schic

    peak_gene_df, peak_gene_grn = build_hic_peak_gene_network_schic(adata_atac, hic_data_df, score_col="hic_score") 
        
    # obs: peaks, var: gene 
    
    peak_gene_grn.write_h5ad(output_path + "peak_gene_network.h5ad")
    peak_gene_df.to_pickle(output_path + "peak_gene_df.pkl")
    
    print("Hi-C peak-gene network saved in: " + output_path + "peak_gene_network.h5ad")
    
    ##### ******************************************************************************************
    

    adata_tf_gene = build_tf_gene_network_from_anndata(tf_peak_network, peak_gene_grn)

    adata_tf_gene.write_h5ad(output_path + "tf_gene_network.h5ad")
    print("TF-gene network saved in: " + output_path + "tf_gene_network.h5ad")



