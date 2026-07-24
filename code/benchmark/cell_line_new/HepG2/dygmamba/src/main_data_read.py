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


sys.path.append("/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src")


from pdata.data_read import read_genenotation


from pdata.data_read import read_rna_from_txt, read_atac_featurecounts, anndata_to_bed
from pdata.data_preprocess import filter_atac_base


################################################################################################
##### ******************************************************************************************
################################################################################################

def main_read():
    
    data_root= "/home/liyang/BioWuYan/dygmamba_project/data/"
    
    cell_type = "HepG2"

    data_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/" + cell_type + "/original/"

    output_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/" + cell_type + "/process/"
    
    chip_seq_path = "/home/liyang/BioWuYan/dygmamba_project/data/TF_ChIP_seq/code/combined_chip_seq.parquet"

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



    ##### ******************************************************************************************
    print("\n ********************---read chip-seq---***************************** \n")

    cell_cluster = ['GM12878', 'HepG2', 'IMR90', 'K562', 'MCF7', 'PC3', 'Panc1']

    if cell_type in cell_cluster:
        
        tf_chip_seq_scenic = pd.read_parquet(chip_seq_path)
        
        tf_chip_seq_scenic = tf_chip_seq_scenic[tf_chip_seq_scenic["cell_type"] == cell_type]

        tf_chip_seq_scenic.to_parquet(output_path + "combined_chip_seq.parquet")
        
        print("Data save in: " + output_path + "combined_chip_seq.parquet")
        
    else:
        print("*"*60)
        print("Cell type not in chip-seq data")

    
    ################################################################################################
    ##### ******************************************************************************************
    ################################################################################################
 
    
    

if __name__ == "__main__":
    
    main_read()
