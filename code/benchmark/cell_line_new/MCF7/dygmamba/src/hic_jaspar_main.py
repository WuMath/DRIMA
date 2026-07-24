    
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

from pdata.hic_data_process import read_hic_data
from pdata.data_read import read_fimo_jaspar
    
if __name__ == "__main__":
    ##### ******************************************************************************************
    print("\n ********************---read hic---***************************** \n")
    
    data_root= "/home/liyang/BioWuYan/dygmamba_project/data/"
    
    cell_type = "MCF7"
    
    data_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/" + cell_type + "/original/"

    output_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/" + cell_type + "/process/"
    
    if cell_type == "K562":
        HIC_FILE = data_root + "hic/K562_ENCFF080DPJ.hic"
    elif cell_type == "HepG2":
        HIC_FILE = data_root + "hic/HepG2_ENCFF020DPP.hic"
    elif cell_type == "IMR90":
        HIC_FILE = data_root + "hic/IMR90_ENCFF685BLG.hic"
    elif cell_type == "GM12878":
        HIC_FILE = data_root + "hic/GM12878_ENCFF053VBX.hic"
    elif cell_type == "HCT116":
        HIC_FILE = data_root + "hic/HCT116_ENCFF750AOC.hic"
    else:
        print("Unknow HIC-FILE for cell type: " + cell_type)
        HIC_FILE = None
    
    if HIC_FILE:

        RESOLUTION = 5000
        NORMALIZATION = 'SCALE'

        hic_data_df = read_hic_data(
            hic_file_path=HIC_FILE,
            resolution=RESOLUTION,
            normalization=NORMALIZATION
        )

        hic_data_df.to_pickle(output_path + 'hic_data_new.pkl')
        print("Data save in: " + output_path + 'hic_data_new.pkl')
    

    ##### ******************************************************************************************

    meme_file = "/home/liyang/BioWuYan/dygmamba_project/data/jaspar/JASPAR2026_CORE_vertebrates_non-redundant_pfms.meme" 
    
    jaspar_fimo = output_path + "fimo_out/fimo.txt"

    jaspar_df = read_fimo_jaspar(jaspar_fimo, meme_file)  

    jaspar_df.to_pickle(output_path + "jaspar_df.pkl")

    print("\n ****************** Finished ****************** \n")

    print("Data save in: " + output_path + "jaspar_df.pkl")

