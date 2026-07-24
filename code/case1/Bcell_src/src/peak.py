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



from pdata.data_read import anndata_to_bed


################################################################################################
##### ******************************************************************************************
################################################################################################


if __name__ == "__main__":
    
    data_root = "/home/wuyan/dygmamba_project/NewRealPlan/Case1/process/"
    output_path =  data_root + 'Bcell' + '/process/'
    
    adata_atac = ad.read_h5ad(output_path + "atac_processed.h5ad")

    
    peak_bed_df = anndata_to_bed(adata_atac, output_path + "peaks.bed")