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

import os
import gzip
import shutil


sys.path.append("/home/wuyan/dygmamba_project/model/dygmamba/src")




from pdata.data_read import read_rna_from_txt, read_atac_featurecounts, anndata_to_bed
from pdata.data_preprocess import filter_atac_base

from pdata.data_preprocess import calculate_rp_distance, calculate_peak_peak_rp_distance
from pdata.data_preprocess import analyze_score_distribution
from pdata.data_preprocess import rna_preprocess, atac_preprocess
from pdata.data_preprocess import dataframe_to_anndata_sparse
from pdata.data_preprocess import find_zero_sum_elements



def check_gtf_library():
    """诊断 read_gtf 来自哪个库，行为不同"""
    try:
        from pyranges import read_gtf
        print("read_gtf 来自 pyranges — 支持直接读 .gtf.gz")
        return 'pyranges'
    except ImportError:
        pass
    try:
        from gtfparse import read_gtf
        print("read_gtf 来自 gtfparse — 不支持 .gtf.gz，需要先解压")
        return 'gtfparse'
    except ImportError:
        pass
    raise ImportError("找不到 read_gtf，请安装: pip install pyranges 或 pip install gtfparse")

def prepare_gtf(gtf_path: str) -> str:
    """
    处理各种 GTF 格式，返回可直接读取的文件路径。
    支持:
      - .gtf          → 直接返回
      - .gtf.gz       → 解压到同目录，返回解压路径
      - .tar.gz       → 提取内部 genes/genes.gtf，返回路径
    """
    if not os.path.exists(gtf_path):
        raise FileNotFoundError(f"找不到文件: {gtf_path}")

    # 已经是普通 GTF
    if gtf_path.endswith('.gtf') and not gtf_path.endswith('.tar.gz'):
        print(f"GTF 文件就绪: {gtf_path}")
        return gtf_path

    # gzip 压缩的 GTF（.gtf.gz）
    if gtf_path.endswith('.gtf.gz'):
        out_path = gtf_path[:-3]  # 去掉 .gz
        if not os.path.exists(out_path):
            print(f"解压 {gtf_path} → {out_path} ...")
            with gzip.open(gtf_path, 'rb') as f_in, \
                 open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"解压完成: {out_path}")
        else:
            print(f"已存在解压文件: {out_path}")
        return out_path

    # tar.gz（10x 参考包）
    if gtf_path.endswith('.tar.gz'):
        import tarfile
        extract_dir = os.path.dirname(gtf_path)
        # 10x 包内 GTF 的固定路径
        inner_path = None
        print(f"扫描 tar 包内容: {gtf_path}")
        with tarfile.open(gtf_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('genes/genes.gtf'):
                    inner_path = member.name
                    break
        if inner_path is None:
            raise ValueError(f"在 {gtf_path} 中找不到 genes/genes.gtf")

        out_path = os.path.join(extract_dir, inner_path)
        if not os.path.exists(out_path):
            print(f"从 tar 包提取: {inner_path}")
            with tarfile.open(gtf_path, 'r:gz') as tar:
                tar.extract(inner_path, path=extract_dir)
            print(f"提取完成: {out_path}")
        else:
            print(f"已存在提取文件: {out_path}")
        return out_path

    raise ValueError(f"不支持的文件格式: {gtf_path}")


def read_genenotation(gtf_file_path: str):
    try:
        ready_path = prepare_gtf(gtf_file_path)
    except Exception as e:
        print(f"Error 准备 GTF 文件: {e}")
        return None

    try:
        from pyranges import read_gtf
        gr = read_gtf(ready_path)

        # 兼容新旧版本 pyranges
        if hasattr(gr, 'as_df'):
            gtf_df = gr.as_df()          # 新版 pyranges >= 0.0.120
        elif hasattr(gr, 'to_pandas'):
            gtf_df = gr.to_pandas()      # 旧版
        elif hasattr(gr, 'df'):
            gtf_df = gr.df              # 某些中间版本
        else:
            # 终极兜底：直接转 DataFrame
            import pandas as pd
            gtf_df = pd.DataFrame(gr)

        print(f"GTF 读取成功: {gtf_df.shape[0]} 行")
        print(f"列名: {list(gtf_df.columns[:8])}")
        return gtf_df

    except Exception as e:
        print(f"Error 读取 GTF: {e}")
        return None




################################################################################################
##### ******************************************************************************************
################################################################################################


def main_process(output_path, gtf_file_path):
    
    ##### ******************************************************************************************
    print("\n ********************---read gene info---***************************** \n")

    gtf_df = read_genenotation(gtf_file_path)
    
    gtf_df['feature'] = gtf_df['Feature']
    gtf_df['seqname'] = gtf_df['Chromosome']
    gtf_df['strand'] = gtf_df['Strand']
    gtf_df['start'] = gtf_df['Start']
    gtf_df['end'] = gtf_df['End']

    gtf_df.to_pickle(output_path + "gene_info_data.pkl")

    print("Successfully svae Gene info data in ", output_path + "gene_info_data.pkl")
    

    ################################################################################################
    ##### ******************************************************************************************
    ################################################################################################
    
    print("\n ********************---process rna data---***************************** \n")

    adata_rna = ad.read_h5ad(output_path + "rna_origin.h5ad")
    
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
        
    data_root = "/home/wuyan/dygmamba_project/NewRealPlan/case2/data/AD/process/"
    output_path =  data_root + 'model2_WT_Microglia' + '/process/'
    
    gtf_file_path = "/home/wuyan/dygmamba_project/NewRealPlan/case2/data/refdata-cellranger-arc-mm10-2020-A-2.0.0/genes/genes.gtf.gz" 

    
    print("*"*60)
    print("Data Process")
    
    main_process(output_path, gtf_file_path)
    
    print("*"*60)
    print("Finished")
    
    os._exit(0)
