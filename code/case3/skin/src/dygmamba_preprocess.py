
import os
import sys
import subprocess

import networkx as nx
import scanpy as sc
import pandas as pd
import anndata as ad
import pickle
import code
import scipy.sparse as sp
import numpy as np
import h5py


import matplotlib.pyplot as plt
import scipy.io
from scipy.io import mmread

from pkg_resources import resource_filename
from datetime import datetime
from collections import Counter

# ************************** self function ******************

sys.path.append("/home/wuyan/dygmamba_project/model/dygmamba/src")

from self_utils.new_regulation import ExtractGeneInfo
from self_utils.new_regulation import RP_peaks
from self_utils.new_regulation import Graph_data, Graph_data_optimized, \
    get_graph_node_feature, get_graph_node_feature_new
from self_utils.new_regulation import RP_AddExonRemovePromoter

# ************************** function define ******************

def check_consistency_cell(adata_rna, adata_atac, pseudotime):
    
    # 建立 小写 → 原始名称 的映射
    rna_lower2orig  = {str(c).lower().strip(): c for c in adata_rna.obs_names}
    atac_lower2orig = {str(c).lower().strip(): c for c in adata_atac.obs_names}
    pt_lower2orig   = {str(c).lower().strip(): c for c in pseudotime.index}

    # 用小写做交集
    common_lower = (set(rna_lower2orig.keys())
                    & set(atac_lower2orig.keys())
                    & set(pt_lower2orig.keys()))
    
    print(f"RNA  cells : {len(rna_lower2orig)}")
    print(f"ATAC cells : {len(atac_lower2orig)}")
    print(f"Pseudotime cells: {len(pt_lower2orig)}")
    print(f"共同 cells : {len(common_lower)}")

    # 用各自原始名称索引
    rna_cells  = [rna_lower2orig[c]  for c in common_lower]
    atac_cells = [atac_lower2orig[c] for c in common_lower]
    pt_cells   = [pt_lower2orig[c]   for c in common_lower]

    adata_rna  = adata_rna[rna_cells].copy()
    adata_atac = adata_atac[atac_cells].copy()
    pseudotime = pseudotime.loc[pt_cells].copy()

    return adata_rna, adata_atac, pseudotime


def calculate_regulation_prior(adata_rna, adata_atac):
    # ## get gene and peak info

    outprefix = "Regulation"
    model = "Enhanced"
    genedistance = 100000

    species = "GRCh38" 
    split_str="-"
    gene_bed = os.path.join(annotation_path, species + "_refgenes.txt")

    genes_list = []
    genes_info = ExtractGeneInfo(gene_bed)
    genes_info_tss = list()
    genes_info_full = list() ### [chrom, tss, start, end, 1, unique_id]
    all_genes = set(adata_rna.var_names)
    seen = set()
    gene_num = 0
    for igene in range(len(genes_info)):
        tmp_gene = genes_info[igene]
        temp_gene = tmp_gene[-1].split("@")[0]
        if temp_gene in all_genes and temp_gene not in seen :
            seen.add(temp_gene)
            genes_list.append(temp_gene)
            genes_info_full.append(tmp_gene + [gene_num])
            genes_info_tss.append([tmp_gene[0], tmp_gene[3], tmp_gene[1], tmp_gene[2]] + tmp_gene[4:] + [gene_num])
            gene_num += 1

    genes_info_Data = pd.DataFrame(genes_info_full)
    genes_info_Data.columns = ['chrom', 'start', 'end', 'tss', 'promoter',
                                'exons', 'length', 'flag 1', 'uid','gene_id']
    genes_info_Data["gene symbol"] = genes_list

    #############################################
    # get peak info
    peaks_list = [f for f in adata_atac.var_names]
    peaks_info = []
    for ipeak, peak in enumerate(peaks_list):
        peaks_tmp = peak.rsplit(split_str, maxsplit=2)
        peaks_info.append([peaks_tmp[0], \
                            (int(peaks_tmp[1])+int(peaks_tmp[2]))/2.0, \
                            int(peaks_tmp[1]), int(peaks_tmp[2]), \
                            0, peak, ipeak])

    peaks_info_Data = pd.DataFrame(peaks_info)
    peaks_info_Data.columns = ['chrom', 'center','start', 'end', 'floag 0', 'peak', 'peak_id']


    # ## 获取 gene-peak 调控潜力 


    id_peak_df = peaks_info_Data[['peak', 'peak_id']]
    id_peak_df.columns = ["peak", "id"]
    id_peak_df.set_index("peak", inplace=True)

    id_gene_df = genes_info_Data[['gene symbol','gene_id']]
    id_gene_df.columns = ['gene', 'id']
    id_gene_df.set_index("gene", inplace=True)

    id_gene_df = id_gene_df[~id_gene_df.index.duplicated(keep="first")]
    id_peak_df = id_peak_df[~id_peak_df.index.duplicated(keep="first")]

    genes_peaks_score_dok = RP_AddExonRemovePromoter(peaks_info, genes_info_full, genes_info_tss, float(genedistance))

    adata_rp_gene_peak = ad.AnnData(X= genes_peaks_score_dok,
                                        obs = pd.DataFrame(index=genes_info_Data["gene symbol"]),
                                        var = pd.DataFrame(index=peaks_info_Data["peak"]))

    adata_rp_gene_peak.X = adata_rp_gene_peak.X.tocsr().sign().astype(int)

    peak_score_dok = RP_peaks(peaks_info, float(genedistance))

    adata_rp_peak = ad.AnnData(X= peak_score_dok,
                                        obs = pd.DataFrame(index=peaks_info_Data["peak"]),
                                        var = pd.DataFrame(index=peaks_info_Data["peak"]))

    adata_rp_peak.X = adata_rp_peak.X.tocsr().sign().astype(int)

    return adata_rp_gene_peak, adata_rp_peak







def sample_cells_by_pseudotime(adata_rna, adata_atac, pseudotime, N=1000):
    """在伪时间轴上等间隔采样 N 个细胞，通过小写 barcode 匹配三者"""
    
    # 建立 小写barcode → 各自原始名称 的映射
    rna_lower2orig  = {str(c).lower().strip(): c for c in adata_rna.obs_names}
    atac_lower2orig = {str(c).lower().strip(): c for c in adata_atac.obs_names}
    pt_lower2orig   = {str(c).lower().strip(): c for c in pseudotime.index}
    
    # 三者交集（小写）
    common_lower = list(
        set(rna_lower2orig) & set(atac_lower2orig) & set(pt_lower2orig)
    )
    
    if len(common_lower) <= N:
        print(f"共同细胞数 ({len(common_lower)}) ≤ N ({N})，跳过采样，返回全部共同细胞")
        rna_cells  = [rna_lower2orig[c]  for c in common_lower]
        atac_cells = [atac_lower2orig[c] for c in common_lower]
        pt_cells   = [pt_lower2orig[c]   for c in common_lower]
        
        adata_rna  = adata_rna[rna_cells].copy()
        adata_atac = adata_atac[atac_cells].copy()
        pseudotime = pseudotime.loc[pt_cells].copy()
        
        unified = [c for c in common_lower]
        adata_rna.obs_names  = unified
        adata_atac.obs_names = unified
        pseudotime.index     = unified
        return adata_rna, adata_atac, pseudotime
    
    # 以 pseudotime 的值为基准做等间隔采样
    pt_vals = {c: pseudotime.loc[pt_lower2orig[c], 'pseudotime'] for c in common_lower}
    pt_min, pt_max = min(pt_vals.values()), max(pt_vals.values())
    targets = np.linspace(pt_min, pt_max, N)
    
    remaining = set(common_lower)
    selected = []
    
    for t in targets:
        if not remaining:
            break
        best = min(remaining, key=lambda c: abs(pt_vals[c] - t))
        selected.append(best)
        remaining.remove(best)
    
    # 按伪时间排序
    selected = sorted(selected, key=lambda c: pt_vals[c])
    
    # 用各自原始名称索引
    rna_cells  = [rna_lower2orig[c]  for c in selected]
    atac_cells = [atac_lower2orig[c] for c in selected]
    pt_cells   = [pt_lower2orig[c]   for c in selected]
    
    adata_rna  = adata_rna[rna_cells].copy()
    adata_atac = adata_atac[atac_cells].copy()
    pseudotime = pseudotime.loc[pt_cells].copy()
    
    # 统一 barcode 为小写
    adata_rna.obs_names  = selected
    adata_atac.obs_names = selected
    pseudotime.index     = selected
    
    print(f"共同细胞: {len(common_lower)}")
    print(f"采样后: {len(selected)} 细胞")
    
    return adata_rna, adata_atac, pseudotime


if __name__ == "__main__":

    
    sub_set = False
    
    data_root = "/home/wuyan/dygmamba_project/NewRealPlan/case3/data/process/"
    traj = 'skin'
    data_path = data_root + traj + "/process/"
    
    output_path = data_path

    os.makedirs(output_path, exist_ok=True)

    print("start preprocess")
    
    ################################################################################################
    # **************************** Preprocess Data *******************************
    ################################################################################################


    adata_rna = ad.read_h5ad(data_path + "rna_processed.h5ad")

    adata_atac = ad.read_h5ad(data_path + "atac_processed.h5ad")

    pseudotime = pd.read_csv(data_path + "avg_lineage_pseudotime.csv")

    pseudotime['cell_barcode'] = pseudotime['cell_barcode'].str.lower()

    pseudotime.set_index("cell_barcode", inplace=True)

    # **************************** Preprocess Data *******************************
    # ****************************  统一细胞名称，确定各个数据的细胞名称对应

    adata_rna, adata_atac, pseudotime = check_consistency_cell(adata_rna, adata_atac, pseudotime)

    # 建立小写→伪时序的映射
    pt_lower = pseudotime.copy()
    pt_lower.index = pt_lower.index.str.lower().str.strip()

    # 按小写对齐
    rna_lower = adata_rna.obs_names.str.lower().str.strip()
    atac_lower = adata_atac.obs_names.str.lower().str.strip()

    adata_rna.obs["pseudotime"]  = pt_lower['pseudotime'].reindex(rna_lower).values
    adata_atac.obs["pseudotime"] = pt_lower['pseudotime'].reindex(atac_lower).values

    # 检查有没有没对上的
    n_nan = adata_rna.obs["pseudotime"].isna().sum()
    if n_nan > 0:
        print(f"[警告] {n_nan} 个细胞没有匹配到伪时序")

    # **************************** Preprocess Data *******************************
    # ****************************  获取调控潜力数据

    adata_rp_gene_peak = ad.read_h5ad(data_path + "binary_peak_gene_rp_network.h5ad")
    adata_rp_peak = ad.read_h5ad(data_path + "binary_peak_peak_rp_network.h5ad")

    # ============================================================
    # Peak 过滤：只保留与基因有调控关系的 peak
    # ============================================================
    print("=" * 50)
    print("Peak 过滤（减少计算量）")
    print(f"  过滤前 ATAC peaks: {adata_atac.shape[1]}")
    
    # 策略 1: 只保留在 RP 网络中与至少 1 个基因有连接的 peak
    # adata_rp_gene_peak: obs=genes, var=peaks, 非零表示有调控关系
    connected_peaks = set()
    rp_csc = adata_rp_gene_peak.X.tocsc()
    for j in range(rp_csc.shape[1]):
        if rp_csc.getcol(j).nnz > 0:
            connected_peaks.add(adata_rp_gene_peak.var_names[j])
    
    # 取交集
    peaks_to_keep = list(connected_peaks & set(adata_atac.var_names))
    print(f"  与基因有调控连接的 peak: {len(peaks_to_keep)}")
    
    # 策略 2: 如果还是太多（>15000），进一步按 variance 过滤
    MAX_PEAKS = 10000
    if len(peaks_to_keep) > MAX_PEAKS:
        # 计算每个 peak 跨细胞的方差
        atac_sub = adata_atac[:, peaks_to_keep]
        if sp.issparse(atac_sub.X):
            # 稀疏矩阵的方差计算
            mean = np.array(atac_sub.X.mean(axis=0)).flatten()
            mean_sq = np.array(atac_sub.X.power(2).mean(axis=0)).flatten()
            var = mean_sq - mean**2
        else:
            var = np.var(atac_sub.X, axis=0)
        
        # 取 top MAX_PEAKS 高变异 peak
        top_idx = np.argsort(var)[-MAX_PEAKS:]
        peaks_to_keep = [peaks_to_keep[i] for i in top_idx]
        print(f"  按 variance 筛选后: {len(peaks_to_keep)}")
    
    # 过滤
    adata_atac = adata_atac[:, peaks_to_keep].copy()
    adata_rp_gene_peak = adata_rp_gene_peak[:, 
        [p for p in adata_rp_gene_peak.var_names if p in set(peaks_to_keep)]].copy()
    adata_rp_peak = adata_rp_peak[
        [p for p in adata_rp_peak.obs_names if p in set(peaks_to_keep)], :]
    adata_rp_peak = adata_rp_peak[:,
        [p for p in adata_rp_peak.var_names if p in set(peaks_to_keep)]].copy()
    
    print(f"  过滤后 ATAC peaks: {adata_atac.shape[1]}")
    print(f"  节点总数: {adata_rna.shape[1]} genes + {adata_atac.shape[1]} peaks "
          f"= {adata_rna.shape[1] + adata_atac.shape[1]}")
    print("=" * 50)
    
    # **************************** Preprocess Data *******************************
    # ****************************  统一基因和peak
    print("******************* Consistency Gene ***************************")
    print(f"Before filter: adata_rna have gene: {adata_rna.shape[1]}, \
        adata_rp_gene_peak have gene: {adata_rp_gene_peak.shape[0]}")

    total_gene= set(adata_rna.var_names) & set(adata_rp_gene_peak.obs_names)
    adata_rna = adata_rna[:,list(total_gene)].copy()
    adata_rp_gene_peak = adata_rp_gene_peak[list(total_gene),:].copy()

    print(f"After filter: adata_rna have gene: {adata_rna.shape[1]}, \
        adata_rp_gene_peak have gene: {adata_rp_gene_peak.shape[0]}")


    print("******************* Consistency Peak ***************************")

    print(f"Before filter: adata_atac have peak: {adata_atac.shape[1]}")
    print(f"adata_rp_gene_peak have peak: {adata_rp_gene_peak.shape[0]}")
    print(f"adata_rp_peak have peak: {adata_rp_peak.shape[1]}")

    total_peak= set(adata_atac.var_names) & set(adata_rp_gene_peak.var_names) & set(adata_rp_peak.var_names)
    adata_atac = adata_atac[:,list(total_peak)].copy()
    adata_rp_gene_peak = adata_rp_gene_peak[:, list(total_peak)].copy()

    adata_rp_peak = adata_rp_peak[:, list(total_peak)].copy()
    adata_rp_peak = adata_rp_peak[list(total_peak), :].copy()

    print(f"After filter: adata_atac have gene: {adata_atac.shape[1]}")
    print(f"adata_rp_gene_peak have gene: {adata_rp_gene_peak.shape[0]}")
    print(f"adata_rp_peak have peak: {adata_rp_peak.shape[1]}")

    # **************************** Preprocess Data *******************************
    # ****************************  统计所有的节点

    # get gene data
    gene_names = adata_rna.var_names
    df_genes = pd.DataFrame({
        "name" : list(gene_names),
        "type" : "gene"
    })
    df_genes = df_genes.sort_values(by = "name", ascending = True)

    ########################
    # get peak data
    peak_names = adata_atac.var_names
    df_peaks = pd.DataFrame({
        "name" : list(peak_names),
        "type" : "peak"
    })
    df_peaks = df_peaks.sort_values(by = "name", ascending = True)

    node_id =  pd.concat([df_genes, df_peaks], ignore_index=True)


    ############################################################################
    # ********************************* Data save
    adata_atac.write_h5ad(data_path + "atac_processed.h5ad")
    adata_rna.write_h5ad(data_path + "rna_processed.h5ad")
    adata_rp_gene_peak.write_h5ad(data_path + "rp_gene_peak.h5ad")
    adata_rp_peak.write_h5ad(data_path + "rp_peak.h5ad")
    pseudotime.to_pickle(data_path + "cell_pseudotime.pkl")
    node_id.to_pickle(output_path + "node_id.pkl")

    print("********************* Finished Preprocess and Save Data *******************************")
    print(adata_atac)
    print(adata_rna)
    # ###############################################################################################
    # **************************** Comprehensive Data Analysis*******************************
    # ###############################################################################################
    # 综合分析

    adata_atac = ad.read_h5ad(data_path + "atac_processed.h5ad")

    adata_rna = ad.read_h5ad(data_path + "rna_processed.h5ad")

    adata_rp_gene_peak = ad.read_h5ad(data_path + "rp_gene_peak.h5ad")

    adata_rp_peak = ad.read_h5ad(data_path + "rp_peak.h5ad")

    pseudotime = pd.read_pickle(data_path + "cell_pseudotime.pkl")

    node_id = pd.read_pickle(output_path + "node_id.pkl")


    ############################################################################
    # ********************************* 构建图数据

    print("********************* strat construct graph data *******************************")

    matrix_path = output_path + "node_feat_matrix.dat"
    meta_path   = output_path + "node_feat_meta.pkl"
    node_feature = get_graph_node_feature_new(
        adata_rna, adata_atac, pseudotime, node_id,
        matrix_path=matrix_path, meta_path=meta_path
    )

    Graph_df, edge_feature, edge_label, edge_records = Graph_data_optimized(adata_rna, adata_atac, pseudotime,
                        node_id, adata_rp_gene_peak, adata_rp_peak)

    Graph_df["Unnamed"] = Graph_df.index
    name_list = ["Unnamed", "source_node", "target_node", "time", "label", "edge_idx"]
    New_Graph = Graph_df[name_list].copy()
    New_Graph.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx']

    print(f"Edge feature:\n {edge_feature[0:10]}")

    print(f"Edge label: \n {edge_label[0:10]}")

    print(f"edge records: \n {edge_records[0:10]}")

    Graph_df.to_pickle(output_path + "Graph_df.pkl")

    feat_path = output_path + "edge_features.npy"
    edge_label_path = output_path + "edge_labels.npy"
    np.save(feat_path, edge_feature)
    np.save(edge_label_path, edge_label)

    with open(output_path + "edge_records_data.pkl", "wb") as f:
        pickle.dump({'edge_records': edge_records}, f)
    
    print("*************** Finished ****************************")
