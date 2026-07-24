
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
from self_utils.new_regulation import Graph_data, Graph_data_optimized, get_graph_node_feature
from self_utils.new_regulation import RP_AddExonRemovePromoter

# ************************** function define ******************

def check_consistency_cell(adata_rna, adata_atac, pseudotime):
    
    rna_cell = {str(cell).lower().strip() for cell in adata_atac.obs_names.to_list()}
    atac_cell = {str(cell).lower().strip() for cell in adata_rna.obs_names.to_list()}
    pseudotime_cell = {str(cell).lower().strip() for cell in pseudotime.index.to_list()}

    total_cell = list(set(rna_cell) & set(atac_cell) & set(pseudotime_cell))

    adata_atac = adata_atac[total_cell,].copy()
    adata_rna = adata_rna[total_cell,].copy()
    pseudotime = pseudotime[pseudotime.index.str.lower().isin(total_cell)].copy()

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



if __name__ == "__main__":
    # # Data Load
    cell_type = "SK"
    
    data_root = "/home/wuyan/dygmamba_project/"
    
    sub_set = False
    
    data_path = data_root + "data/cell_line/" + cell_type +"/process/"
    
    output_path = data_root + "/data/cell_line/" + cell_type + "/data_dyg/"

    os.makedirs(output_path, exist_ok=True )

    print("start preprocess")
    
    ################################################################################################
    # **************************** Preprocess Data *******************************
    ################################################################################################


    adata_rna = ad.read_h5ad(data_path + "rna_processed.h5ad")

    adata_atac = ad.read_h5ad(data_path + "atac_processed.h5ad")

    pseudotime = pd.read_csv(data_path + "max_cells_lineage_Lineage1_pseudotime.csv")

    pseudotime['cell_barcode'] = pseudotime['cell_barcode'].str.lower()

    pseudotime.set_index("cell_barcode", inplace=True)

    ############################################
    if sub_set == True:
        
        sub_gene_num = 500
        sub_peak_num = 5000
        sub_cell_sep = 1

        adata_atac.var['n_cells'] = adata_atac.X.sum(axis=0).A1
        top_peaks_idx = adata_atac.var['n_cells'].nlargest(sub_peak_num).index
        adata_atac = adata_atac[:, top_peaks_idx].copy()

        genes_to_keep = adata_rna.var.sort_values('highly_variable_rank').head(sub_gene_num).index
        adata_rna = adata_rna[:, genes_to_keep].copy()

        pseudotime = pseudotime.iloc[list(range(0, len(pseudotime), sub_cell_sep)),:]

    # **************************** Preprocess Data *******************************
    # ****************************  统一细胞名称，确定各个数据的细胞名称对应

    adata_rna, adata_atac, pseudotime = check_consistency_cell(adata_rna, adata_atac, pseudotime)


    # **************************** Preprocess Data *******************************
    # ****************************  获取调控潜力数据

    adata_rp_gene_peak = ad.read_h5ad(data_path + "binary_peak_gene_rp_network.h5ad")
    adata_rp_peak = ad.read_h5ad(data_path + "binary_peak_peak_rp_network.h5ad")

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

    node_feature = get_graph_node_feature(adata_rna, adata_atac, pseudotime, node_id)

    with open(output_path + "node_feature_data.pkl", "wb") as f:
        pickle.dump({'node_feature': node_feature}, f)

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
