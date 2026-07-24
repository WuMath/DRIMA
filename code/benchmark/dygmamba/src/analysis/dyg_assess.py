import logging
import time
from datetime import datetime
import warnings
import shutil
import json
import pickle
import torch
import sys

import os
import pandas as pd
import numpy as np
import networkx as nx
import scanpy as sc
import anndata as ad

import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from itertools import islice

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import sys
sys.path.append("/home/liyang/BioWuYan/dygmamba_project/model/dygmamba")
sys.path.append("/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src")

from src.models.DyGMamba import DyGMamba
from src.models.modules import MergeLayer, MergeLayerTD
from src.utils.load_configs import load_link_prediction_args
from src.utils.DataLoader import get_model_data
from src.utils.DataLoader import get_idx_data_loader
from src.utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from src.utils.utils import get_parameter_sizes
from src.utils.utils import set_random_seed
from src.utils.utils import convert_to_gpu, create_optimizer
from src.utils.EarlyStopping import EarlyStopping
from src.utils.metrics import get_link_prediction_metrics
from src.models.evaluate_models_utils import evaluate_model_link_prediction
from src.models.inference_grn import model_link_prediction
from src.data_preprocess import filter_jaspar_tf, adata_to_dataframe
from src.analysis.assess import dygmamba_assess



if __name__ == "__main__":

    print("********************** start ********************")

    start_time = time.time()  # start the time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start the job")

    # get arguments
    args = load_link_prediction_args(is_evaluation=False)

    print("**********************device********************")
    print(f"Now use device is {args.device}")

    org_data_path = "/home/liyang/BioWuYan/dygmamba_project/data/original/"

    dyg_result_path = "/home/liyang/BioWuYan/dygmamba_project/data/dygmamba/res/result2/"

    assess_result = "/home/liyang/BioWuYan/dygmamba_project/data/assess_result/"
    
    ########################################################################################
    
    adata_atac = ad.read_h5ad(dyg_result_path + "atac.h5ad")

    adata_rna = ad.read_h5ad(dyg_result_path + "rna.h5ad")
    
    benchmark_data_path = org_data_path

    benchmark_tf_gene_grn = ad.read_h5ad(benchmark_data_path + "tf_gene_network.h5ad")
    benchmark_tf_gene_grn = benchmark_tf_gene_grn[:,adata_rna.var_names].copy()

    benchmark_peak_gene_grn = ad.read_h5ad(benchmark_data_path + "peak_gene_network.h5ad")
    benchmark_peak_gene_grn = benchmark_peak_gene_grn[adata_atac.var_names, adata_rna.var_names].copy()

    benchmark_tf_peak_grn = ad.read_h5ad(benchmark_data_path + "tf_peak_network.h5ad")
    benchmark_tf_peak_grn = benchmark_tf_peak_grn[adata_atac.var_names,:]

    benchmark_tf_peak_df = adata_to_dataframe(benchmark_tf_peak_grn)
    benchmark_tf_peak_df = benchmark_tf_peak_df.rename(columns= {"obs":"Peak", "var":"TF"})
    print(benchmark_tf_peak_df)

    benchmark_peak_gene_df = adata_to_dataframe(benchmark_peak_gene_grn)
    benchmark_peak_gene_df = benchmark_peak_gene_df.rename(columns= {"obs":"Peak", "var":"Gene"})
    print(benchmark_peak_gene_df)

    benchmark_tf_gene_df = adata_to_dataframe(benchmark_tf_gene_grn)
    benchmark_tf_gene_df = benchmark_tf_gene_df.rename(columns= {"obs":"TF", "var":"Gene"})
    print(benchmark_tf_gene_df)
    
    benchmark_tf_gene_threshold = 5
    
    
    print(benchmark_peak_gene_grn)
    print(f"Peak-Gene: {benchmark_peak_gene_df['Peak'].nunique()}, \
        {benchmark_peak_gene_df['Gene'].nunique()}, edge: {len(benchmark_peak_gene_df)}")

    print(benchmark_tf_gene_grn)
    print(f"TF-Gene: {benchmark_tf_gene_df['TF'].nunique()}, \
        {benchmark_tf_gene_df['Gene'].nunique()}, edge: {len(benchmark_tf_gene_df)}")

    print(benchmark_tf_peak_grn)
    print(f"Benchmark TF-Peak: {benchmark_tf_peak_df['TF'].nunique()}, \
        {benchmark_tf_peak_df['Peak'].nunique()}, edge: {len(benchmark_tf_peak_df)}")
    
    ########################################################################################
    
    jaspar_tf_region_file = org_data_path + "jaspar_data.h5ad"
    jaspar_data = ad.read_h5ad(jaspar_tf_region_file)
    adata_region_tf = filter_jaspar_tf(jaspar_data)


    coo_matrix = adata_region_tf.X.tocoo()
    tf_peak_df = pd.DataFrame({
        'Peak': adata_region_tf.obs_names[coo_matrix.row],
        'TF': adata_region_tf.var_names[coo_matrix.col],
        'value': coo_matrix.data
    })
    tf_peak_df = tf_peak_df[tf_peak_df["Peak"].isin(set(adata_atac.var_names))]

    print("*"*50)
    print(adata_region_tf)
    print(f"TF-Peak: {tf_peak_df['TF'].nunique()}, {tf_peak_df['Peak'].nunique()}, edge: {len(tf_peak_df)}")
    print("*"*50)
    
    dygmamba_tf_peak_df = tf_peak_df
    dygmamba_tf_peak_df = dygmamba_tf_peak_df.rename(columns={"value":"predict"})

    total_tf = set(benchmark_tf_peak_df["TF"]) & set(dygmamba_tf_peak_df["TF"])
    benchmark_tf_peak_df = benchmark_tf_peak_df[benchmark_tf_peak_df["TF"].isin(total_tf)].copy()
    dygmamba_tf_peak_df = dygmamba_tf_peak_df[dygmamba_tf_peak_df["TF"].isin(total_tf)].copy()

    total_peak = set(adata_atac.var_names)
    benchmark_tf_peak_df = benchmark_tf_peak_df[benchmark_tf_peak_df["Peak"].isin(total_peak)].copy()
    dygmamba_tf_peak_df = dygmamba_tf_peak_df[dygmamba_tf_peak_df["Peak"].isin(total_peak)].copy()


    dyg_merged_tf_peak_data = pd.merge(benchmark_tf_peak_df, dygmamba_tf_peak_df, 
                                       on = ["TF", "Peak"], how="outer").fillna(0)

    print("*"*50)
    print(f"Merged TF-Peak: {dyg_merged_tf_peak_data['TF'].nunique()}, {dyg_merged_tf_peak_data['Peak'].nunique()}, \
        {len(dyg_merged_tf_peak_data)}")

    print(f"Dygmamba TF-Peak: {dygmamba_tf_peak_df['TF'].nunique()}, {dygmamba_tf_peak_df['Peak'].nunique()},\
        edge: {len(dygmamba_tf_peak_df)}")

    print(f"Benchmark TF-Peak: {benchmark_tf_peak_df['TF'].nunique()}, \
        {benchmark_tf_peak_df['Peak'].nunique()}, edge: {len(benchmark_tf_peak_df)}")
    print("*"*50)
    
    benchmark_result = []
    result_type = "binary"
    beta_value = 1

    dyg_y_true = dyg_merged_tf_peak_data["value"].astype(int)
    dyg_y_pre = dyg_merged_tf_peak_data["predict"].astype(int)
    dyg_model_name = "Dygmamba_peak"

    dyg_dict = dygmamba_assess(dyg_y_true, dyg_y_pre, model_name = dyg_model_name, 
                                beta = beta_value, type = result_type, fig_path = assess_result)
    dyg_tf_region_result = pd.DataFrame([dyg_dict])
    print(dyg_tf_region_result)
    ########################################################################################

    feat_path = dyg_result_path + "edge_features.npy"
    edge_label_path = dyg_result_path + "edge_labels.npy"

    Edge_feature = np.load(feat_path, mmap_mode="r")
    Edge_feature = Edge_feature.reshape(-1,1).copy()
    Edge_label = np.load(edge_label_path, mmap_mode="r")
    Edge_label = Edge_label.reshape(-1,1).copy()

    with open(dyg_result_path + "node_feature_data.pkl", "rb") as f:
        load_data = pickle.load(f)

    Node_feature = load_data['node_feature']

    Node_id = pd.read_pickle(dyg_result_path + "node_id.pkl")

    graph_df = pd.read_pickle(dyg_result_path + "Graph_df.pkl")
    graph_df["Unnamed"] = graph_df.index
    name_list = ["Unnamed", "source_node", "target_node", "time", "label", "edge_idx"]
    New_Graph = graph_df[name_list].copy()
    New_Graph.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx']
    
    
    ####################################################################################

    adata_rp_gene_peak = ad.read_h5ad(dyg_result_path + "rp_gene_peak.h5ad")
    prior_peak_gene_df = adata_to_dataframe(adata_rp_gene_peak)

    prior_peak_gene_df = prior_peak_gene_df.rename(columns= {"obs":"Gene", "var":"Peak"})

    print(adata_rp_gene_peak)
    print(f"Prior peak-gene: {prior_peak_gene_df['Peak'].nunique()}, {prior_peak_gene_df['Gene'].nunique()},\
        edge: {len(prior_peak_gene_df)}")
    
    ####################################################################################
    result_path = dyg_result_path + 'my_result_run{run}.npy'
    predict_edge_label = np.load(result_path)

    New_Graph["predict"] = predict_edge_label
    predict_grn = New_Graph.copy()
    
    mapping_series = Node_id["name"]
    predict_grn['source'] = (predict_grn['u'] - 1).map(mapping_series)
    predict_grn['target'] = (predict_grn['i'] - 1).map(mapping_series)

    peak_gene_df = predict_grn[['source', 'target', 'ts','predict']].rename(
        columns={'source': 'Peak', 'target': 'Gene'}
    )
    peak_gene_df = peak_gene_df[~peak_gene_df["Gene"].str.startswith('chr')].copy()

    print("*"*50)
    print(f"Peak-Gene: {peak_gene_df['Peak'].nunique()}, {peak_gene_df['Gene'].nunique()}, edge:{len(peak_gene_df)}")
    print("*"*50)
    
    ####################################################################################
    
    dygmamba_peak_gene_grn = peak_gene_df
    avg_active_peak_gene_grn = dygmamba_peak_gene_grn.groupby(['Peak', 'Gene']).agg(
        avg_ts_weight=('predict', 'mean'),  # 对 weight 列做均值，新列名叫 avg_weight
        avg_total_weight=('predict', 'sum')  # (可选) 建议顺便算个总权重
    ).reset_index()
    avg_active_peak_gene_grn = avg_active_peak_gene_grn[avg_active_peak_gene_grn["Peak"].isin(total_peak)].copy()
    benchmark_peak_gene_df = benchmark_peak_gene_df[benchmark_peak_gene_df["Peak"].isin(total_peak)].copy()

    dyg_merged_peak_gene_data = pd.merge(benchmark_peak_gene_df, avg_active_peak_gene_grn, 
                                         on = ["Gene", "Peak"], how="outer").fillna(0)

    print("*"*50)
    print(f"Merged Peak-Gene: {dyg_merged_peak_gene_data['Peak'].nunique()}, {dyg_merged_peak_gene_data['Gene'].nunique()}, \
        {len(dyg_merged_peak_gene_data)}")

    print(f"Dygmamba Peak-Gene: {avg_active_peak_gene_grn['Peak'].nunique()}, {avg_active_peak_gene_grn['Gene'].nunique()}, \
        edge: {len(avg_active_peak_gene_grn)}")

    print(f"Benchmark Peak-Gene: {benchmark_peak_gene_df['Peak'].nunique()}, \
        {benchmark_peak_gene_df['Gene'].nunique()}, edge: {len(benchmark_peak_gene_df)}")
    print("*"*50)
    
    
    dyg_merged_peak_gene_data["predict"] = (dyg_merged_peak_gene_data["avg_ts_weight"]>0.9).astype(int)

    dyg_y_true = dyg_merged_peak_gene_data["value"].astype(int)
    dyg_y_pre = dyg_merged_peak_gene_data["predict"].astype(int)
    dyg_model_name = "Dygmamba_peak_gene"

    dyg_dict = dygmamba_assess(dyg_y_true, dyg_y_pre, model_name = dyg_model_name, 
                                beta = beta_value, type = result_type, fig_path = assess_result)
    dyg_peak_gene_result = pd.DataFrame([dyg_dict])

    print("*"*50)
    print(dyg_peak_gene_result)
    print(f"merged: {len(dyg_merged_peak_gene_data)}, benchmark peak gene: {len(benchmark_peak_gene_df)},\
        dyg peak gene {len(avg_active_peak_gene_grn)}")
    print(f"Peak-Gene: {dyg_merged_peak_gene_data['Peak'].nunique()}, \
        {dyg_merged_peak_gene_data['Gene'].nunique()}, edge:{len(dyg_merged_peak_gene_data)}")
    print("*"*50)
    
   
    #################################################################################### 
    dygmamba_tf_peak_df = dygmamba_tf_peak_df.rename(columns={"predict":"value"})
    peak_gene_grn = peak_gene_df[peak_gene_df["Peak"].isin(total_peak)]
    merged_df = pd.merge(dygmamba_tf_peak_df, peak_gene_grn, on='Peak')
    
    tf_gene_grn = merged_df.groupby(['TF', 'Gene', 'ts']).agg(
        peak_num=('Peak', 'nunique'),   # 对 Peak 列做去重计数，新列名叫 peak_num
        avg_weight=('predict', 'mean'),  # 对 weight 列做均值，新列名叫 avg_weight
        total_weight=('predict', 'sum')  # (可选) 建议顺便算个总权重
    ).reset_index()

    # 查看结果
    print("*"*50)
    print(f"Dygmamba TF-Peak: {dygmamba_tf_peak_df['TF'].nunique()}, {dygmamba_tf_peak_df['Peak'].nunique()},\
        edge: {len(dygmamba_tf_peak_df)}")
    print(f"Peak-Gene: {peak_gene_df['Peak'].nunique()}, {peak_gene_df['Gene'].nunique()}, edge:{len(peak_gene_df)}")
    print(f"TF-Gene: {tf_gene_grn['TF'].nunique()}, {tf_gene_grn['Gene'].nunique()}, edge: {len(tf_gene_grn)}")

    tf_gene_grn.to_pickle(dyg_result_path + "new_tf_gene_grn_1224.pkl")

    avg_active_tf_gene_grn = tf_gene_grn.groupby(['TF', 'Gene']).agg(
        avg_ts_weight=('total_weight', 'mean'),  # 对 weight 列做均值，新列名叫 avg_weight
        avg_total_weight=('total_weight', 'sum')  # (可选) 建议顺便算个总权重
    ).reset_index()
    avg_active_tf_gene_grn.to_pickle(dyg_result_path + "average_active_tf_gene_grn.pkl")

    pivoted_grn = tf_gene_grn.pivot_table(
        index=['TF', 'Gene'],
        columns='ts',
        values='total_weight',
        fill_value=0
    )
    pivoted_grn['average_active_weight'] = pivoted_grn.mean(axis=1)
    avg_global_tf_gene_grn = pivoted_grn.reset_index()
    avg_global_tf_gene_grn = avg_global_tf_gene_grn[["TF","Gene","average_active_weight"]].copy()
    avg_global_tf_gene_grn.columns.name = None
    avg_global_tf_gene_grn.to_pickle(dyg_result_path + "average_global_tf_gene_grn.pkl")

    print("*"*50)
    print(f"TF-Gene: {tf_gene_grn['TF'].nunique()}, {tf_gene_grn['Gene'].nunique()}, edge: {len(tf_gene_grn)}")
    print(f"avg active TF-Gene: {avg_active_tf_gene_grn['TF'].nunique()}, {avg_active_tf_gene_grn['Gene'].nunique()},\
        edge: {len(avg_active_tf_gene_grn)}")
    print(f"avg global TF-Gene: {avg_global_tf_gene_grn['TF'].nunique()}, {avg_global_tf_gene_grn['Gene'].nunique()},\
        edge: {len(avg_global_tf_gene_grn)}")
    print("*"*50)
    #################################################################################### 
    benchmark_tf_gene_df_new = pd.merge(benchmark_tf_peak_df, benchmark_peak_gene_df, on='Peak')
    benchmark_tf_gene_grn_new = benchmark_tf_gene_df_new.groupby(['TF', 'Gene']).agg(
        peak_num=('Peak', 'nunique'),   # 对 Peak 列做去重计数，新列名叫 peak_num
    ).reset_index()
    benchmark_tf_gene_grn_new = benchmark_tf_gene_grn_new.rename(columns= {"peak_num":"value"})
    
    print("*"*50)
    print(f"Benchmark TF-Gene: {benchmark_tf_gene_df['TF'].nunique()}, \
        {benchmark_tf_gene_df['Gene'].nunique()}, edge: {len(benchmark_tf_gene_df)}")
    print(f"New Benchmark TF-Gene: {benchmark_tf_gene_df_new['TF'].nunique()}, \
        {benchmark_tf_gene_df_new['Gene'].nunique()}, edge: {len(benchmark_tf_gene_df_new)}")
    print(f"New count Benchmark TF-Gene: {benchmark_tf_gene_grn_new['TF'].nunique()}, \
        {benchmark_tf_gene_grn_new['Gene'].nunique()}, edge: {len(benchmark_tf_gene_grn_new)}")
    print("*"*50)
    
    
    #################################################################################### 
    benchmark_tf_gene_grn_new = benchmark_tf_gene_grn_new[benchmark_tf_gene_grn_new["TF"].isin(total_tf)].copy()

    dyg_avg_active_tf_gene_grn = avg_active_tf_gene_grn
    dyg_avg_global_tf_gene_grn = avg_global_tf_gene_grn
    dyg_avg_active_tf_gene_grn.columns.name = ""
    dyg_avg_active_tf_gene_grn = dyg_avg_active_tf_gene_grn.reset_index()
    dyg_avg_active_tf_gene_grn = dyg_avg_active_tf_gene_grn.drop(["index"],axis = 1)
    dyg_merged_data = pd.merge(benchmark_tf_gene_grn_new, dyg_avg_active_tf_gene_grn, 
                               on = ["TF", "Gene"], how="outer").fillna(0)

    dyg_avg_global_tf_gene_grn.columns.name = ""
    dyg_avg_global_tf_gene_grn = dyg_avg_global_tf_gene_grn.reset_index()
    dyg_avg_global_tf_gene_grn = dyg_avg_global_tf_gene_grn.drop(["index"],axis = 1)
    dyg_global_merged_data = pd.merge(benchmark_tf_gene_grn_new, dyg_avg_global_tf_gene_grn, 
                                      on = ["TF", "Gene"], how="outer").fillna(0)
    
    benchmark_tf_gene_threshold = 200
    threshold_weight_global = 0.2
    threshold_weight_active = 0.2
    benchmark_result = []
    result_type = "binary"
    beta_value = 1

    dyg_merged_data["label"] = (dyg_merged_data["value"] > benchmark_tf_gene_threshold).astype(int)
    dyg_merged_data["predict_label"] = (dyg_merged_data["avg_ts_weight"]> threshold_weight_active).astype(int)
    dyg_merge_grn = dyg_merged_data.copy()
    dyg_y_true = dyg_merge_grn["label"].astype(int)
    dyg_y_pre = dyg_merge_grn["predict_label"].astype(int)
    dyg_model_name = "Dygmamba"
    dyg_dict = dygmamba_assess(dyg_y_true, dyg_y_pre, model_name = dyg_model_name, 
                                beta = beta_value, type = result_type, fig_path = assess_result)

    benchmark_result.append(dyg_dict)

    dyg_global_merged_data["label"] = (dyg_global_merged_data["value"] > benchmark_tf_gene_threshold).astype(int)
    dyg_global_merged_data["predict_label"] = (dyg_global_merged_data["average_active_weight"]> threshold_weight_global).astype(int)
    dyg_global_merge_grn = dyg_global_merged_data.copy()
    dyg_global_y_true = dyg_global_merge_grn["label"].astype(int)
    dyg_global_y_pre = dyg_global_merge_grn["predict_label"].astype(int)
    dyg_global_model_name = "Dygmamba" + "_global"
    dyg_global_dict = dygmamba_assess(dyg_global_y_true, dyg_global_y_pre, model_name = dyg_global_model_name, 
                                beta = beta_value, type = result_type, fig_path = assess_result)

    benchmark_result.append(dyg_global_dict)

    benchmark_result_df = pd.DataFrame(benchmark_result)
    
    active_num = dyg_merged_data["label"].sum(axis=0)
    active_total = len(dyg_merged_data)
    global_num = dyg_global_merged_data["label"].sum(axis=0)
    global_total = len(dyg_global_merged_data)

    print("*"*50)
    print(f"active_num: {active_num}/{active_total}; global num: {global_num}/{global_total}")
    print(benchmark_result_df)
    
    print(f"Benchmark TF-Gene: {benchmark_tf_gene_grn_new['TF'].nunique()}, \
        {benchmark_tf_gene_grn_new['Gene'].nunique()}, edge: {len(benchmark_tf_gene_grn_new)}")

    print(f"Dygmamba Merge TF-Gene: {dyg_merge_grn['TF'].nunique()}, \
        {dyg_merge_grn['Gene'].nunique()}, edge: {len(dyg_merge_grn)}")

    print(f"Dygmamba Global Merge TF-Gene: {dyg_global_merged_data['TF'].nunique()}, \
        {dyg_global_merged_data['Gene'].nunique()}, edge: {len(dyg_global_merged_data)}")

    print(f"Dygmamba Global TF-Gene: {dyg_avg_global_tf_gene_grn['TF'].nunique()}, \
        {dyg_avg_global_tf_gene_grn['Gene'].nunique()}, edge: {len(dyg_avg_global_tf_gene_grn)}")

    print(f"Dygmamba active TF-Gene: {dyg_avg_active_tf_gene_grn['TF'].nunique()}, \
        {dyg_avg_active_tf_gene_grn['Gene'].nunique()}, edge: {len(dyg_avg_active_tf_gene_grn)}")
    print("*"*50)

