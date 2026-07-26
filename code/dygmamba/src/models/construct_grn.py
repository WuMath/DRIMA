import logging
import time
from datetime import datetime
import warnings
import shutil
import json
import pickle
import torch
import sys

sys.path.append("/home/liyang/BioWuYan/MethodTest/dygmamba/src")

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

from models.DyGMamba import DyGMamba
from models.modules import MergeLayer, MergeLayerTD

from utils.load_configs import load_link_prediction_args

from utils.DataLoader import get_model_data
from utils.DataLoader import get_idx_data_loader
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.utils import get_parameter_sizes
from utils.utils import set_random_seed
from utils.utils import convert_to_gpu, create_optimizer
from utils.EarlyStopping import EarlyStopping
from utils.metrics import get_link_prediction_metrics
from models.evaluate_models_utils import evaluate_model_link_prediction
from models.inference_grn import model_link_prediction

from data_preprocess import filter_jaspar_tf

if __name__ == "__main__":
    print("********************** start ********************")

    start_time = time.time()  # start the time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start the job")

    # get arguments
    args = load_link_prediction_args(is_evaluation=False)

    print("**********************device********************")
    print(f"Now use device is {args.device}")

    data_path = "/home/liyang/BioWuYan/MethodTest/dygmamba/res/result2/"

    #########################################################################################
    # *************************** load data ******************************************

    feat_path = data_path + "edge_features.npy"
    edge_label_path = data_path + "edge_labels.npy"

    Edge_feature = np.load(feat_path, mmap_mode="r")
    Edge_feature = Edge_feature.reshape(-1,1).copy()

    Edge_label = np.load(edge_label_path, mmap_mode="r")
    Edge_label = Edge_label.reshape(-1,1).copy()

    with open(data_path + "node_feature_data.pkl", "rb") as f:
        load_data = pickle.load(f)
    Node_feature = load_data['node_feature']

    Node_id = pd.read_pickle(data_path + "node_id.pkl")

    graph_df = pd.read_pickle(data_path + "Graph_df.pkl")
    graph_df["Unnamed"] = graph_df.index
    name_list = ["Unnamed", "source_node", "target_node", "time", "label", "edge_idx"]
    New_Graph = graph_df[name_list].copy()
    New_Graph.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx']

    #########################################################################################
    # *************************** TF-region data ******************************************

    print("****************************** Get TF-region data****************************************")

    jaspar_tf_region_file = "/home/liyang/BioWuYan/MethodTest/Data/All2/0process/jaspar_data.h5ad"

    jaspar_data = ad.read_h5ad(jaspar_tf_region_file)

    adata_region_tf = filter_jaspar_tf(jaspar_data)

    coo_matrix = adata_region_tf.X.tocoo()

    # 创建一个 DataFrame 来存储 TF-Peak 的连接
    tf_peak_df = pd.DataFrame({
        'Peak': adata_region_tf.obs_names[coo_matrix.row],
        'TF': adata_region_tf.var_names[coo_matrix.col],
        'value': coo_matrix.data
    })

    tf_peak_df.to_pickle(data_path + "tf_region_grn.pkl")

    #########################################################################################
    # *************************** region-gene data ******************************************

    print("****************************** Get region-gene data****************************************")
    result_graph = New_Graph.copy()
    result_path = data_path + 'my_result_run{run}.npy'

    predict_edge_label = np.load(result_path)

    # binary_output = (predict_edge_label > 0.5).astype(int)

    # result_graph["predict"] = binary_output
    # predict_grn = result_graph[result_graph["predict"]==1].copy()
    
    result_graph["predict"] = predict_edge_label
    predict_grn = result_graph.copy()

    mapping_series = Node_id["name"]
    predict_grn['source'] = (predict_grn['u'] - 1).map(mapping_series)
    predict_grn['target'] = (predict_grn['i'] - 1).map(mapping_series)

    predict_grn.to_pickle(data_path + "region_gene_grn.pkl")

    #########################################################################################
    # *************************** TF-gene data ******************************************

    print("****************************** Get TF-Gene data****************************************")
    peak_gene_df = predict_grn[['source', 'target', 'ts', 'predict']].rename(
        columns={'source': 'Peak', 'target': 'Gene'}
    )

    merged_df = pd.merge(tf_peak_df, peak_gene_df, on='Peak')

    # condition = ~(merged_df['Gene'].str.startswith('chr'))
    # dygmamba_tf_gene_grn = merged_df[condition].copy()
    
    # tf_gene_grn = merged_df.groupby(['TF', 'Gene', 'ts'])['Peak'].nunique()
    # tf_gene_grn = tf_gene_grn.reset_index(name='peak_num')
    
    tf_gene_grn = merged_df.groupby(['TF', 'Gene', 'ts']).agg(
        peak_num=('Peak', 'nunique'),   # 对 Peak 列做去重计数，新列名叫 peak_num
        avg_weight=('predict', 'mean'),  # 对 weight 列做均值，新列名叫 avg_weight
        total_weight=('predict', 'sum')  # (可选) 建议顺便算个总权重
    ).reset_index()

    tf_gene_grn.to_pickle(data_path + "new_tf_gene_grn.pkl")

    #########################################################################################
    # *************************** Average grn ******************************************

    print("****************************** Get AVG TF-Gene data****************************************")

    avg_active_tf_gene_grn = tf_gene_grn.groupby(['TF', 'Gene']).agg(
        avg_weight2=('avg_weight', 'mean'),  # 对 weight 列做均值，新列名叫 avg_weight
        avg_total_weight=('avg_weight', 'sum')  # (可选) 建议顺便算个总权重
    ).reset_index()

    avg_active_tf_gene_grn.to_pickle(data_path + "average_active_tf_gene_grn.pkl")
    
    pivoted_grn = tf_gene_grn.pivot_table(
        index=['TF', 'Gene'],
        columns='ts',
        values='avg_weight',
        fill_value=0
    )

    pivoted_grn['average_active_weight'] = pivoted_grn.mean(axis=1)
    avg_global_tf_gene_grn = pivoted_grn.reset_index()
    avg_global_tf_gene_grn = avg_global_tf_gene_grn[["TF","Gene","average_active_weight"]]
    avg_global_tf_gene_grn.columns.name = None

    avg_global_tf_gene_grn.to_pickle(data_path + "average_global_tf_gene_grn.pkl")
    
    print("********************** end ********************")





