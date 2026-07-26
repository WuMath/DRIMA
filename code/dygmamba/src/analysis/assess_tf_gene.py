import sys
import code
import os
import pickle
import time
from datetime import datetime

import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

from gtfparse import read_gtf
from collections import defaultdict
from sklearn.metrics import auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


sys.path.append('/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src/')

from pdata.data_preprocess import filter_jaspar_tf

##############################################################################
# 
#
##############################################################################

def evaluate_predictability(expression_matrix, grn_df):
    """
    方法一：评估 GRN 的转录组预测能力
    
    参数:
    expression_matrix: DataFrame (index=Cells, columns=Genes) 表达矩阵
    grn_df: DataFrame (columns=['TF', 'Target']) 预测的调控网络
    
    返回:
    results_df: 包含每个基因预测相关性的 DataFrame
    trained_models: 训练好的模型字典 (用于后续 In silico 分析)
    """
    print(">>> 开始评估预测能力 (Predictability)...")
    
    results = []
    trained_models = {} # Key: Target_Gene, Value: {model, tf_names}
    
    # 获取所有靶基因
    target_genes = grn_df['Target'].unique()
    
    # 遍历每个靶基因
    for target in target_genes:
        if target not in expression_matrix.columns:
            continue
            
        # 1. 找出预测调控该基因的 TF
        regulators = grn_df[grn_df['Target'] == target]['TF'].unique()
        valid_tfs = [tf for tf in regulators if tf in expression_matrix.columns]
        
        if len(valid_tfs) == 0:
            continue
            
        # 2. 准备 X (TF表达) 和 y (靶基因表达)
        X = expression_matrix[valid_tfs]
        y = expression_matrix[target]
        
        # 3. 划分 80% 训练, 20% 测试
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. 训练 GBM 回归模型
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # 5. 预测并计算相关性
        y_pred = model.predict(X_test)
        corr, p_val = pearsonr(y_test, y_pred)
        
        results.append({
            'Target': target,
            'Num_TFs': len(valid_tfs),
            'Correlation': corr,
            'P_value': p_val
        })
        
        # 保存模型供模块三使用
        trained_models[target] = {'model': model, 'tfs': valid_tfs}

    return pd.DataFrame(results), trained_models


##############################################################################
# 
#
##############################################################################
def dyg_tf_gene_data():
    
    data_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/process/"

    output_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/benchmark/"
    
    unibind_df_file = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/benchmark/unibind_df.pkl"
    
    model_result_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/data_dyg/"
    
    os.makedirs(output_path, exist_ok=True )
    
    adata_atac = ad.read_h5ad(model_result_path + "atac.h5ad")
    adata_rna = ad.read_h5ad(model_result_path + "rna.h5ad")
    total_peak = set(adata_atac.var_names)
    
    
    ##################################################################
    print("********************** start ********************")

    start_time = time.time()  # start the time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start the job")



    
    ########################################################################################
    
    jaspar_tf_region_file = data_path + "jaspar_data_processed.h5ad"
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
    
    total_peak = set(adata_atac.var_names)
    dygmamba_tf_peak_df = dygmamba_tf_peak_df[dygmamba_tf_peak_df["Peak"].isin(total_peak)].copy()

    ########################################################################################

    feat_path = model_result_path + "edge_features.npy"
    edge_label_path = model_result_path + "edge_labels.npy"

    Edge_feature = np.load(feat_path, mmap_mode="r")
    Edge_feature = Edge_feature.reshape(-1,1).copy()
    Edge_label = np.load(edge_label_path, mmap_mode="r")
    Edge_label = Edge_label.reshape(-1,1).copy()

    with open(model_result_path + "node_feature_data.pkl", "rb") as f:
        load_data = pickle.load(f)

    Node_feature = load_data['node_feature']

    Node_id = pd.read_pickle(model_result_path + "node_id.pkl")

    graph_df = pd.read_pickle(model_result_path + "Graph_df.pkl")
    graph_df["Unnamed"] = graph_df.index
    name_list = ["Unnamed", "source_node", "target_node", "time", "label", "edge_idx"]
    New_Graph = graph_df[name_list].copy()
    New_Graph.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx']
    
    
    ####################################################################################
    
    ####################################################################################
    run = 0
    result_path = model_result_path + f'my_result_run{run}.npy'
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

    tf_gene_grn.to_pickle(model_result_path + "new_tf_gene_grn_1224.pkl")

    avg_active_tf_gene_grn = tf_gene_grn.groupby(['TF', 'Gene']).agg(
        avg_ts_weight=('total_weight', 'mean'),  # 对 weight 列做均值，新列名叫 avg_weight
        avg_total_weight=('total_weight', 'sum')  # (可选) 建议顺便算个总权重
    ).reset_index()
    avg_active_tf_gene_grn.to_pickle(model_result_path + "average_active_tf_gene_grn.pkl")

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
    avg_global_tf_gene_grn.to_pickle(model_result_path + "average_global_tf_gene_grn.pkl")

    print("*"*50)
    print(f"TF-Gene: {tf_gene_grn['TF'].nunique()}, {tf_gene_grn['Gene'].nunique()}, edge: {len(tf_gene_grn)}")
    print(f"avg active TF-Gene: {avg_active_tf_gene_grn['TF'].nunique()}, {avg_active_tf_gene_grn['Gene'].nunique()},\
        edge: {len(avg_active_tf_gene_grn)}")
    print(f"avg global TF-Gene: {avg_global_tf_gene_grn['TF'].nunique()}, {avg_global_tf_gene_grn['Gene'].nunique()},\
        edge: {len(avg_global_tf_gene_grn)}")
    print("*"*50)


##############################################################################
# 
#
##############################################################################

def tf_gene_assess():

    output_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/benchmark/"

    dyg_result_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/data_dyg/"
    
    os.makedirs(output_path, exist_ok=True )
    
    adata_atac = ad.read_h5ad(dyg_result_path + "atac.h5ad")
    adata_rna = ad.read_h5ad(dyg_result_path + "rna.h5ad")

    avg_active_tf_gene_grn = pd.read_pickle(dyg_result_path + "average_active_tf_gene_grn.pkl")
    avg_global_tf_gene_grn = pd.read_pickle(dyg_result_path + "average_global_tf_gene_grn.pkl")
    
    active_grn = avg_active_tf_gene_grn[["TF","Gene"]].copy()
    active_grn.rename(columns={'Gene':'Target'},inplace = True)
    matrix_df = pd.DataFrame(adata_rna.X.toarray(), index=adata_rna.obs_names,
                             columns=adata_rna.var_names)
    
    active_result, trained_models = evaluate_predictability(matrix_df, active_grn)
    
    active_result.to_csv(output_path + "dyg_tf_gene_active_assess_result.csv", index=False)
    
    global_grn = avg_global_tf_gene_grn[["TF","Gene"]].copy()
    global_grn.rename(columns={'Gene':'Target'},inplace = True)
    
    global_result, global_trained_models = evaluate_predictability(matrix_df, global_grn)
    
    global_result.to_csv(output_path + "dyg_tf_gene_global_assess_result.csv", index=False)
    
    print("*"*50)
    print(trained_models)
    
    print("*"*50)
    print(global_trained_models)
    
    






if __name__ == "__main__":
    
    tf_gene_assess()