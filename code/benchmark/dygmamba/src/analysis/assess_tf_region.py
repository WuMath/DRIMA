import sys
import code
import os
import pickle

import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import scipy.sparse
from scipy.sparse import csr_matrix
from gtfparse import read_gtf
from collections import defaultdict
from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import pybedtools

sys.path.append('/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src/')

from pdata.data_preprocess import filter_jaspar_tf, adata_to_dataframe
from pdata.data_preprocess import build_tf_peak_network





def calculate_tf_metrics(pred_df, unibind_df, f_beta = 1):
    """
    计算 TF-Region 预测的 Precision, Recall, F1
    
    参数:
    pred_df: 包含 SCENIC+ 预测结果的 DataFrame
             必须列: 'TF', 'peak' (格式如 chr1-100-200)
             
    unibind_df: 包含 UniBind 金标准的 DataFrame
                必须列: 'TF', 'chrom', 'start', 'end'
    """
    
    # ---------------------------------------------------------
    # 1. 预处理预测数据 (解析 Peak 坐标)
    # ---------------------------------------------------------
    print("正在解析 Consensus Peaks 坐标...")
    
    # 提取所有唯一的 Consensus Peaks (作为全集/背景)
    all_consensus_peaks = pred_df['Peak'].unique()
    consensus_map_df = pd.DataFrame({'peak_id': all_consensus_peaks})
    
    # 解析 "chr1-start-end" 格式
    coords = consensus_map_df['peak_id'].str.extract(r'(?P<chrom>.+)-(?P<start>\d+)-(?P<end>\d+)')
    consensus_map_df = pd.concat([consensus_map_df, coords], axis=1)
    
    # 转换坐标类型
    consensus_map_df['start'] = consensus_map_df['start'].astype(int)
    consensus_map_df['end'] = consensus_map_df['end'].astype(int)
    
    consensus_bed_df = consensus_map_df[['chrom', 'start', 'end', 'peak_id']]
    consensus_bed_all = pybedtools.BedTool.from_dataframe(consensus_bed_df)
    
    # ---------------------------------------------------------
    # 2. 逐个 TF 计算指标
    # ---------------------------------------------------------
    # 找出两个数据集中共有的 TF
    common_tfs = set(pred_df['TF'].unique()) & set(unibind_df['TF'].unique())
    print(f"共发现 {len(common_tfs)} 个共有 TF，开始评估...")
    
    metrics_list = []
    
    for tf in common_tfs:
        # --- A. 准备真值 (Ground Truth) ---
        # 获取该 TF 在 UniBind 中的所有物理结合区域
        tf_unibind_subset = unibind_df[unibind_df['TF'] == tf][['chrom', 'start', 'end']]
        
        if tf_unibind_subset.empty:
            continue
            
        tf_unibind_bed = pybedtools.BedTool.from_dataframe(tf_unibind_subset)
        
        # --- B. 确定真值集合 (Ground Truth Set of Consensus Peaks) ---
        # 核心逻辑：如果一个 Consensus Peak 与 UniBind 区域有重叠，它就是该 TF 的"真"靶点
        # -u: 只要有重叠就输出
        # -wa: 输出原始的 A (即 Consensus Peak)
        true_targets_bed = consensus_bed_all.intersect(tf_unibind_bed, u=True, wa=True)
        
        # 获取真值 Peak ID 集合
        true_peak_ids = set([f.name for f in true_targets_bed])
        
        # --- C. 获取预测集合 (Predicted Set) ---
        # SCENIC+ 预测该 TF 结合的 Consensus Peaks
        pred_peak_ids = set(pred_df[pred_df['TF'] == tf]['Peak'])
        
        # --- D. 计算指标 ---
        # TP: 预测了，且是真的
        tp = len(pred_peak_ids.intersection(true_peak_ids))
        
        # FP: 预测了，但不是真的
        fp = len(pred_peak_ids) - tp
        
        # FN: 没预测，但是真的 (在 true_peak_ids 里，但不在 pred_peak_ids 里)
        fn = len(true_peak_ids) - tp
        
        # 防止除以 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (1+f_beta**2) * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_list.append({
            'TF': tf,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'GroundTruth_Count': len(true_peak_ids),
            'Predicted_Count': len(pred_peak_ids)
        })
        
    # ---------------------------------------------------------
    # 3. 汇总结果
    # ---------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_list)
    
    # 清理临时文件 (pybedtools 习惯)
    pybedtools.cleanup()
    
    return metrics_df



def tf_region_main():
    
    data_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/process/"

    output_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/benchmark/"

    unibind_df_file = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/benchmark/unibind_df.pkl"

    os.makedirs(output_path, exist_ok=True )

    adata_atac = ad.read_h5ad(data_path + "atac_processed.h5ad")

    ##################################################################
    # Benchmark TF-Region
    # 读取整个文件
    tf_chip_seq = pd.read_parquet(data_path + "combined_chip_seq.parquet")
    cell_type_chip_seq = tf_chip_seq[["chrom", "start", "end", "tf_name"]].copy()
    cell_type_chip_seq.rename(columns={'tf_name':'tf'},inplace = True)
    cell_type_chip_seq = cell_type_chip_seq[['tf','chrom','start','end']].copy()

    unibind_df = pd.read_pickle(unibind_df_file)
    unibind_chip_df = unibind_df[unibind_df['TF'].isin(set(cell_type_chip_seq["tf"]))].copy()

    unibind_chip_df = unibind_chip_df[["chrom", "start", "end", "TF"]].copy()
    unibind_chip_df.rename(columns={'TF':'tf'},inplace = True)
    unibind_chip_df = unibind_chip_df[['tf','chrom','start','end']].copy()

    unibind_tf_peak = build_tf_peak_network(adata_atac, unibind_chip_df)
    unibind_tf_peak_grn = adata_to_dataframe(unibind_tf_peak)
    unibind_tf_peak_grn.rename(columns={'obs':'peak_id', "var":"tf"},inplace = True)
    unibind_tf_peak_grn[['chrom', 'start', 'end']] = unibind_tf_peak_grn['peak_id'].str.split('-', expand=True)
    unibind_tf_peak_grn['start'] = unibind_tf_peak_grn['start'].astype(int)
    unibind_tf_peak_grn['end'] = unibind_tf_peak_grn['end'].astype(int)
    unibind_tf_peak_grn.rename(columns={'tf':'TF'},inplace = True)
    print(unibind_tf_peak_grn)
    ####################################################################
    # 
    jaspar_tf_region_file = data_path + "jaspar_data_processed.h5ad"
    jaspar_data = ad.read_h5ad(jaspar_tf_region_file)
    adata_region_tf = filter_jaspar_tf(jaspar_data)

    coo_matrix = adata_region_tf.X.tocoo()
    dyg_tf_peak_df = pd.DataFrame({
        'Peak': adata_region_tf.obs_names[coo_matrix.row],
        'TF': adata_region_tf.var_names[coo_matrix.col],
        'value': coo_matrix.data
    })
    dyg_tf_peak_df = dyg_tf_peak_df[dyg_tf_peak_df["Peak"].isin(set(adata_atac.var_names))]

    ###################################################################################
    # 
    unibind_tf_peak_grn.rename(columns={'tf':'TF'},inplace = True)

    dyg_results = calculate_tf_metrics(dyg_tf_peak_df, unibind_tf_peak_grn)



if __name__ == "__main__":
    
    tf_region_main()



