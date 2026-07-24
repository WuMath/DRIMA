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


sys.path.append('/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src/')

from pdata.data_read import read_unibind_file


def calculate_recovery_metrics(ground_truth_ranked, predicted_set, top_n=40):
    """
    计算累积恢复曲线和 AUC
    
    参数:
    ground_truth_ranked: list, 已排序的金标准 TF 列表
    predicted_set: set, 预测的 TF 集合
    top_n: int, 评估前多少名 (论文中是 40)
    """
    
    # 1. 初始化坐标轴
    x_ranks = [] # X轴: Rank position (1, 2, ..., 40)
    y_recovered = [] # Y轴: 累积恢复的 TF 数量
    
    # 2. 遍历每一个 Rank 位置
    cumulative_count = 0
    
    # 只看前 top_n 个
    limit = min(top_n, len(ground_truth_ranked))
    
    for i in range(limit):
        current_tf = ground_truth_ranked[i]
        rank_position = i + 1
        
        # 判断当前这个重要的 TF 是否在我们的预测里
        if current_tf in predicted_set:
            cumulative_count += 1
            
        x_ranks.append(rank_position)
        y_recovered.append(cumulative_count)
        
    # 3. 计算 AUC (Area Under Curve)
    # 使用梯形法则计算面积
    # 注意：论文中的 AUC 可能是归一化的，也可能是原始面积。
    # 这里计算原始面积，最大可能的面积是 (1+40)*40/2 (如果是完美对角线) 或者 40*40 (如果第一名就全找齐了，但这不可能)
    # 为了简单评估，直接用 sklearn 的 auc
    recovery_auc = auc(x_ranks, y_recovered)
    
    # 计算归一化 AUC (Normalized AUC)，即除以完美预测时的面积
    # 完美预测线：y = x (第1名找到1个，第2名找到2个...)
    perfect_y = list(range(1, limit + 1))
    max_auc = auc(x_ranks, perfect_y)
    normalized_auc = recovery_auc / max_auc

    return x_ranks, y_recovered, recovery_auc, normalized_auc







def TF_recovery_analysis(tf_peak_grn, count_region_df, output_path, model_name):

    ######################
    # Optional 1: 计算TF恢复曲线
    
    predicted_tfs = set(tf_peak_grn.var_names)
    ground_truth_ranked = count_region_df.sort_values(by="PeakCount", ascending=False)["TF"].tolist()
    # --- 运行计算 ---
    x, y, raw_auc, norm_auc = calculate_recovery_metrics(ground_truth_ranked, predicted_tfs, top_n=40)

    print(f"Top 40 AUC (Raw): {raw_auc:.2f}")
    print(f"Top 40 AUC (Normalized): {norm_auc:.4f}")

    plt.figure(figsize=(6, 6))

    # 绘制我们的方法的曲线
    plt.plot(x, y, label=f'{model_name} (AUC={norm_auc:.2f})', color='dodgerblue', linewidth=2, marker='o', markersize=4)

    # 绘制“随机猜测”基线 (Random Baseline)
    # 假设总共有 N 个 TF，我们预测了 M 个。随机选一个选中的概率是 M/N。
    total_tfs_pool = len(ground_truth_ranked) # 假设全集
    n_predicted = len(predicted_tfs)
    random_prob = n_predicted / total_tfs_pool
    plt.plot([0, 40], [0, 40 * random_prob], 'k--', label='Random Chance', alpha=0.5)

    # 绘制“完美预测”基线 (Perfect Recovery)
    plt.plot([0, 40], [0, 40], 'g:', label='Perfect Recovery', alpha=0.5)

    plt.xlabel('Rank in UniBind / logFC (Top 40)', fontsize=12)
    plt.ylabel('Cumulative Number of Recovered TFs', fontsize=12)
    plt.title('TF Recovery Curve', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 40)
    plt.ylim(0, 40) # 或者是实际恢复的最大值
    plt.savefig(output_path + model_name + 'TF_Recovery_Curve.png', dpi=300, bbox_inches='tight')

    ######################
    # Optional 2: 计算PR曲线

    col_sums = tf_peak_grn.X.sum(axis=0)
    if scipy.sparse.issparse(tf_peak_grn.X):
        col_sums = col_sums.A1  # 展平为一维数组
    else:
        col_sums = np.asarray(col_sums).flatten()
    tf_peak_grn.var['peak_counts'] = col_sums
    dygmamba_df = tf_peak_grn.var[['peak_counts']].copy()
    dygmamba_df = dygmamba_df.sort_values(by='peak_counts', ascending=False)

    all_tfs = list(predicted_tfs | set(ground_truth_ranked))
    top_k_truth = 100 
    y_true = np.array([1 if tf in ground_truth_ranked[:top_k_truth] else 0 for tf in all_tfs])
    y_scores = np.array([1 if tf in predicted_tfs else 0 for tf in all_tfs])

    # 计算 PR
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color='#ff7f0e', linewidth=2.5, label=f'AP = {ap_score:.2f}')
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path + model_name + 'TF_PR_curve.png', dpi=300, bbox_inches='tight')
    
    return norm_auc


def tf_recovery_main():
    
    data_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/process/"

    output_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/benchmark/"
    
    os.makedirs(output_path, exist_ok=True )
    
    ###########################################################
    # Unibind data processing

    unibind_file = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/unibind/UniBind_search_qlr7ymfp.tar.gz"

    unibind_df, count_region_df = read_unibind_file(unibind_file)

    unibind_df["CellLine"] = unibind_df["CellLine"].str.split('_').str[0]
    count_region_df["CellLine"] = count_region_df["CellLine"].str.split('_').str[0]

    print(unibind_df.head())
    print(count_region_df.head())

    unibind_df.to_pickle(output_path + "unibind_df.pkl")
    count_region_df.to_pickle(output_path + "count_region_df.pkl")
    
    
    ##########################################################################
    # TF recovery
    tf_peak_grn = ad.read_h5ad(data_path + "tf_peak_network.h5ad")
    
    model_name = "DyGMAMBA"
    
    TF_recovery_analysis(tf_peak_grn, count_region_df, output_path, model_name)





if __name__ == "__main__":
    
    tf_recovery_main()