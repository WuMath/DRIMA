import sys
import code
import os
import pickle

import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from tqdm import tqdm
import scipy.sparse
from scipy.sparse import csr_matrix
from gtfparse import read_gtf
from collections import defaultdict
from sklearn.metrics import auc, precision_recall_curve, average_precision_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



sys.path.append('/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src/')


from pdata.data_preprocess import adata_to_dataframe






def each_confusion_matrix(y_true, y_pred):
    
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    model_TN = cm[0,0]
    model_FP = cm[0,1]
    model_FN = cm[1,0]
    model_TP = cm[1,1]
    
    return model_TN, model_FP, model_FN, model_TP




def manual_assess(model_TN, model_FP, model_FN, model_TP, beta = 1 ):
    
    model_precision = float(model_TP/(model_TP+model_FP))
    model_recall = float(model_TP/(model_TP+model_FN))
    model_FPR = float(model_FP/(model_FP+model_TN)) 
    
    model_AUC = (1+model_recall - model_FPR)/2
    if (beta**2 * model_precision + model_recall)==0:
        model_f_beta = 0
    else: 
        model_f_beta = (1+beta**2)*(model_precision*model_recall)/(beta**2 * model_precision + model_recall)
    
    return model_precision, model_recall, model_FPR, model_AUC, model_f_beta 


def auc_plot(y_true, y_predict, fig_title = "", fig_path = None):
    # 计算 ROC 曲线的指标
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_predict)
    roc_auc = auc(fpr, tpr)

    # 绘图
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 绘制随机猜测线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(fig_title +" " + 'Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    if fig_path:
        plt.savefig(fig_path + fig_title + "_roc_curve.png",dpi = 800)
        print("save fig as", fig_path + fig_title + "_roc_curve.png")
    plt.show()




def pr_plot(y_true, y_predict, fig_title = "", fig_path = None):
    # 计算 PR 曲线的指标
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_predict)
    avg_precision = average_precision_score(y_true, y_predict)

    # 绘图
    plt.figure(figsize=(6, 5))
    plt.plot(recall[:-1], precision[:-1], color='green', lw=2, label=f'PR curve (AUPR = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(fig_title +" " + 'Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    if fig_path:
        plt.savefig(fig_path + fig_title + "_pr_curve.png",dpi = 800)
        print("save fig as", fig_path + fig_title + "_pr_curve.png")
    plt.show()



def precision_analysis(y_true, y_pre, model_name = "", fig_path = None):
    
    if sum(y_true) == len(y_true) or sum(y_true) == 0:
        auc_value = 0
        print(" *********************** sample are all true/false************************************** ")
    else:
        auc_value = roc_auc_score(y_true, y_pre)
        auc_plot(y_true, y_pre,fig_title = model_name, fig_path = fig_path)


    aupr_value = average_precision_score(y_true, y_pre)
    pr_plot(y_true, y_pre, fig_title = model_name, fig_path = fig_path)

    print(f"{model_name} result: AUC is {auc_value}, AUPR is {aupr_value}")
    
    return auc_value, aupr_value





def calculate_f_beta(y_true, y_pre, beta=0.1):
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pre)
    
    # F_beta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * (precision * recall)
    denominator = (beta_sq * precision) + recall
    
    # 安全除法 (防止分母为 0 的情况)
    # 如果 precision 和 recall 都是 0，结果设为 0
    f_beta_scores = np.divide(
        numerator, 
        denominator, 
        out=np.zeros_like(numerator), 
        where=denominator != 0
    )
    
    # 找到最大值 ("largest value across all possible cutoffs")
    max_idx = np.argmax(f_beta_scores)
    max_f_score = f_beta_scores[max_idx]
    
    # 注意：thresholds 数组的长度比 precision/recall 少 1 (最后一个点是 P=1, R=0)
    
    if max_idx < len(thresholds):
        optimal_thresh = thresholds[max_idx]
    else:
        optimal_thresh = thresholds[-1] # 边界情况
        
    return max_f_score, optimal_thresh, precision, recall





def region_gene_assess(y_true, y_pre, model_name, beta = 1, type = "binary", fig_path = None):
    
    if type == "binary":
        model_TN, model_FP, model_FN, model_TP = each_confusion_matrix(y_true, y_pre)
        model_precision, model_recall, model_FPR, model_AUC, model_f_beta \
            = manual_assess(model_TN, model_FP, model_FN, model_TP, beta = beta )
            
        return {"model_name": model_name, "TN": model_TN,
                        "FP": model_FP, "FN": model_FN, "TP": model_TP,
                        "precision": model_precision, "recall": model_recall, "FPR": model_FPR,
                        "AUC": model_AUC, "f_score": model_f_beta}
    else:
        
        auc_value, aupr_value = precision_analysis(y_true, y_pre, 
                                               model_name = model_name, fig_path = fig_path)
    
        f_score, best_thresh, precision_value, recall_value = \
            calculate_f_beta(y_true, y_pre, beta=beta)

        print(f" {model_name} F1_score is {f_score}")
        
        return {"model_name":model_name, "auc":auc_value, "aupr": aupr_value,
                "precision": precision_value, "recall": recall_value, "f_score": f_score}





###################################################################################
#
#
###################################################################################

def evaluate_per_gene_correlation(df_pred, df_hic, marker_genes, min_links=5):
    """
    对每个 Marker 基因，分别计算预测分数与 Hi-C 分数的相关性。
    
    参数:
    ----------
    df_pred : pd.DataFrame
        预测数据 [region, gene, value] (value 为预测重要性/Co-accessibility)
    df_hic : pd.DataFrame
        Hi-C 真值 [region, gene, value] (value 为接触频率)
    marker_genes : list
        需要评估的 Marker 基因列表
    min_links : int
        最小连接数阈值。如果某个基因的重叠连接数少于此值，不计算相关性（统计学上无意义）。
        
    返回:
    ----------
    gene_corrs : pd.DataFrame
        包含每个基因的相关性结果 [gene, correlation, p_value, n_links]
    """
    
    # 1. 预处理列名，确保统一
    pred = df_pred.rename(columns={'value': 'score_pred'})
    hic = df_hic.rename(columns={'value': 'score_hic'})
    
    # 确保 key 列是字符串
    for df in [pred, hic]:
        df['region'] = df['region'].astype(str)
        df['gene'] = df['gene'].astype(str)

    # 2. 筛选 Marker Genes (减少数据量，加速合并)
    print(f"筛选 {len(marker_genes)} 个 Marker Genes...")
    pred = pred[pred['gene'].isin(marker_genes)]
    hic = hic[hic['gene'].isin(marker_genes)]
    
    # 3. 数据对齐：取交集 (Inner Join)
    # SCENIC+ 逻辑：只评估那些“既被预测出来，又有 Hi-C 证据”的边，看强度是否匹配
    # 如果您希望评估“未预测到”的惩罚，可以改用 Left Join 并填充 0，但通常 Inner Join 用于评估 Rank 一致性
    merged_df = pd.merge(pred, hic, on=['region', 'gene'], how='inner')
    
    print(f"合并后共有 {len(merged_df)} 条边用于计算。")

    # 4. 逐个基因计算相关性
    results = []
    
    # 按基因分组
    grouped = merged_df.groupby('gene')
    
    for gene, group in grouped:
        n = len(group)
        
        # 只有当该基因有足够多的 Region 连接时，计算相关性才有意义
        if n >= min_links:
            # Spearman 相关性 (非参数，适合 Hi-C 这种幂律分布数据)
            corr, pval = stats.spearmanr(group['score_pred'], group['score_hic'])
            
            # 处理可能的 NaN (例如所有分数都一样导致方差为0)
            if np.isnan(corr):
                corr = 0
                
            results.append({
                'gene': gene,
                'correlation': corr,
                'p_value': pval,
                'n_links': n
            })
    
    # 转换为 DataFrame
    res_df = pd.DataFrame(results)
    
    # 简单统计
    if not res_df.empty:
        mean_corr = res_df['correlation'].mean()
        median_corr = res_df['correlation'].median()
        print(f"评估完成：共 {len(res_df)} 个基因有效。")
        print(f"平均 Spearman 相关性: {mean_corr:.4f}")
        print(f"中位 Spearman 相关性: {median_corr:.4f}")
    else:
        print("警告：没有基因满足计算条件（连接数过少或无重叠）。")
        
    return res_df




###################################################################################
#
#
###################################################################################

def evaluate_scenic_plus_correlation(df_pred, df_hic, marker_genes=None, method='spearman'):
    """
    仿照 SCENIC+ 评估 Region-Gene 预测质量。
    
    参数:
    ----------
    df_pred : pd.DataFrame
        预测结果。必须包含列: ['region', 'gene', 'value'] (value 为重要性得分)
    df_hic : pd.DataFrame
        Hi-C 真值。必须包含列: ['region', 'gene', 'value'] (value 为接触频率)
    marker_genes : list, optional
        Top 100 Marker Genes 列表。如果提供，仅评估这些基因涉及的连接。
    method : str
        'spearman' (推荐，非线性秩相关) 或 'pearson' (线性相关)。
        SCENIC+ 原文使用的是 Spearman。
        
    返回:
    ----------
    correlation : float
        相关系数
    merged_df : pd.DataFrame
        合并后的用于绘图的数据框
    """
    
    # 1. 数据预处理：重命名列以防止冲突
    # 假设输入列名为 region, gene, value
    pred = df_pred.rename(columns={'value': 'pred_score'})
    hic = df_hic.rename(columns={'value': 'hic_score'})
    
    # 确保 key 列是字符串类型，防止因类型不一致导致 merge 失败
    for df in [pred, hic]:
        df['region'] = df['region'].astype(str)
        df['gene'] = df['gene'].astype(str)

    # 2. (关键步骤) 筛选 Marker Genes
    # SCENIC+ 原文："for the top 100 marker genes... correlations were calculated"
    if marker_genes is not None:
        print(f"正在筛选 {len(marker_genes)} 个 Marker Genes...")
        pred = pred[pred['gene'].isin(marker_genes)]
        # Hi-C 数据通常很大，先过滤可以加速 merge
        hic = hic[hic['gene'].isin(marker_genes)]
        
        if pred.empty:
            print("警告：筛选后预测结果为空！请检查 Marker Genes 名字是否与 dataframe 一致。")
            return 0.0, pd.DataFrame()

    # 3. (关键步骤) 数据对齐 - Inner Join (交集)
    # 只有同时存在于预测和 Hi-C 中的边才参与相关性计算
    print("正在合并预测数据与 Hi-C 数据...")
    merged_df = pd.merge(pred, hic, on=['region', 'gene'], how='inner')
    
    n_links = len(merged_df)
    print(f"共找到 {n_links} 个重叠的 Region-Gene 连接用于评估。")
    
    if n_links < 10:
        print("警告：重叠连接数过少，相关性计算可能不可靠。")

    # 4. 计算相关性
    # Spearman 关注的是“排名”：预测分越高的，是不是 Hi-C 分也越高？
    # 这比 Pearson 更适合，因为 Hi-C 数据通常不服从正态分布
    if method == 'spearman':
        corr, p_val = stats.spearmanr(merged_df['pred_score'], merged_df['hic_score'])
    else:
        corr, p_val = stats.pearsonr(merged_df['pred_score'], merged_df['hic_score'])
        
    print(f"评估结果 ({method}): Correlation = {corr:.4f} (P-value = {p_val:.2e})")
    
    return corr, merged_df








def region_gene_evaluate(benchmark_peak_gene, predict_peak_gene_grn, model_name, output_path = None):
    """
    benchmark_peak_gene: pd.DataFrame, columns = ["Peak", "Gene", "value"]
    
    predict_peak_gene_grn: pd.DataFrame, columns = ["Peak", "Gene", "predict"]
    
    """
    
    merged_peak_gene_data = pd.merge(benchmark_peak_gene, predict_peak_gene_grn, 
                                            on = ["Gene", "Peak"], how="inner").fillna(0)

    y_true = merged_peak_gene_data["label"].astype(int)
    y_pre = merged_peak_gene_data["predict"].astype(int)

    result_type = "binary"
    beta_value = 1
    result_dict = region_gene_assess(y_true, y_pre, model_name = model_name, 
                                beta = beta_value, type = result_type, fig_path = output_path)
    
    peak_gene_result = pd.DataFrame([result_dict])

    # peak_gene_result.to_csv(output_path + "dyg_peak_gene_assess_result.csv", index=False)

    print("*"*50)
    
    print(f"merged: {len(merged_peak_gene_data)}, benchmark peak gene: {len(benchmark_peak_gene)},\
        dyg peak gene {len(predict_peak_gene_grn)}")
    
    print(f"Peak-Gene: {merged_peak_gene_data['Peak'].nunique()}, \
        {merged_peak_gene_data['Gene'].nunique()}, edge:{len(merged_peak_gene_data)}")
    
    print(f"Merged Peak-Gene: {merged_peak_gene_data['Peak'].nunique()}, {merged_peak_gene_data['Gene'].nunique()}, \
        {len(merged_peak_gene_data)}")

    print(f"{model_name} Peak-Gene: {predict_peak_gene_grn['Peak'].nunique()}, {predict_peak_gene_grn['Gene'].nunique()}, \
        edge: {len(predict_peak_gene_grn)}")

    print(f"Benchmark Peak-Gene: {benchmark_peak_gene['Peak'].nunique()}, \
        {benchmark_peak_gene['Gene'].nunique()}, edge: {len(benchmark_peak_gene)}")
    
    print("*"*50)
    
    return peak_gene_result













def region_gene_main():
        
    data_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/process/"

    output_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/benchmark/"

    unibind_df_file = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/benchmark/unibind_df.pkl"

    model_result_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/data_dyg/"

    os.makedirs(output_path, exist_ok=True )

    adata_atac = ad.read_h5ad(model_result_path + "atac.h5ad")
    adata_rna = ad.read_h5ad(model_result_path + "rna.h5ad")
    total_peak = set(adata_atac.var_names)

    ##################################################################

    benchmark_peak_gene_grn = ad.read_h5ad(data_path + "peak_gene_network.h5ad")

    benchmark_peak_gene_grn = benchmark_peak_gene_grn[adata_atac.var_names, adata_rna.var_names].copy()

    benchmark_peak_gene_df = adata_to_dataframe(benchmark_peak_gene_grn)
    benchmark_peak_gene_df = benchmark_peak_gene_df.rename(columns= {"obs":"Peak", "var":"Gene"})
    benchmark_peak_gene_df = benchmark_peak_gene_df[benchmark_peak_gene_df["Peak"].isin(total_peak)].copy()
    
    df_hic = benchmark_peak_gene_df.copy()
    df_hic = df_hic.rename(columns={"Peak":"region", "Gene":"gene"})
    
    ############################################################################################
    
    Markrer_Genes = adata_rna.var["highly_variable_rank"].copy()
    Markrer_Genes = Markrer_Genes.sort_values()
    top_100_marker_genes = list(Markrer_Genes.index[:100])
    
    ############################################################################################
    # DYGMAMBA peak-gene grn
        
    Node_id = pd.read_pickle(model_result_path + "node_id.pkl")
    graph_df = pd.read_pickle(model_result_path + "Graph_df.pkl")
    graph_df["Unnamed"] = graph_df.index
    name_list = ["Unnamed", "source_node", "target_node", "time", "label", "edge_idx"]
    New_Graph = graph_df[name_list].copy()
    New_Graph.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx']

    result_path = model_result_path + 'my_result_run0.npy'
    predict_edge_label = np.load(result_path)

    # binary_output = (predict_edge_label > 0.5).astype(int)
    New_Graph["predict"] = predict_edge_label
    dyg_predict_grn = New_Graph.copy()

    mapping_series = Node_id["name"]
    dyg_predict_grn['source'] = (dyg_predict_grn['u'] - 1).map(mapping_series)
    dyg_predict_grn['target'] = (dyg_predict_grn['i'] - 1).map(mapping_series)

    dyg_peak_gene_df = dyg_predict_grn[['source', 'target', 'ts','predict']].rename(
        columns={'source': 'Peak', 'target': 'Gene'}
    )
    dyg_peak_gene_df = dyg_peak_gene_df[~dyg_peak_gene_df["Gene"].str.startswith('chr')].copy()
        
    ####################################################################################
    
    avg_active_peak_gene_grn = dyg_peak_gene_df.groupby(['Peak', 'Gene']).agg(
        avg_ts_weight=('predict', 'mean'),  # 对 weight 列做均值，新列名叫 avg_weight
        avg_total_weight=('predict', 'sum')  # (可选) 建议顺便算个总权重
    ).reset_index()
    avg_active_peak_gene_grn = avg_active_peak_gene_grn[avg_active_peak_gene_grn["Peak"].isin(total_peak)].copy()
    
    
    avg_active_peak_gene_grn["predict"] = (avg_active_peak_gene_grn["avg_ts_weight"]> 0.9).astype(int)

    dyg_peak_gene_result = region_gene_evaluate(benchmark_peak_gene_df, avg_active_peak_gene_grn, model_name="DYGMAMBA")

    df_pred = avg_active_peak_gene_grn.copy()
    df_pred = df_pred.rename(columns={"Peak":"region", "Gene":"gene", "avg_ts_weight":"value"})

    dyg_results = evaluate_per_gene_correlation(df_pred, df_hic, marker_genes=top_100_marker_genes, min_links=3)
    
    dyg_corr_score, dyg_plot_data = evaluate_scenic_plus_correlation(df_pred, df_hic, 
                                                             marker_genes=top_100_marker_genes)
    print("*"*50)
    print(dyg_corr_score)
    print("*"*50)
    print(dyg_results)
    print("*"*50)
    print(dyg_peak_gene_result)
    #######################################################################################
    #
    #
    #######################################################################################






if __name__ == "__main__":
    
    region_gene_main()
    