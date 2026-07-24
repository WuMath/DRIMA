import pandas as pd
import anndata as ad
import numpy as np
import pickle
from datetime import datetime
import sys
import os
import re

sys.path.append("/home/liyang/BioWuYan/MethodTest/dygmamba/src")

from data_preprocess import adata_to_dataframe
from sklearn.metrics import average_precision_score, roc_auc_score

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix


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


#################################################################
#################################################################

def each_confusion_matrix(y_true, y_pred):
    
    cm = confusion_matrix(y_true, y_pred)
    model_TN = cm[0,0]
    model_FP = cm[0,1]
    model_FN = cm[1,0]
    model_TP = cm[1,1]
    
    return model_TN, model_FP, model_FN, model_TP


#################################################################
#****************************************************************
#*******  precision analysis   *************
#****************************************************************
#################################################################

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








#################################################################
#****************************************************************
#*******  TF recovery analysis   *************
#****************************************************************
#################################################################

def analysis_tf_recovery(model_merged_grn, model_name = "", fig_path = None):
    
    model_grn_df = model_merged_grn[["TF", "Gene", "predict_label"]]
    model_grn = model_grn_df[model_grn_df["predict_label"] ==1 ].copy()
    gold_grn_df = model_merged_grn[["TF", "Gene", "label"]]
    gold_grn = gold_grn_df[gold_grn_df["label"] ==1].copy()

    x, y, score_auc, total_gold = calculate_tf_recovery(model_grn, gold_grn, 
                                                        fig_title = model_name, fig_path = fig_path)
    
    return model_grn, gold_grn


#################################################################
#****************************************************************
#*******  Compute TF recovery   *************
#****************************************************************
#################################################################


def calculate_tf_recovery(pred_df, gold_df, fig_title = "", fig_path = None, score_col=None):
    """
    计算 TF Recovery Curve 数据
    
    参数:
    pred_df: 预测的 GRN DataFrame, 包含 "TF", "Gene" 列，如果有分数的话就需要 "score_col"
    gold_df: 金标准 GRN DataFrame, 包含 "TF" 列
    score_col: 如果预测 GRN 有分数列（如 importance），填列名；否则填 None，将按靶基因数量排序
    """

    true_tfs = set(gold_df['TF'].unique())
    total_true_tfs = len(true_tfs)
    print(f"Gold Standard 中共有 {total_true_tfs} 个唯一 TF。")

    # 2. 对预测的 TF 进行排序 (Ranking)
    if score_col and score_col in pred_df.columns:
        # 如果有分数，按分数聚合（例如取最大值或总和）后排序
        # 这里假设每一行是一个 TF-Gene link，我们按 TF 的最大分数排序
        ranked_pred_tfs = pred_df.groupby('TF')[score_col].max().sort_values(ascending=False).index.tolist()
    else:
        # 如果没有分数，按 Out-Degree (调控的基因数量) 排序
        print("未检测到分数列，将按 TF 的靶基因数量 (Degree) 进行排序。")
        ranked_pred_tfs = pred_df.groupby('TF')['Gene'].count().sort_values(ascending=False).index.tolist()

    # 3. 计算累积恢复 (Cumulative Recovery)
    cumulative_hits = []
    current_hits = 0
    
    # 遍历排序后的预测 TF 列表
    for tf in ranked_pred_tfs:
        if tf in true_tfs:
            current_hits += 1
        cumulative_hits.append(current_hits)
        
    x_axis = np.arange(1, len(ranked_pred_tfs) + 1)

    y_axis = np.array(cumulative_hits)
    
    # 计算 AUC (Area Under Recovery Curve) - 归一化到 0-1 之间
    # 这是一个简化的 AUC，用于衡量排序的好坏
    # 完美的曲线是：前 N 个预测全是 True TF
    score_auc = auc(x_axis, y_axis) / (len(x_axis) * total_true_tfs)
    
    ###################################################################
    plt.figure(figsize=(8, 6))

    # 绘制恢复曲线
    plt.plot(x_axis, y_axis, label=f'Prediction (AUC={score_auc:.3f})', color='#d62728', linewidth=2)

    # 绘制 "随机猜测" (Random Chance) 参考线
    # 随机情况下，恢复率是线性的
    plt.plot([0, len(x_axis)], [0, (len(x_axis)/len(x_axis))*total_true_tfs * (len(y_axis)/len(pred_df['TF'].unique()))], 
            linestyle='--', color='gray', label='Random')

    # 绘制 "完美预测" (Perfect) 参考线
    # 完美情况下，前 N 个全是 Gold TF
    plt.plot([0, total_true_tfs], [0, total_true_tfs], linestyle=':', color='green', label='Optimal')
    plt.hlines(total_true_tfs, total_true_tfs, len(x_axis), linestyle=':', color='green')

    plt.title(fig_title + " " + 'TF Recovery Curve')
    plt.xlabel('Rank of Predicted TFs (Ranked by Importance)')
    plt.ylabel('Cumulative Number of Recovered TFs')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if fig_path:
        plt.savefig(fig_path + fig_title + "_TF_recovery_curve.png",dpi = 800)
        print("save fig as", fig_path + fig_title + "_TF_recovery_curve.png")
    plt.show()

    return x_axis, y_axis, score_auc, total_true_tfs


#################################################################
#****************************************************************
#*******  Compute F1 score   *************
#****************************************************************
#################################################################

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


def calculate_max_f_beta(precision, recall, thresholds, beta=0.1):
    
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
        
    return max_f_score, optimal_thresh

#################################################################
#****************************************************************
#*******  Read glue result Data   *************
#****************************************************************
#################################################################

def glue_read_full_grn(file_path):

    cols = [
        'TF', 'MotifID', 'AUC', 'NES', 'MotifSimilarityQvalue', 
        'OrthologousIdentity', 'Annotation', 'Context', 'TargetGenes', 'RankAtMax'
    ]
    
    df = pd.read_csv(
        file_path, 
        skiprows=3, 
        names=cols, 
        header=None,
        index_col=False,
        dtype={'TargetGenes': str} 
    )

    gene_pattern = re.compile(r"\(['\"]([^'\"]+)['\"]\s*,\s*([-+]?\d*\.\d+|\d+)\)")


    expanded_records = []
    
    for record in df.to_dict('records'):
        target_str = str(record.get('TargetGenes', ''))

        matches = gene_pattern.findall(target_str)
        
        if not matches:
            # record['Target'] = None
            # record['Importance'] = None
            # expanded_records.append(record)
            continue
            
        for gene, importance in matches:
            new_row = record.copy()

            new_row['Target'] = gene
            new_row['Importance'] = float(importance)
            
            expanded_records.append(new_row)

    df_long = pd.DataFrame(expanded_records)

    
    return df_long



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




if __name__ == "__main__":
    
    start_time = datetime.now()
    print(f"********************* start time: {start_time} *********************")

    assess_result = "/home/liyang/BioWuYan/MethodTest/assess_result2/"
    os.makedirs(assess_result, exist_ok=True)
    
    benchmark_result = []
    
    ##############################################################################################
    # ************************ Benchmark Data **************
    benchmark_data_path = "/home/liyang/BioWuYan/MethodTest/Data/All2/0process/"

    benchmark_tf_gene_grn = ad.read_h5ad(benchmark_data_path + "tf_gene_network.h5ad")

    benchmark_peak_gene_grn = ad.read_h5ad(benchmark_data_path + "peak_gene_network.h5ad")

    benchmark_tf_peak_grn = ad.read_h5ad(benchmark_data_path + "tf_peak_network.h5ad")
        
    benchmark_tf_gene_df = adata_to_dataframe(benchmark_tf_gene_grn)

    benchmark_tf_gene_df = benchmark_tf_gene_df.rename(columns= {"obs":"TF", "var":"Gene"})
    
    benchmark_tf_gene_threshold = 5
    
    ##############################################################################################
    # ************************ dygmamba Data **************
    
    dygmamba_data_path = "/home/liyang/BioWuYan/MethodTest/dygmamba/res/result2/"
    
    dygmamba_tf_peak_df = pd.read_pickle(dygmamba_data_path + "tf_region_grn.pkl")

    dygmamba_peak_gene_df = pd.read_pickle(dygmamba_data_path + "region_gene_grn.pkl")

    dygmamba_tf_gene_df = pd.read_pickle(dygmamba_data_path + "new_tf_gene_grn.pkl")

    dygmamba_avg_tf_gene_df = pd.read_pickle(dygmamba_data_path + "average_tf_gene_grn.pkl")
    
    
    dygmamba_peak_gene_df = dygmamba_peak_gene_df[["source", "target", "ts", "label", "predict"]]
    dygmamba_peak_gene_df.rename(columns={"source": "Peak", "target":"Gene"})

    condition = ~(dygmamba_tf_gene_df['Gene'].str.startswith('chr'))
    dygmamba_tf_gene_grn = dygmamba_tf_gene_df[condition]
    
    print("********************** TF-Gene for each time **********************************")
    print(dygmamba_tf_gene_grn.head())

    condition = ~(dygmamba_avg_tf_gene_df['Gene'].str.startswith('chr'))
    dygmamba_avg_tf_gene_grn = dygmamba_avg_tf_gene_df[condition]

    print("********************** TF-Gene for average **********************************")
    print(dygmamba_avg_tf_gene_grn.head())
    
    ##############################################################################################
    # =================== dygmamba result analysis ===================
    ##############################################################################################
    dygmamba_avg_tf_gene_grn.columns.name = ""
    dygmamba_avg_tf_gene_grn = dygmamba_avg_tf_gene_grn.reset_index()
    dygmamba_avg_tf_gene_grn = dygmamba_avg_tf_gene_grn.drop(["index"],axis = 1)
    dygmamba_avg_tf_gene_grn["predict_label"] = (dygmamba_avg_tf_gene_grn["average_peak_num"]>1).astype(int)

    dyg_merged_data = pd.merge(benchmark_tf_gene_df, dygmamba_avg_tf_gene_grn, on = ["TF", "Gene"], how="outer").fillna(0)
    dyg_merged_data["label"] = (dyg_merged_data["value"] > benchmark_tf_gene_threshold).astype(int)

    dyg_merge_grn = dyg_merged_data.copy()
    
    dyg_y_true = dyg_merge_grn["label"].astype(int)
    dyg_y_pre = dyg_merge_grn["predict_label"].astype(int)
    dyg_model_name = "Dygmamba"
    
    dyg_auc_value, dyg_aupr_value = precision_analysis(dyg_y_true, dyg_y_pre, 
                                               model_name = dyg_model_name, fig_path = assess_result)
    
    dyg_f_score, dyg_best_thresh, dyg_precision, dyg_recall = \
        calculate_f_beta(dyg_y_true, dyg_y_pre, beta=0.1)

    print(f" DyGMamba F1_score is {dyg_f_score}")
    
    dyg_gold_grn, dyg_gold_grn = analysis_tf_recovery(dyg_merged_data, 
                                                      model_name = dyg_model_name, fig_path = assess_result)
            
    benchmark_result.append({"model_name":dyg_model_name, "auc":dyg_auc_value, "aupr": dyg_aupr_value,
                             "precision": dyg_precision, "recall": dyg_recall, "f_score": dyg_f_score})
    ##############################################################################################
    # =================== celloracle result analysis ===================
    ##############################################################################################
    
    oracle_data_path = '/home/liyang/BioWuYan/MethodTest/celloracle/result/'
    celloracle_grn = pd.read_csv(oracle_data_path + "celloracle_results/grn_df_" + "cluster0" + ".csv")

    celloracle_grn.rename(columns={"source": "TF", "target": "Gene"}, inplace= True)

    oracle_tf_gene = celloracle_grn[["TF", "Gene", "-logp"]].copy()
    oracle_merged_df = pd.merge(benchmark_tf_gene_df, oracle_tf_gene, on = ["TF", "Gene"], how="outer").fillna(0)
    
    oracle_merged_df["label"] = (oracle_merged_df["value"] > benchmark_tf_gene_threshold).astype(int)

    oracle_merged_grn = oracle_merged_df.copy()

    oracle_merged_grn["predict_label"] = (oracle_merged_grn["-logp"] > 1.3).astype(int)
    
    print("********************** Celloracle result **********************************")
    print(oracle_merged_grn.head())

    oracle_y_true = oracle_merged_grn["label"].astype(int)
    oracle_y_pre = oracle_merged_grn["predict_label"].astype(int)
    oracle_model_name = "Celloracle"
    
    oracle_auc_value, oracle_aupr_value = precision_analysis(oracle_y_true, oracle_y_pre, 
                                               model_name = oracle_model_name, fig_path = assess_result)
    
    oracle_f_score, oracle_best_thresh, oracle_precision, oracle_recall = \
        calculate_f_beta(oracle_y_true, oracle_y_pre, beta=0.1)

    print(f" Cleeoracle F1_score is {oracle_f_score}")
    
    oracle_gold_grn, oracle_gold_grn = analysis_tf_recovery(oracle_merged_grn, 
                                                      model_name = oracle_model_name, fig_path = assess_result)
    
    benchmark_result.append({"model_name":oracle_model_name, "auc":oracle_auc_value, "aupr": oracle_aupr_value,
                             "precision": oracle_precision, "recall": oracle_recall, "f_score": oracle_f_score})
    
    ##############################################################################################
    # =================== glue result analysis ===================
    ##############################################################################################
    
    glue_data_path = "/home/liyang/BioWuYan/MethodTest/new_scglue/result/"

    glue_df = glue_read_full_grn(glue_data_path + 'pruned_grn.csv')

    glue_df = glue_df[['TF', 'Target', 'Importance', 'MotifID', 'NES', 'AUC', 'RankAtMax', 'MotifSimilarityQvalue', 'Annotation']]
    glue_df.drop_duplicates(subset=['TF', 'Target'], inplace=True)

    glue_grn = glue_df[["TF","Target","Importance"]].rename(columns = {"Target":"Gene"})

    glue_merged_grn = pd.merge(benchmark_tf_gene_df, glue_grn, on = ["TF", "Gene"], how="outer").fillna(0)
    glue_merged_grn["predict_label"] = (glue_merged_grn["Importance"]>0).astype(int)
    glue_merged_grn["label"] = (glue_merged_grn["value"]>benchmark_tf_gene_threshold).astype(int)

    print("********************** GLUE result **********************************")
    print(glue_merged_grn.head())
    
    glue_y_true = glue_merged_grn["label"].astype(int)
    glue_y_pre = glue_merged_grn["predict_label"].astype(int)
    glue_model_name = "GLUE"
    
    glue_auc_value, glue_aupr_value = precision_analysis(glue_y_true, glue_y_pre, 
                                               model_name = glue_model_name, fig_path = assess_result)
    
    glue_f_score, glue_best_thresh, glue_precision, glue_recall = \
        calculate_f_beta(glue_y_true, glue_y_pre, beta=0.1)

    print(f" {glue_model_name} F1_score is {glue_f_score}")
    
    glue_gold_grn, glue_gold_grn = analysis_tf_recovery(glue_merged_grn, 
                                                      model_name = glue_model_name, fig_path = assess_result)
    
    benchmark_result.append({"model_name":glue_model_name, "auc":glue_auc_value, "aupr": glue_aupr_value,
                             "precision": glue_precision, "recall": glue_recall, "f_score": glue_f_score})
    
    ##############################################################################################
    # =================== Pando result analysis ===================
    ##############################################################################################
    
    pando_df = pd.read_csv("/home/liyang/BioWuYan/MethodTest/Pando/result/grn.csv")
    pando_grn = pando_df[["tf","target","pval"]].copy()
    pando_grn.rename(columns={"tf": "TF", "target": "Gene"},inplace=True)
    pando_grn['-log10p'] = -np.log10(pando_grn['pval'])

    pando_merged_grn = pd.merge(benchmark_tf_gene_df, pando_grn, on = ["TF", "Gene"], how="outer").fillna(0)
    pando_merged_grn["label"] = (pando_merged_grn["value"]> benchmark_tf_gene_threshold).astype(int)
    pando_merged_grn["predict_label"] = (pando_merged_grn["-log10p"]> 1.3).astype(int)

    print("********************** Pando result **********************************")
    print(pando_merged_grn.head())
    
    pando_y_true = pando_merged_grn["label"].astype(int)
    pando_y_pre = pando_merged_grn["predict_label"].astype(int)
    pando_model_name = "Pando"
    
    pando_auc_value, pando_aupr_value = precision_analysis(pando_y_true, pando_y_pre, 
                                               model_name = pando_model_name, fig_path = assess_result)
    
    pando_f_score, pando_best_thresh, pando_precision, pando_recall = \
        calculate_f_beta(pando_y_true, pando_y_pre, beta=0.1)

    print(f" {pando_model_name} F1_score is {pando_f_score}")
    
    pando_gold_grn, pando_gold_grn = analysis_tf_recovery(pando_merged_grn, 
                                                      model_name = pando_model_name, fig_path = assess_result)
    
    benchmark_result.append({"model_name":pando_model_name, "auc":pando_auc_value, "aupr": pando_aupr_value,
                             "precision": pando_precision, "recall": pando_recall, "f_score": pando_f_score})
    
    ##############################################################################################
    ##############################################################################################
    ##############################################################################################
    
    dyg_TN, dyg_FP, dyg_FN, dyg_TP = each_confusion_matrix(dyg_y_true, dyg_y_pre)
    print(f"TN: {dyg_TN}, FP: {dyg_FP}, FN: {dyg_FN}, TP: {dyg_TP}")

    glue_TN, glue_FP, glue_FN, glue_TP = each_confusion_matrix(glue_y_true, glue_y_pre)
    print(f"TN: {glue_TN}, FP: {glue_FP}, FN: {glue_FN}, TP: {glue_TP}")

    oracle_TN, oracle_FP, oracle_FN, oracle_TP = each_confusion_matrix(oracle_y_true, oracle_y_pre)
    print(f"TN: {oracle_TN}, FP: {oracle_FP}, FN: {oracle_FN}, TP: {oracle_TP}")

    pando_TN, pando_FP, pando_FN, pando_TP = each_confusion_matrix(pando_y_true, pando_y_pre)
    print(f"TN: {pando_TN}, FP: {pando_FP}, FN: {pando_FN}, TP: {pando_TP}")
    
    benchmark_result_df = pd.DataFrame(benchmark_result)
    benchmark_result_df.to_csv(assess_result + "benchmark_result.csv",
                                index=False, 
                                encoding='utf-8-sig')

    end_time = datetime.now()
    print(f"********************* end time: {end_time} *********************")

    time_diff = end_time - start_time
    print(f"********************* 总运行时间: {time_diff.total_seconds():.2f} 秒 *********************")
