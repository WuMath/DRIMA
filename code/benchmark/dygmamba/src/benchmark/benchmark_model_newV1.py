import sys
import code
import os
import pickle
import tarfile
import io

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
import networkx as nx

sys.path.append('/home/wuyan/dygmamba_project/model/dygmamba/src/')

from pdata.data_preprocess import filter_jaspar_tf, adata_to_dataframe
from pdata.data_preprocess import build_tf_peak_network
import pybedtools
import dill


def _annotate_bars(ax, fmt='.4f', fontsize = 16):
    """在柱状图每个柱子顶端标注数值。"""
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), fmt),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 9), textcoords='offset points', fontsize=fontsize
        )

########################################################################################################
#
# TF recovery analysis
#
########################################################################################################

from pdata.benchmark_data import glue_read_ctx_grn


def calculate_recovery_metrics(ground_truth_ranked, predicted_set, top_n=40):
    """
    计算累积恢复曲线和 AUC
    
    参数:
    ground_truth_ranked: list, 已排序的金标准 TF 列表
    predicted_set: set, 预测的 TF 集合
    top_n: int, 评估前多少名 (论文中是 40)
    """
    
    # 1. 初始化坐标轴
    x_ranks = [] 
    y_recovered = []
    
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


#################################################################
#****************************************************************
#*******  Read Unibind Data   *************
#****************************************************************
#################################################################

def read_unibind_file(tar_file_path):


    bed_columns = ['chrom', 'start', 'end', 'name', 'score', 
                   'strand', 'signal', 'p_val', 'q_val', 'peak_center']

    all_tf_regions = []
    tf_stats_list = []    


    try:
        with tarfile.open(tar_file_path, "r") as tar:
            
            # 获取包内所有成员列表
            for member in tar.getmembers():

                # 1. 只处理 BED 文件
                if member.isfile() and member.name.endswith('.bed'):

                    # 2. 从文件名中提取 TF 名称
                    # UniBind 文件名通常格式: DATASET_ID.CELL_LINE.TF_NAME.MOTIF_ID.bed
                    # 例如: ENCSR000AKB.GM12878.CTCF.MA0139.1.bed
                    filename = os.path.basename(member.name)
                    parts = filename.split('.')

                    # 假设 TF 名称在第 3 个位置 (索引 2)，根据实际情况调整
                    # 这里的逻辑是寻找全大写的单词，或者依赖 UniBind 的命名规范
                    # 如果你有 metadata TSV 文件，最好配合那个用，这里演示纯文件名提取
                    if len(parts) >= 3:
                        tf_name = parts[2]
                    else:
                        tf_name = "Unknown"

                    # 3. 读取 BED 文件内容
                    f = tar.extractfile(member)
                    if f:
                        # 使用 io.TextIOWrapper 将字节流转为文本流
                        content = io.TextIOWrapper(f, encoding='utf-8')

                        # 读取数据
                        df = pd.read_csv(content, sep='\t', header=None, names=bed_columns)

                        current_count = len(df)
                        # 提取元数据 (增加容错处理)
                        dataset_id = parts[0] if len(parts) > 0 else "Unknown"
                        cell_line = parts[1] if len(parts) > 1 else "Unknown"
                        tf_name = parts[2] if len(parts) > 2 else "Unknown"
                        tf_stats_list.append({
                            'TF': tf_name,
                            'CellLine': cell_line,
                            'DatasetID': dataset_id,
                            'FileName': filename,
                            'PeakCount': current_count
                        })
                        
                        df = df[['chrom', 'start', 'end']]
                        df['TF'] = tf_name
                        df['CellLine'] = cell_line
                        df['SourceID'] = parts[0]

                        all_tf_regions.append(df)

        # 4. 合并所有数据
        if all_tf_regions:
            unibind_df = pd.concat(all_tf_regions, ignore_index=True)
            
            # 可选：保存为 CSV 备用
            # unibind_df.to_csv("GM12878_UniBind_Regions.csv", index=False)
        else:
            print("警告：未在压缩包中找到 .bed 文件。")
            
        if tf_stats_list:
            summary_df = pd.DataFrame(tf_stats_list)
            
            # 逻辑：对于同一个 CellLine 下的同一个 TF，可能来自多个 Dataset
            # 我们取 Peak 数最多的那个作为该 TF 在该细胞系下的代表 (Max)
            # 或者你也可以选择 sum() 计算总和
            
            ranked_df = summary_df.groupby(['CellLine', 'TF'])['PeakCount'].max().reset_index()
            
            # 按 PeakCount 降序排列
            ranked_df = ranked_df.sort_values(['CellLine', 'PeakCount'], ascending=[True, False])
            
            # 添加组内排名
            ranked_df['Rank'] = ranked_df.groupby('CellLine')['PeakCount'].rank(method='first', ascending=False)
            
            # 保存统计表
            # ranked_df.to_csv("UniBind_TF_Rank_Summary.csv", index=False)
            # print(f"统计排名已保存至: UniBind_TF_Rank_Summary.csv")

    except FileNotFoundError:
        print("错误：找不到指定的 tar 文件，请检查路径。")
        
    return unibind_df, ranked_df



def analysis_tf_recovery(data_path, output_path, method_list, jaspar_threshold, method_colors):
    ###########################################################
    # Unibind data processing
    
    benchmark_df = pd.read_pickle(output_path + "count_region_df.pkl")
    ground_truth_ranked = benchmark_df.sort_values(by="PeakCount", ascending=False)["TF"].tolist()
    
    plt.figure(figsize=(6, 7))

    adata_atac = ad.read_h5ad(data_path + "process/atac_processed.h5ad")
    adata_rna = ad.read_h5ad(data_path + "process/rna_processed.h5ad")
    
    # ============================================================
    # 新增：提取 JASPAR 有效 TF 集合（所有方法取交集用）
    # ============================================================
    jaspar_data = ad.read_h5ad(data_path + "process/jaspar_data_processed.h5ad")
    jaspar_filtered = filter_jaspar_tf(jaspar_data, jaspar_threshold)
    jaspar_tfs = set(jaspar_filtered.var_names)  

    
    max_num = 0    
    top_num= 50
    ##########################################################################
    # DygMamba TF recovery
    if "DyGMamba" in method_list:
        
        jaspar_tf_region_file = data_path + "process/jaspar_data_processed.h5ad"
        jaspar_data = ad.read_h5ad(jaspar_tf_region_file)
        adata_region_tf = filter_jaspar_tf(jaspar_data, jaspar_threshold)

        coo_matrix = adata_region_tf.X.tocoo()
        tf_peak_df = pd.DataFrame({
            'Peak': adata_region_tf.obs_names[coo_matrix.row],
            'TF': adata_region_tf.var_names[coo_matrix.col],
            'value': coo_matrix.data
        })
        tf_peak_df = tf_peak_df[tf_peak_df["TF"].isin(set(adata_rna.var_names))]
        tf_peak_df = tf_peak_df[tf_peak_df["Peak"].isin(set(adata_atac.var_names))]
        
        dyg_tfs = set(tf_peak_df["TF"])

        dyg_x, dyg_y, dyg_raw_auc, dyg_norm_auc = \
            calculate_recovery_metrics(ground_truth_ranked, dyg_tfs, top_n=top_num)

        # 绘制我们的方法的曲线
        
        if max(dyg_y) > max_num:
            max_num = max(dyg_y)
            
        plt.plot(dyg_x, dyg_y, label=f'{"DyGMamba"} \n(AUC={dyg_norm_auc:.2f})', 
                 color=method_colors['DyGMamba'], linewidth=2, marker='o', markersize=4)

    ##########################################################################
    # GLUE TF recovery
    if "GLUE" in method_list:

        glue_grn = glue_read_ctx_grn(data_path + "data_glue/pruned_grn.csv")


        df_edges = nx.to_pandas_edgelist(glue_grn, source="TF", target="Gene")

        glue_tfs  = set(df_edges["TF"]) & jaspar_tfs          # ← 取交集

        glue_x, glue_y, glue_raw_auc, glue_norm_auc = \
            calculate_recovery_metrics(ground_truth_ranked, glue_tfs, top_n=top_num)

        if max(glue_y) > max_num:
            max_num = max(glue_y)
            
        plt.plot(glue_x, glue_y, label=f'{"GLUE"} \n(AUC={glue_norm_auc:.2f})', 
                 color=method_colors['GLUE'], linewidth=2, marker='o', markersize=4)

    ##########################################################################
    # CellOracle TF recovery
    if "CellOracle" in method_list:

        celloracle_grn = pd.read_csv(data_path + "data_celloracle/celloracle_results/grn_df_" + "cluster0" + ".csv")

        celloracle_grn.rename(columns={"source": "TF", "target": "Gene"}, inplace= True)

        oracle_tfs = set(celloracle_grn["TF"]) & jaspar_tfs   # ← 取交集

        oracle_x, oracle_y, oracle_raw_auc, oracle_norm_auc = \
            calculate_recovery_metrics(ground_truth_ranked, oracle_tfs, top_n=top_num)
            
        if max(oracle_y) > max_num:
            max_num = max(oracle_y)

        plt.plot(oracle_x, oracle_y, label=f'{"CellOracle"} \n(AUC={oracle_norm_auc:.2f})', 
                 color=method_colors['CellOracle'], linewidth=2, marker='o', markersize=4)


    ##########################################################################
    # FigR TF recovery
    if "FigR" in method_list:

        figr_grn = pd.read_csv(data_path + "data_FigR/TF_Gene_Network.csv")

        figr_tfs  = set(figr_grn["TF"]) & jaspar_tfs          # ← 取交集

        figr_x, figr_y, figr_raw_auc, figr_norm_auc = \
            calculate_recovery_metrics(ground_truth_ranked, figr_tfs, top_n=top_num)
            
        if max(figr_y) > max_num:
            max_num = max(figr_y)

        plt.plot(figr_x, figr_y, label=f'{"FigR"} \n(AUC={figr_norm_auc:.2f})', 
                color=method_colors['FigR'], linewidth=2, marker='o', markersize=4)

    ##########################################################################
    # Linger TF recovery
    if "LINGER" in method_list:

        linger_grn = pd.read_csv(data_path + "data_linger/train/cell_population_TF_RE_binding.txt", sep='\t')

        linger_tfs  = set(linger_grn.columns[1:]) & jaspar_tfs  # ← 取交集

        linger_x, linger_y, linger_raw_auc, linger_norm_auc = \
            calculate_recovery_metrics(ground_truth_ranked, linger_tfs, top_n=top_num)

        if max(linger_y) > max_num:
            max_num = max(linger_y)
            
        plt.plot(linger_x, linger_y, label=f'{"LINGER"} \n(AUC={linger_norm_auc:.2f})', 
                color=method_colors['LINGER'], linewidth=2, marker='o', markersize=4)

    ##########################################################################
    # GRaNIE TF recovery
    if "GRaNIE" in method_list:

        granie_grn = pd.read_csv(data_path + "GRaNIE_output/GRaNIE_tf_gene_Links.csv")

        granie_tfs  = set(granie_grn["TF.name"]) & jaspar_tfs  # ← 取交集

        granie_x, granie_y, granie_raw_auc, granie_norm_auc = \
            calculate_recovery_metrics(ground_truth_ranked, granie_tfs, top_n=top_num)
            
        if max(granie_y) > max_num:
            max_num = max(granie_y)

        plt.plot(granie_x, granie_y, label=f'{"GRaNIE"} \n(AUC={granie_norm_auc:.2f})', 
                color=method_colors['GRaNIE'], linewidth=2, marker='o', markersize=4)


    ##########################################################################
    # Pando TF recovery
    if "Pando" in method_list:

        pando_grn = pd.read_csv(data_path + "data_pando/tf_gene_network.csv")

        pando_tfs  = set(pando_grn["TF"]) & jaspar_tfs         # ← 取交集

        pando_x, pando_y, pando_raw_auc, pando_norm_auc = \
            calculate_recovery_metrics(ground_truth_ranked, pando_tfs, top_n=top_num)
            
        if max(pando_y) > max_num:
            max_num = max(pando_y)

        plt.plot(pando_x, pando_y, label=f'{"FigR"} \n(AUC={pando_norm_auc:.2f})', 
                color=method_colors['Pando'], linewidth=2, marker='o', markersize=4)


    ############################################################################
    FONT_SIZE = 20
    plt.title('TF Recovery Curve', fontsize=FONT_SIZE-2)
    plt.legend(loc = "upper left", bbox_to_anchor=(1, 1), ncol = 1, fontsize = FONT_SIZE-2)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    max_tf = min(40, len(ground_truth_ranked))
    max_tf2 = min(max_tf, max_num)  # 确保 x 轴范围至少覆盖所有方法的 TF 数量
    
    plt.xlim(0, max_tf)
    plt.ylim(0, int(max_tf2*1.1)) # 或者是实际恢复的最大值
    plt.ylabel("Number of Top TFs Recovered", fontsize=FONT_SIZE)
    plt.xlabel("TF Rank", fontsize = FONT_SIZE)
    plt.tick_params(labelsize=FONT_SIZE-2)

    plt.savefig(output_path + 'Benchmark of TF Recovery Curve.png', dpi=1200, bbox_inches='tight')
    
    

########################################################################################################
#
# TF-Region analysis
#
########################################################################################################

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
            'fscore': f1,
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


def analysis_tf_region(unibind_df_file, data_path, f_score, method_list, jaspar_threshold):
    
    
    adata_atac = ad.read_h5ad(data_path + "process/atac_processed.h5ad")
    adata_rna = ad.read_h5ad(data_path + "process/rna_processed.h5ad")
    ##################################################################
    # Benchmark TF-Region
    # 读取整个文件
    tf_chip_seq = pd.read_parquet(data_path + "process/combined_chip_seq.parquet")
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


    method_result = {}
    
    # ============================================================
    # 新增：构建 JASPAR 过滤集合
    # 每个方法的预测在评估前先与此集合取交集
    # ============================================================
    
    jaspar_raw = ad.read_h5ad(data_path + "process/jaspar_data_processed.h5ad")
    jaspar_filtered = filter_jaspar_tf(jaspar_raw, score_threshold=jaspar_threshold)

    coo = jaspar_filtered.X.tocoo()
    jaspar_filter_df = pd.DataFrame({
        'Peak': jaspar_filtered.obs_names[coo.row],
        'TF': jaspar_filtered.var_names[coo.col],
        'value': coo.data
    })
    jaspar_filter_df = jaspar_filter_df[jaspar_filter_df["TF"].isin(set(adata_rna.var_names))]
    jaspar_filter_df = jaspar_filter_df[jaspar_filter_df["Peak"].isin(set(adata_atac.var_names))]

    jaspar_key_set = set(zip(jaspar_filter_df["Peak"], jaspar_filter_df["TF"]))
    
    def apply_jaspar_filter(pred_df, peak_col='Peak', tf_col='TF'):
        """
        将预测结果与 JASPAR 取交集：
        只保留 (Peak, TF) 同时存在于 JASPAR motif 中的边
        """
        before = len(pred_df)
        mask = pred_df.apply(
            lambda row: (row[peak_col], row[tf_col]) in jaspar_key_set, axis=1
        )
        filtered = pred_df[mask].copy()
        after = len(filtered)

        return filtered
    
    ####################################################################
    # dyg TF-Region
    if "DyGMamba" in method_list:
        jaspar_tf_region_file = data_path + "process/jaspar_data_processed.h5ad"
        jaspar_data = ad.read_h5ad(jaspar_tf_region_file)
        adata_region_tf = filter_jaspar_tf(jaspar_data, score_threshold= jaspar_threshold)

        coo_matrix = adata_region_tf.X.tocoo()
        tf_peak_df = pd.DataFrame({
            'Peak': adata_region_tf.obs_names[coo_matrix.row],
            'TF': adata_region_tf.var_names[coo_matrix.col],
            'value': coo_matrix.data
        })
        tf_peak_df = tf_peak_df[tf_peak_df["TF"].isin(set(adata_rna.var_names))]
        tf_peak_df = tf_peak_df[tf_peak_df["Peak"].isin(set(adata_atac.var_names))]

        dyg_results = calculate_tf_metrics(tf_peak_df, unibind_tf_peak_grn, f_beta = f_score)
        
        dyg_results['Method'] = 'DyGMamba'
        
        method_result.update({"DyGMamba": dyg_results})

    # dyg_results.to_csv(output_path + "dyg_tf_region_jaspar_unibind_results.csv", index=False)

    ####################################################################
    # GLUE TF-Region
    if "GLUE" in method_list:

        glue_data_path = data_path + "data_glue/"
        
        with open(glue_data_path + "peak2tf.pkl", "rb") as f:
            tf_peak_nx = dill.load(f)
        
        glue_tf_peak_df = nx.to_pandas_edgelist(tf_peak_nx) 
        glue_tf_peak_df["value"] = 1
        glue_tf_peak_df.rename(columns={'source':'Peak', 'target':'TF'}, inplace=True)
        
        glue_tf_peak_df = apply_jaspar_filter(glue_tf_peak_df)   # ← 新增
        
        glue_results = calculate_tf_metrics(glue_tf_peak_df, unibind_tf_peak_grn, f_beta = f_score)
        
        glue_results['Method'] = 'GLUE'
        
        method_result.update({"GLUE": glue_results})

    ##########################################################################
    # LINGER TF-Region predictability
    if "LINGER" in method_list:
        linger_grn = pd.read_csv(data_path + "data_linger/train/cell_population_TF_RE_binding.txt", sep='\t')
        linger_grn = linger_grn.rename(columns={'Unnamed: 0': 'Peak'})
        linger_df= linger_grn.melt(id_vars=['Peak'], var_name='TF', value_name='regulation')
        
        if linger_df['Peak'].str.contains(':').any():
            linger_df['Peak'] = linger_df['Peak'].str.replace(':', '-', regex=False)
            
        else:
            print("LINGER 格式已是连字符（chr1-100-200），无需转换。")

        
        linger_df = apply_jaspar_filter(linger_df)                # ← 新增
        
        linger_results = calculate_tf_metrics(linger_df, unibind_tf_peak_grn, f_beta = f_score)
        
        linger_results["Method"] = "LINGER"
        
        method_result.update({"LINGER": linger_results})


    ##########################################################################
    # GRaNIE TF-Region predictability
    if "GRaNIE" in method_list:

        granie_grn = pd.read_csv(data_path + "GRaNIE_output/GRaNIE_tf_region_Links.csv")

        granie_df = granie_grn[["TF.name", "peak.ID", "TF_peak.r", "TF_peak.fdr"]].copy()
        granie_df.rename(columns={"TF.name": "TF", "peak.ID": "Peak"}, inplace=True)

        if granie_df['Peak'].str.contains(':').any():

            granie_df['Peak'] = granie_df['Peak'].str.replace(':', '-', regex=False)
            
        else:
            print("GRaNIE 格式已是连字符（chr1-100-200），无需转换。")
        
        granie_df = apply_jaspar_filter(granie_df)                # ← 新增
        
        granie_results = calculate_tf_metrics(granie_df, unibind_tf_peak_grn, f_beta = f_score)
        
        granie_results['Method'] = 'GRaNIE'
        
        method_result.update({"GRaNIE": granie_results})


    ##########################################################################
    # Pando TF-Region predictability

    if "Pando" in method_list:
        pando_grn = pd.read_csv(data_path + "data_pando/tf_region_network.csv")

        pando_df = pando_grn.copy()
        pando_df.rename(columns={"Region": "Peak"}, inplace=True)

        if pando_df['Peak'].str.contains(':').any():
            pando_df['Peak'] = pando_df['Peak'].str.replace(':', '-', regex=False)
            
        else:
            print("Pando 格式已是连字符（chr1-100-200），无需转换。")
        
        pando_df = apply_jaspar_filter(pando_df)                  # ← 新增
        
        pando_results = calculate_tf_metrics(pando_df, unibind_tf_peak_grn, f_beta = f_score)
        
        pando_results['Method'] = 'Pando'
        
        method_result.update({"Pando": pando_results})
    
    return method_result
        






def plot_serval_metric_comparsion(result_df, output_path, method_colors, beta = 1):
    
    FONT_SIZE = 20
    
    label_f = f"$F_{{{beta}}}$"
    
        # ── 绘图 3：Precision vs F_score 分组柱状图 ──────────────────────────────
    plot_df = result_df.melt(
        id_vars="Method",
        value_vars=["Precision", "F_score"],
        var_name="Metric", value_name="Score"
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {"Precision": "Precision", "F_score": label_f}
    )

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=plot_df, x="Method", y="Score",
                    hue="Method", palette=method_colors)
    plt.title(f"Region-Gene: Precision vs {label_f} Score (per Method)", fontsize=13)
    plt.ylabel("Score", fontsize= FONT_SIZE)
    plt.xlabel("", fontsize = FONT_SIZE)
    plt.xticks(rotation=30, ha="right", fontsize= FONT_SIZE)
    plt.tick_params(axis='y', labelsize=FONT_SIZE)
    plt.legend(title="", loc = "upper left", bbox_to_anchor=(1.05, 1), ncol = 1, fontsize=FONT_SIZE-1)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path + f"Benchmark of Region-Gene Precision vs {label_f}.png",
                dpi=1200, bbox_inches="tight")
    print(f"✓ 已保存：Region-Gene Precision vs {label_f}")









def plot_PR_curve(pr_curve_data,  benchmark_peak_gene_df, result_df,
                  output_file, method_colors):
    
    FONT_SIZE = 20
    
    plt.figure(figsize=(8, 6))
    for method_name, (rec_c, prec_c, auprc_val) in pr_curve_data.items():
        plt.plot(rec_c, prec_c, color = method_colors.get(method_name, 'blue'),
                label=f"{method_name} \n (AUPRC={auprc_val:.3f})", linewidth=2)

    # 随机基线（正例比例）
    baseline = result_df["N_true"].iloc[0] / len(benchmark_peak_gene_df) \
        if len(result_df) > 0 else 0.5
    plt.axhline(y=baseline, color='gray', linestyle='--',
                linewidth=1.2, label=f"Random (P={baseline:.3f})")

    plt.xlabel("Recall", fontsize = FONT_SIZE)
    plt.ylabel("Precision", fontsize = FONT_SIZE)
    plt.tick_params(labelsize=FONT_SIZE)
    
    plt.title("Benchmark of Region-Gene PR Curve", fontsize=14)
    plt.legend(loc = "upper left", bbox_to_anchor=(1.05, 1), ncol = 1, fontsize = FONT_SIZE-4)
    
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_file + "Benchmark of Region-Gene PR Curve.png",
                dpi=1200, bbox_inches="tight")
    # plt.close()
    print("✓ 已保存：Region-Gene PR Curve")





  



def plot_comparsion(method_data, output_file, score_col="Precision", 
                    palette="Set2", beta = 1, my_order = None):
    """
    对比多种方法的表现。
    
    参数:
    :param method_dict: 字典，Key 为方法名 (str), Value 为对应的 DataFrame。
    :param score_col: 用于排序和评价的列名 (str)，如 "Precision", "Spearman_Rho", "F1"。
    :param palette: 绘图配色方案。
    """
    if score_col == "fscore" or score_col == "F_score":
        y_label = f"$F_{{{beta}}}$ Score"
        title = f"$F_{{{beta}}}$ Score)"
    else:
        y_label = score_col
        title = score_col
    FONT_SIZE = 20
    
    method_key = "Method"
    
    # 2. 绘图
    plt.figure(figsize=(6, 6))
    sns.set_style("whitegrid") # 设置清爽的白色网格背景

    ax = sns.barplot(x=method_key, y=score_col, data=method_data, palette=palette)

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=FONT_SIZE-4)

    plt.title(title, fontsize= FONT_SIZE)
    plt.ylim(0, max(method_data[score_col]) * 1.2) # 给上方数值留点空间
    plt.xlabel("", fontsize=FONT_SIZE) # 去掉 x 轴的 'Method' 字样，因为标签已经很清楚了
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    plt.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE-1)
    plt.tick_params(axis='y', labelsize=FONT_SIZE-1)
    plt.savefig(output_file + 'overall' + title + '.png', dpi=1200, bbox_inches='tight')






def get_whisker_ylim(data, group_col, score_col, margin=0.1):
    """
    按照箱线图的须线（1.5×IQR）计算 y 轴范围，
    与 showfliers=False 的视觉范围完全一致
    """
    all_whisker_vals = []
    for _, group in data.groupby(group_col):
        vals = group[score_col].dropna()
        if len(vals) == 0:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr     = q3 - q1
        lower   = max(vals[vals >= q1 - 1.5 * iqr].min(), 0.0)
        upper   = min(vals[vals <= q3 + 1.5 * iqr].max(), 1.0)
        all_whisker_vals.extend([lower, upper])

    if not all_whisker_vals:
        return 0.0, 1.0

    w_min, w_max = min(all_whisker_vals), max(all_whisker_vals)
    span  = max(w_max - w_min, 1e-4)
    y_min = max(0.0, w_min - span * margin)
    y_max = min(1.0, w_max + span * margin)
    return y_min, y_max

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_top_n_comparison(method_dict, output_file, top_n=None, 
                          score_col="Precision", palette="Set2", my_order = None,
                          decay_curve = True, beta = 1, show_dot = False):
    """
    对比多种方法表现最好的前 N 个基因。
    
    参数:
    :param method_dict: 字典，Key 为方法名 (str), Value 为对应的 DataFrame。
    :param top_n: 取前多少个基因进行对比 (int)。
    :param score_col: 用于排序和评价的列名 (str)，如 "Precision", "Spearman_Rho", "F1"。
    :param palette: 绘图配色方案。
    """
    
    combined_list = []
    
    for method_name, df in method_dict.items():
        # 1. 检查列名是否存在
        if score_col not in df.columns:
            print(f"跳过方法 {method_name}: 找不到列 '{score_col}'")
            continue
        
        if score_col == "fscore":
            df = df.dropna(subset=[score_col, "Precision", "Recall"])
        else:
            df = df.dropna(subset=[score_col])
        
        
        # 2. 排序并提取前 N 个
        # 深拷贝一份以防修改原始数据，并按降序排序
        if top_n:
            temp_df = df.sort_values(by=score_col, ascending=False).head(top_n).copy()
        else:
            temp_df = df.sort_values(by=score_col, ascending=False).copy()
        # 3. 添加必要的信息
        temp_df['Method'] = method_name
        temp_df['Rank'] = range(1, len(temp_df) + 1)
        
        combined_list.append(temp_df[['Method', score_col, 'Rank']])
    
    if not combined_list:
        print("错误: 没有有效的数据可以绘制。")
        return None
        
    # 4. 合并所有数据
    df_plot = pd.concat(combined_list)
    
    # ==========================================
    # 5. 开始绘图
    # ==========================================
    sns.set_theme(style="ticks")

    FONT_SIZE = 20
    
    # 计算排序逻辑（按中位数从高到低排列箱线图）
    if my_order:
        t_order= set(df_plot['Method'])
        order = [method for method in my_order if method in t_order]
    else:
        order = df_plot.groupby('Method')[score_col].median().sort_values(ascending=False).index
    
    df_plot = df_plot.reset_index(drop=True)
    
    # --- 图 1: Top N 分布对比 (Boxplot) ---
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=df_plot, x='Method', y=score_col, 
                order=order, palette=palette, width=0.6, showfliers=show_dot)
    # sns.stripplot(data=df_plot, x='Method', y=score_col, 
    #               order=order, color='black', size=2, alpha=0.3, jitter=True)
    
    if show_dot:
        y_max_axis = min(1, df_plot[score_col].max() * 1.2)
        y_min_axis = max(0, df_plot[score_col].min() * 0.8)
    else:
        y_min_axis, y_max_axis = get_whisker_ylim(df_plot, 'Method', score_col, margin=0.1)

        y_max_axis = min(1, y_max_axis)
        y_min_axis = max(0, y_min_axis)
    
     
    if score_col == "fscore" or score_col == "F_score":
        y_label = f"$F_{{{beta}}}$ Score"
        if top_n:
            title = f"Top {top_n} F{beta} Score Distribution"
        else:
            title = f"$F{beta} Score Distribution"
    else:
        y_label = score_col
    
        if top_n:
            title = f"Top {top_n} {y_label} Distribution"
        else:
            title = f"{y_label} Distribution"
    
    plt.title(title, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    plt.xlabel("", fontsize=FONT_SIZE)
    
    plt.ylim(y_min_axis, y_max_axis)
    
    plt.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE)
    plt.tick_params(axis='y', labelsize=FONT_SIZE)
    plt.tight_layout()
    sns.despine()
    plt.savefig(output_file + title + ".png", dpi=1200, bbox_inches='tight')
    
    if decay_curve:
        # --- 图 2: 性能衰减曲线 (Line Plot) ---
        
        if score_col == "fscore" or score_col == "F_score":
            y_label = f"$F_{{{beta}}}$ Score"
            if top_n:
                title = f"Top {top_n} F{beta} Score Decay Curve"
            else:
                title = f"F{beta} Score Decay Curve"
        else:
            y_label = score_col
        
            if top_n:
                title = f"Top {top_n} {y_label} Decay Curve"
            else:
                title = f"{y_label} Decay Curve"
                
            
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_plot, x='Rank', y=score_col, 
                    hue='Method', hue_order=order, palette=palette, lw=2.5)
        
        plt.title(title, fontsize=FONT_SIZE)
        plt.xlabel("Gene Rank", fontsize=FONT_SIZE)
        plt.ylabel(y_label, fontsize=FONT_SIZE)
        plt.tick_params(labelsize=FONT_SIZE)
        plt.grid(True, alpha=0.3)
        plt.legend(loc = "upper left", bbox_to_anchor=(1.05, 1), ncol = 1, fontsize = FONT_SIZE-1)
        plt.tight_layout()
        sns.despine()
        plt.savefig(output_file + title + ".png", dpi=1200, bbox_inches='tight')
    
    return df_plot # 返回处理后的数据，方便后续查阅






def plot_top_n_mean_comparsion(method_dict, output_file, top_n=None, 
                          score_col="Precision", palette="Set2", my_order = None,
                          beta = 1):
    
    """
    对 analysis_tf_region() 返回的 tf_region_result（每 TF 的 Precision/Recall/F1），
    计算两种整体指标：
    - Macro 平均：各 TF 指标的算术平均（不考虑样本量差异）
    - Micro 平均：将所有 TF 的 TP/FP/FN 累加后重新计算（考虑样本量）

    Parameters
    ----------
    tf_region_result : dict  {method_name: DataFrame}
        每个 DataFrame 需包含列 ['TF', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN']
        （calculate_tf_metrics 的标准输出）
    output_path : str  图片输出目录
    beta : float       F-beta 的 beta 值
    """
    rows = []
    for method_name, df in method_dict.items():
        
        if len(df) == 0:
            rows.append({
            "Method":         method_name,
            "Macro_score":    0,
            "Micro_score":    0,
            "Median_score":   0,
            "N_TF":            len(df),
            })
            continue
        
        if score_col == "fscore":
            df = df.dropna(subset=[score_col, "Precision", "Recall"])
        else:
            df = df.dropna(subset=[score_col])
            
        if len(df) == 0:
            rows.append({
                "Method":         method_name,
                "Macro_score":    0,
                "Micro_score":    0,
                "Median_score":   0,
                "N_TF":            len(df),
            })
            continue
        
        if top_n:
            df = df.sort_values(by=score_col, ascending=False).head(top_n).copy()
        else:
            df = df.sort_values(by=score_col, ascending=False).copy()
        
        # ── Median  ──────────────
        
        if score_col == "fscore" or score_col == "F_score":
            median_prec = df["Precision"].median()
            median_rec  = df["Recall"].median()
            denom_median = (1 + beta**2) * median_prec + beta**2 * median_rec
            median_score = ((1 + beta**2) * median_prec * median_rec / denom_median
                    if denom_median > 0 else 0.0)
        else:
            median_score = df[score_col].median()
        
         
        # ── Macro 平均 ──────────────
        if score_col == "fscore" or score_col == "F_score":
            macro_prec = df["Precision"].mean()
            macro_rec  = df["Recall"].mean()
            denom_macro = (1 + beta**2) * macro_prec + beta**2 * macro_rec
            macro_score = ((1 + beta**2) * macro_prec * macro_rec / denom_macro
                    if denom_macro > 0 else 0.0)
        else:
            macro_score = df[score_col].mean()
        
        # ── Micro 平均（需要原始 TP/FP/FN；若列不存在则跳过）──────────────
        if all(c in df.columns for c in ["TP", "FP", "FN"]):
            total_tp = df["TP"].sum()
            total_fp = df["FP"].sum()
            total_fn = df["FN"].sum()
            if score_col == "fscore" or score_col == "F_score":
                micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                denom_micro = (1 + beta**2) * micro_prec + beta**2 * micro_rec
                micro_score = ((1 + beta**2) * micro_prec * micro_rec / denom_micro
                        if denom_micro > 0 else 0.0)
            elif score_col == "Precision":
                micro_score = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            elif score_col == "Recall":
                micro_score  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            else:
                micro_score = macro_score

        else:
            # TP/FP/FN 列不存在时，以 Macro 代替（保守估计）
            micro_score = macro_score
            print(f"  [警告] {method_name} 缺少 TP/FP/FN 列，Micro 平均以 Macro 代替。")

        rows.append({
            "Method":         method_name,
            "Macro_score":    macro_score,
            "Micro_score":    micro_score,
            "Median_score":   median_score,
            "N_TF":            len(df),
        })

    overall_df = pd.DataFrame(rows)
    
    
    print(overall_df)
    
    
    if score_col == "fscore" or score_col == "F_score":
        label_f = f"$F_{{{beta}}}$"
    else:
        label_f = score_col
    
    if my_order:
        t_order= set(overall_df['Method'])
        order = [method for method in my_order if method in t_order]
    else:
        order = overall_df.sort_values("Macro_score", ascending=False)["Method"].tolist()

    # ── 绘图：4 指标横向对比（2×2 子图）─────────────────────────────────────
    # fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.set_theme(style="whitegrid")
    FONT_SIZE = 20
    
    if top_n:
        Macro_title = f" Top {top_n} Macro {label_f} Score"
        Micro_title = f" Top {top_n} Micro {label_f} Score"
        Median_title = f" Top {top_n} Median {label_f} Score"
    else:
        Macro_title = f" Macro {label_f} Score"
        Micro_title = f" Micro {label_f} Score"
        Median_title = f" Median {label_f} Score"

    metrics = [
        ("Macro_score", Macro_title, f"{label_f}"),
        ("Micro_score", Micro_title, f"{label_f}"),
        ("Median_score", Median_title, f"{label_f}"),
    ]

    for col, title, y_label in metrics:
        
        plt.figure(figsize=(8, 6))
        
        sorted_df = overall_df.sort_values(col, ascending=False)
        
        bp = sns.barplot(data=sorted_df, x="Method", y=col, hue="Method", order= order, palette=palette)
        plt.title(title, fontsize=FONT_SIZE)
        plt.ylabel(y_label, fontsize=FONT_SIZE)
        plt.xlabel("", fontsize = FONT_SIZE)
        
        # 动态计算y轴范围
        y_max = sorted_df[col].max()  # 数据的最大值
        y_max_label = y_max * 1.15  # 为标签留出15%的额外空间
        plt.ylim(0, y_max_label)
        
        plt.tick_params(axis='x', rotation=35, labelsize=FONT_SIZE)
        plt.tick_params(axis='y', labelsize=FONT_SIZE)
        _annotate_bars(bp, fmt='.4f', fontsize = FONT_SIZE-2)

        # plt.suptitle("Benchmark of TF-Region Overall Precision & F Score", fontsize=FONT_SIZE, y=1.01)
        plt.tight_layout()
        plt.savefig(output_file + title +".png",  dpi=1200, bbox_inches="tight")

    return overall_df







def plot_top_n_df_comparison(corr_df, output_file, method_colors, top_n=100, 
                             compare_method = 'Spearman_Rho', order = None, 
                             decay_curve = True, beta = 1):
    """
    提取每种方法相关性最好的前 N 个基因并进行比较
    :param corr_df: 包含 [Method, Gene, Spearman_Rho] 的 DataFrame
    :param top_n: 每种方法选取的基因数量
    """
    
    FONT_SIZE= 20
    
    # 1. 提取每种方法的前 N 个基因
    if top_n:
        top_n_df = corr_df.groupby('Method').apply(
            lambda x: x.sort_values(compare_method, ascending=False).head(top_n)
        ).reset_index(drop=True)
    else:
        top_n_df = corr_df.groupby('Method').apply(
            lambda x: x.sort_values(compare_method, ascending=False)
        ).reset_index(drop=True)
        
    # 2. 为绘图计算排名（Rank 1 到 N）
    top_n_df['Rank'] = top_n_df.groupby('Method')[compare_method].rank(
        ascending=False, method='first'
    ).astype(int)


    if order:
        New_order = top_n_df.groupby('Method')[compare_method].median().sort_values(ascending=False).index
        method_order = [method for method in order if method in New_order]
    else:
        method_order = top_n_df.groupby('Method')[compare_method].median().sort_values(ascending=False).index
    
    # --- 图 1: Top N 基因的相关性分布 (Boxplot) ---
    if compare_method == "Spearman_Rho" or compare_method == "Abs_Spearman_Rho":
        y_label = "Spearman Rho"
    elif compare_method == "fscore":
        y_label = f"$F_{{{beta}}}$ Score"
    else:
        y_label = compare_method
    
    plt.figure(figsize=(6, 6))
    sns.boxplot(
        data=top_n_df, x='Method', y=compare_method, 
        order=method_order, palette=method_colors)
    # sns.stripplot(
    #     data=top_n_df,  x='Method',  y=compare_method, 
    #     order=method_order,  color='black',  alpha=0.3, 
    #     size=2,  jitter=True  )
    plt.title(f'Distribution of {y_label} (Top {top_n} Genes)')
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    plt.xlabel("", fontsize=FONT_SIZE)
    plt.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE-1)
    plt.tick_params(axis='y', labelsize=FONT_SIZE-1)

    plt.tight_layout()
    plt.savefig(output_file + "_distribution.png", dpi=1200, bbox_inches='tight')
    
    # --- 图 2: Top N 基因的相关性衰减曲线 (Line Plot) ---
    # 这展示了随着排名增加，高质量预测消失的速度
    if decay_curve:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=top_n_df, x='Rank', y=compare_method, 
            hue='Method', hue_order=method_order, lw=2.5, palette=method_colors)

        plt.title(f'Performance Profile: Top {top_n} Genes Rank vs. Rho')
        plt.xlabel('Gene Rank', fontsize=FONT_SIZE)
        plt.ylabel(y_label, fontsize=FONT_SIZE)
        plt.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE-1)
        plt.tick_params(axis='y', labelsize=FONT_SIZE-1)
        plt.grid(True, alpha=0.3)
        plt.legend(loc = "upper left", bbox_to_anchor=(1, 1), ncol = 1, fontsize = FONT_SIZE-1)

        plt.tight_layout()
        plt.savefig(output_file + "_Decay.png", dpi=1200, bbox_inches='tight')



    



def plot_top_n_df_mean_comparison(corr_df, output_file, method_colors, top_n=None, 
                             compare_method = 'Spearman_Rho', order = None, beta = 1):
    """
    提取每种方法相关性最好的前 N 个基因并进行比较
    :param corr_df: 包含 [Method, Gene, Spearman_Rho] 的 DataFrame
    :param top_n: 每种方法选取的基因数量
    """
    
    FONT_SIZE= 20
    
    model_key = "Model"
    
    # 1. 提取每种方法的前 N 个基因
    if top_n:
        top_n_df = corr_df.groupby('Method').apply(
            lambda x: x.sort_values(compare_method, ascending=False).head(top_n)
        ).reset_index(drop=True)
    else:
        top_n_df = corr_df.groupby('Method').apply(
            lambda x: x.sort_values(compare_method, ascending=False)
        ).reset_index(drop=True)
        


    if order:
        New_order = top_n_df.groupby('Method')[compare_method].median().sort_values(ascending=False).index
        method_order = [method for method in order if method in New_order]
    else:
        method_order = top_n_df.groupby('Method')[compare_method].median().sort_values(ascending=False).index
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # 3. 绘制条形图
    # 按照 F-beta 从高到低排序会让图表更有说服力
    ax = sns.barplot(x='Model', y=compare_method, 
                     data=top_n_df.sort_values(compare_method, ascending=False), 
                     palette=method_colors,
                     order = method_order)
    
    sorted_df = top_n_df.sort_values(compare_method, ascending=False)
    # 动态计算y轴范围
    y_max = sorted_df[compare_method].dropna().max()  # 数据的最大值
    # 检查是否为有限值（不是Inf或NaN）
    if np.isfinite(y_max):
        y_max_label = y_max * 1.15
        plt.ylim(0, y_max_label)
    else:
        # 处理全是Inf或NaN的情况
        print(f"Warning: No valid finite values in column 'F-beta'. Setting y-axis limit to (0, 1).")
        plt.ylim(0, 1)

    # 在柱状图上方标注具体数值
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points', fontsize = FONT_SIZE-4)

    if compare_method == "F-beta":
        y_label = f"$F_{{{beta}}}$ Score"
        if top_n:
            title = f"Top {top_n} {y_label} Comparison"
        else:
            title = f"{y_label} Comparison"
    else:
        y_label = compare_method
        if top_n:
            title = f"Top {top_n} {y_label} Comparison"
        else:
            title = f"{y_label} Comparison"
    # 4. 添加细节
    plt.title(title, fontsize=15)
    plt.xlabel('Model', fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize= FONT_SIZE)
    plt.xticks(rotation=45, ha='right', fontsize = FONT_SIZE-2) # 防止模型名称太长重叠
    plt.tick_params(axis='y', labelsize=FONT_SIZE-2)
    
    plt.tight_layout()
    plt.savefig(output_file + 'Benchmark of TF-Gene Comparison of F-beta Score.png', 
                dpi=1200, bbox_inches='tight')

  


########################################################################################################
#
# Region-Gene analysis
#
########################################################################################################

from analysis.assess_region_gene import evaluate_per_gene_correlation


from sklearn.metrics import auc, precision_recall_curve, average_precision_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def each_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    model_TN, model_FP, model_FN, model_TP = cm.ravel()
        
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
        
        return {"model_name":model_name, "auc":auc_value, "aupr": aupr_value,
                "precision": precision_value, "recall": recall_value, "f_score": f_score}



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

    
    return peak_gene_result







###################################################################################
#
#
###################################################################################
from scipy import stats

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

        pred = pred[pred['gene'].isin(marker_genes)]
        
        hic = hic[hic['gene'].isin(marker_genes)]
        
        if pred.empty:
            print("警告：筛选后预测结果为空！请检查 Marker Genes 名字是否与 dataframe 一致。")
            return 0.0, pd.DataFrame()

    # 3. (关键步骤) 数据对齐 - Inner Join (交集)
    # 只有同时存在于预测和 Hi-C 中的边才参与相关性计算

    merged_df = pd.merge(pred, hic, on=['region', 'gene'], how='inner')
    
    n_links = len(merged_df)

    
    if n_links < 10:
        print("警告：重叠连接数过少，相关性计算可能不可靠。")

    # 4. 计算相关性
    # Spearman 关注的是“排名”：预测分越高的，是不是 Hi-C 分也越高？
    # 这比 Pearson 更适合，因为 Hi-C 数据通常不服从正态分布
    if method == 'spearman':
        corr, p_val = stats.spearmanr(merged_df['pred_score'], merged_df['hic_score'])
    else:
        corr, p_val = stats.pearsonr(merged_df['pred_score'], merged_df['hic_score'])

    
    return corr, merged_df







def analysis_region_gene(data_path, output_path, method_list):
    
    adata_atac = ad.read_h5ad(data_path + "process/atac_processed.h5ad")
    adata_rna = ad.read_h5ad(data_path + "process/rna_processed.h5ad")
    total_peak = set(adata_atac.var_names)
    total_gene = set(adata_rna.var_names)

    ##################################################################
    benchmark_peak_gene_df = pd.read_pickle(data_path + "process/peak_gene_df.pkl")
    benchmark_peak_gene_df = benchmark_peak_gene_df.rename(columns= {"PeakID":"Peak", 
                                                                     "gene_name":"Gene",
                                                                     "hic_score":"value"})
    benchmark_peak_gene_df["label"] = (benchmark_peak_gene_df["value"]>0).astype(int)
    benchmark_peak_gene_df = benchmark_peak_gene_df[benchmark_peak_gene_df["Peak"].isin(total_peak)].copy()
    benchmark_peak_gene_df = benchmark_peak_gene_df[benchmark_peak_gene_df["Gene"].isin(total_gene)].copy()

    df_hic = benchmark_peak_gene_df.copy()
    df_hic = df_hic.rename(columns={"Peak":"region", "Gene":"gene"})

    ############################################################################################

    Markrer_Genes = adata_rna.var["highly_variable_rank"].copy()
    Markrer_Genes = Markrer_Genes.sort_values()
    top_marker_genes = list(Markrer_Genes.index[:100])


    result_dict ={}
    result_df_dict = {}
    ############################################################################################
    # DYGMAMBA peak-gene grn
    if "DyGMamba" in method_list:
        
        Node_id = pd.read_pickle(data_path + "data_dyg/node_id.pkl")
        graph_df = pd.read_pickle(data_path + "data_dyg/Graph_df.pkl")
        graph_df["Unnamed"] = graph_df.index
        name_list = ["Unnamed", "source_node", "target_node", "time", "label", "edge_idx"]
        New_Graph = graph_df[name_list].copy()
        New_Graph.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx']

        result_path = data_path + 'data_dyg/my_result_run0.npy'
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

        avg_active_peak_gene_grn.drop_duplicates(subset=["Peak", "Gene"], inplace=True)
        avg_active_peak_gene_grn.dropna()

        df_pred = avg_active_peak_gene_grn.copy()
        df_pred = df_pred.rename(columns={"Peak":"region", "Gene":"gene", "avg_ts_weight":"value"})
        avg_active_peak_gene_grn["value"] = avg_active_peak_gene_grn["avg_ts_weight"]

        dyg_peak_gene_result = region_gene_evaluate(benchmark_peak_gene_df, avg_active_peak_gene_grn, 
                                                    model_name="DYGMAMBA", output_path=output_path)

        dyg_corr_score, dyg_plot_data = evaluate_scenic_plus_correlation(df_pred, df_hic, 
                                                                    marker_genes=top_marker_genes)
        
        result_dict.update({"DyGMamba" : abs(dyg_corr_score)})
        
        result_df_dict.update({"DyGMamba" : avg_active_peak_gene_grn})
        
    #######################################################################################
    #
    #
    #######################################################################################
    # GLUE peak-gene grn
    if "GLUE" in method_list:
        
        glue_data_path = data_path + "data_glue/"

        with open(glue_data_path + "gene2peak.pkl", "rb") as f:
            gene_peak_nx = dill.load(f)
        glue_gene_peak_df = nx.to_pandas_edgelist(gene_peak_nx)

        glue_gene_peak_df.drop_duplicates(subset=["source", "target"], inplace=True)
        glue_gene_peak_df.rename(columns={'source':'Gene', 'target':'Peak', 'weight':'predict'}, inplace=True)
        glue_gene_peak_df["predict"] = (glue_gene_peak_df["predict"]>0.9).astype(int)
        glue_gene_peak_df["value"] = glue_gene_peak_df["score"]
        glue_pred = glue_gene_peak_df[["Peak", "Gene", "value"]].copy()
        glue_pred = glue_pred.rename(columns={"Peak":"region", "Gene":"gene"})

        glue_peak_gene_result = region_gene_evaluate(benchmark_peak_gene_df, glue_gene_peak_df, 
                                                    model_name="GLUE", output_path=output_path)  

        glue_corr_score, glue_plot_data = evaluate_scenic_plus_correlation(glue_pred, df_hic, 
                                                                            marker_genes=top_marker_genes)
        
        result_dict.update({"GLUE":abs(glue_corr_score)})
        
        result_df_dict.update({"GLUE": glue_gene_peak_df})

    #######################################################################################
    # Pando peak-gene grn
    if "Pando" in method_list:

        pando_grn = pd.read_csv(data_path + "data_pando/region_gene_network.csv")
        pando_grn.drop_duplicates(subset=["Region", "Gene"], inplace=True)
        pando_grn.dropna()
        pando_gene_peak_df = pando_grn.copy()
        pando_gene_peak_df.rename(columns={"Region": "Peak"}, inplace=True)
        pando_gene_peak_df["predict"] = (pando_gene_peak_df["padj"]<0.05).astype(int)
        pando_gene_peak_df["value"] = -np.log10(pando_gene_peak_df["padj"])
        pando_pred = pando_gene_peak_df[["Peak", "Gene", "value"]].copy()
        pando_pred = pando_pred.rename(columns={"Peak":"region", "Gene":"gene"})

        pando_peak_gene_result = region_gene_evaluate(benchmark_peak_gene_df, pando_gene_peak_df, 
                                                    model_name="Pando", output_path=output_path)  

        pando_corr_score, pando_plot_data = evaluate_scenic_plus_correlation(pando_pred, df_hic, 
                                                                            marker_genes=top_marker_genes)
        
        result_dict.update({"Pando":abs(pando_corr_score)})
        
        result_df_dict.update({"Pando": pando_gene_peak_df})

    #######################################################################################
    # FigR peak-gene grn
    if "FigR" in method_list:
        
        figr_grn = pd.read_csv(data_path + "data_FigR/Region_Gene_Network.csv")
        
        if figr_grn['Region'].str.contains(':').any():

            figr_grn['Region'] = figr_grn['Region'].str.replace(':', '-', regex=False)
            
        else:
            print("格式已是连字符（chr1-100-200），无需转换。")

        figr_grn.drop_duplicates(subset=["Region", "Target_Gene"], inplace=True)

        figr_gene_peak_df = figr_grn.copy()
        figr_gene_peak_df.rename(columns={"Region": "Peak", "Target_Gene":"Gene"}, inplace=True)
        figr_gene_peak_df["predict"] = (figr_gene_peak_df["P_Value"]<0.05).astype(int)
        figr_gene_peak_df["value"] = figr_gene_peak_df["Correlation"]

        figr_pred = figr_gene_peak_df[["Peak", "Gene", "value"]].copy()
        figr_pred = figr_pred.rename(columns={"Peak":"region", "Gene":"gene"})

        figr_peak_gene_result = region_gene_evaluate(benchmark_peak_gene_df, figr_gene_peak_df, 
                                                    model_name="FigR", output_path=output_path)  

        figr_corr_score, figr_plot_data = evaluate_scenic_plus_correlation(figr_pred, df_hic, 
                                                                            marker_genes=top_marker_genes)
        
        result_dict.update({"FigR":abs(figr_corr_score)})
        result_df_dict.update({"FigR": figr_gene_peak_df})

    #######################################################################################
    # LINGER peak-gene grn
    if "LINGER" in method_list:

        linger_grn = pd.read_csv(data_path + "data_linger/train/cell_population_cis_regulatory.txt", 
                                sep='\t', header=None, names=["Peak", "Gene", "reg"])

        if linger_grn['Peak'].str.contains(':').any():

            linger_grn['Peak'] = linger_grn['Peak'].str.replace(':', '-', regex=False)
            
        else:
            print("格式已是连字符（chr1-100-200），无需转换。")

        linger_grn.drop_duplicates(subset=["Peak", "Gene"], inplace=True)

        linger_gene_peak_df = linger_grn.copy()
        linger_gene_peak_df["predict"] = (linger_gene_peak_df["reg"]<0.05).astype(int)
        linger_gene_peak_df["value"] = -np.log10(linger_gene_peak_df["reg"])
        max_val = linger_gene_peak_df.loc[np.isfinite(linger_gene_peak_df["value"]), "value"].max()
        
        linger_gene_peak_df["value"] = linger_gene_peak_df["value"].replace(np.inf, max_val)

        linger_pred = linger_gene_peak_df[["Peak", "Gene", "value"]].copy()
        linger_pred = linger_pred.rename(columns={"Peak":"region", "Gene":"gene"})

        linger_peak_gene_result = region_gene_evaluate(benchmark_peak_gene_df, linger_gene_peak_df, 
                                                    model_name="LINGER", output_path=output_path)

        linger_corr_score, linger_plot_data = evaluate_scenic_plus_correlation(linger_pred, df_hic, 
                                                                            marker_genes=top_marker_genes)

        result_dict.update({"LINGER":abs(linger_corr_score)})
        
        result_df_dict.update({"LINGER": linger_gene_peak_df})
        
    #######################################################################################
    # GRaNIE peak-gene grn
    if "GRaNIE" in method_list:
        
        granie_grn = pd.read_csv(data_path + "GRaNIE_output/GRaNIE_region_gene_Links.csv")

        if granie_grn['peak.ID'].str.contains(':').any():

            granie_grn['peak.ID'] = granie_grn['peak.ID'].str.replace(':', '-', regex=False)
            
        else:
            print("格式已是连字符（chr1-100-200），无需转换。")
        granie_grn.dropna()

        granie_grn = granie_grn.rename(columns={"peak.ID":"Peak", "gene.name":"Gene"})
        granie_grn.dropna()
        granie_grn.drop_duplicates(subset=["Peak", "Gene"], inplace=True)

        granie_gene_peak_df = granie_grn.copy()

        granie_gene_peak_df["predict"] = 1
        granie_gene_peak_df["value"] = -np.log10(granie_gene_peak_df["peak_gene.p_adj"])

        granie_pred = granie_gene_peak_df[["Peak", "Gene", "value"]].copy()
        granie_pred = granie_pred.rename(columns={"Peak":"region", "Gene":"gene"})

        granie_peak_gene_result = region_gene_evaluate(benchmark_peak_gene_df, granie_gene_peak_df, 
                                                    model_name="GRaNIE", output_path=output_path)

        granie_corr_score, granie_plot_data = evaluate_scenic_plus_correlation(granie_pred, df_hic,
                                                                            marker_genes=top_marker_genes)

        result_dict.update({"GRaNIE":abs(granie_corr_score)})
        
        result_df_dict.update({"GRaNIE": granie_gene_peak_df})
        
    #######################################################################################
    # results analysis

    results = pd.DataFrame(result_dict, index=[0])
    
    return results, result_df_dict, benchmark_peak_gene_df




from scipy.stats import spearmanr

def plot_recovery_curves(corr_df, output_file, method_colors, compared_method='Spearman_Rho'):
    """
    恢复曲线：X轴是相关性阈值，Y轴是该方法在该质量下能覆盖的基因总数
    """
    FONT_SIZE = 20
    thresholds = np.linspace(0, 0.9, 100) # 覆盖从低到高的相关性区间
    plot_list = []
    
    for method in corr_df['Method'].unique():
        m_vals = corr_df[corr_df['Method'] == method][compared_method].values
        for thr in thresholds:
            count = np.sum(m_vals > thr)
            plot_list.append({'Method': method, 'Threshold': thr, 'Gene_Count': count})
            
    pdf = pd.DataFrame(plot_list)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=pdf, x='Threshold', y='Gene_Count', hue='Method', palette=method_colors)
    plt.title('Gene Coverage vs Prediction Quality (Spearman Rho)')
    plt.ylabel('Number of Recovered Genes', fontsize = FONT_SIZE)
    plt.xlabel('Spearman Rho (Consistency with Benchmark)', fontsize = FONT_SIZE)
    plt.tick_params(labelsize=FONT_SIZE)
    plt.legend(loc = "upper left", bbox_to_anchor=(1, 1), ncol = 1, fontsize = FONT_SIZE-1)
    
    plt.savefig(output_file, dpi=1200, bbox_inches='tight')





def evaluate_r2g_benchmark(benchmark_df, method_dict, min_links=5):
    """
    评估 R2G 预测性能
    :param benchmark_df: 金标准 [Peak, Gene, value, label]
    :param method_dict: 方法字典，如 {'GRaNIE': df1, 'GLUE': df2}
    :param min_links: 每个基因计算相关性所需的最少边数
    """
    correlation_results = []
    global_metrics = []
    
    # 1. 计算每个基因的相关性（关注质量）
    for method_name, pred_df in method_dict.items():
        # 合并预测与金标准
        merged = pd.merge(
            benchmark_df[['Peak', 'Gene', 'value', 'label']], 
            pred_df[['Peak', 'Gene', 'value']], 
            on=['Peak', 'Gene'],
            how='left',
            suffixes=('_bench', '_method')
        )
        merged['value_method'] = merged['value_method'].fillna(0)
        if merged.empty:
            print(f"警告: 方法 {method_name} 与金标准无共有 Peak-Gene 对，跳过。")
            continue
        # 按基因分组计算 Spearman Rho
        def get_gene_cor(group):
            if len(group) < min_links:
                return np.nan
            # 使用预测分值与金标准连续值（如 Hi-C score）计算相关性
            rho, _ = spearmanr(group['value_bench'], group['value_method'])
            return rho

        gene_rhos = merged.groupby('Gene').apply(get_gene_cor).dropna()
        
        for gene, rho in gene_rhos.items():
            correlation_results.append({
                'Method': method_name,
                'Gene': gene,
                'Spearman_Rho': rho
            })

        try:
            auprc = average_precision_score(merged['label'], merged['value_method'])
            global_metrics.append({
                'Method': method_name, 
                'AUPRC': auprc, 
                'Total_Links': len(pred_df),
                'Common_Links': len(merged)
            })
        except Exception as e:
            print(f"计算 {method_name} 的 AUPRC 时出错: {e}")
            

    corr_df = pd.DataFrame(correlation_results)
    metrics_df = pd.DataFrame(global_metrics)
    
    return corr_df, metrics_df




###########################################################################################
#
#
# complementary
#
#
###########################################################################################



# ══════════════════════════════════════════════════════════════════════════════
# 分析 4：Region-Gene —— Precision / Recall / F1 / AUPRC 柱状图
# ══════════════════════════════════════════════════════════════════════════════

def analysis_region_gene_precision(benchmark_peak_gene_df, method_data, 
                                   output_path, method_colors, beta=1.0):
    """
    计算各方法在 Region-Gene 预测上的分类指标：
    Precision、Recall、F-score、AUPRC（PR 曲线下面积）。

    金标准：benchmark_peak_gene_df 中的 label 列（Hi-C > 0 → 1）。
    预测分数：method_data[method]['value'] 列（连续分数，用 AUPRC）
            以及 method_data[method]['predict'] 列（0/1 标签，用 Precision/F-score）。

    Parameters
    ----------
    benchmark_peak_gene_df : pd.DataFrame
        必须包含 ['Peak', 'Gene', 'label']。
    method_data : dict
        {method_name: DataFrame}，每个 df 必须包含 ['Peak', 'Gene', 'value', 'predict']。
    output_path : str
    beta : float  F-beta 的 beta 值

    Returns
    -------
    pd.DataFrame : 各方法的汇总指标。
    """
    
    FONT_SIZE = 20
    rows = []
    pr_curve_data = {}  # 用于绘制 PR 曲线

    for method_name, pred_df in method_data.items():

        # ── 只保留必要列，并去重 ─────────────────────────────────────────────
        required = [c for c in ['Peak', 'Gene', 'value', 'predict'] if c in pred_df.columns]
        pred_clean = pred_df[required].drop_duplicates(subset=['Peak', 'Gene']).copy()

        # ── 与金标准合并（left join 保证不丢失金标准中的 pair）────────────────
        merged = pd.merge(
            benchmark_peak_gene_df[['Peak', 'Gene', 'label']],
            pred_clean,
            on=['Peak', 'Gene'],
            how='outer'
        )
        # print(f"{method_name}:{len(merged)}")
        merged['value']   = merged['value'].fillna(0.0)
        merged['label']   = merged['label'].fillna(0).astype(int)
        merged['predict'] = merged['predict'].fillna(0).astype(int)

        y_true       = merged['label'].values.astype(int)
        y_score      = merged['value'].values.astype(float)
        y_pred_label = merged['predict'].values.astype(int)

        # ── 分类指标（基于 predict 列二值标签）──────────────────────────────
        tp = int(((y_true == 1) & (y_pred_label == 1)).sum())
        fp = int(((y_true == 0) & (y_pred_label == 1)).sum())
        fn = int(((y_true == 1) & (y_pred_label == 0)).sum())

        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom  = (1 + beta**2) * prec + beta**2 * rec
        fscore = (1 + beta**2) * prec * rec / denom if denom > 0 else 0.0

        # ── AUPRC（基于连续分数）────────────────────────────────────────────
        try:
            auprc = average_precision_score(y_true, y_score)
            prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
            pr_curve_data[method_name] = (rec_curve, prec_curve, auprc)
        except Exception as e:
            auprc = float('nan')
            print(f"  [警告] {method_name} AUPRC 计算失败：{e}")

        rows.append({
            "Method":    method_name,
            "Precision": prec,
            "Recall":    rec,
            "F_score":   fscore,
            "AUPRC":     auprc,
            "TP": tp, "FP": fp, "FN": fn,
            "N_pred": int(y_pred_label.sum()),
            "N_true": int(y_true.sum()),
        })
        label_f = f"F{beta}" if beta != 1.0 else "F1"
        print(f"[Region-Gene] {method_name}: "
            f"Prec={prec:.4f}  Rec={rec:.4f}  {label_f}={fscore:.4f}  "
            f"AUPRC={auprc:.4f}  TP={tp}  FP={fp}  FN={fn}")

    result_df = pd.DataFrame(rows)

    return result_df, pr_curve_data







def analysis_region_gene_per_gene(benchmark_peak_gene_df, method_data,
                                   top_n_list=[20, 50, 100],
                                   beta=1.0):

    label_f   = f"$F_{{{beta}}}$"


    # ============================================================
    # Step 1: 逐方法、逐 Gene 计算指标
    # ============================================================
    def compute_per_gene(benchmark_df, pred_df, method_name, beta):
        required  = [c for c in ['Peak', 'Gene', 'value', 'predict'] if c in pred_df.columns]
        pred_clean = pred_df[required].drop_duplicates(subset=['Peak', 'Gene']).copy()

        merged = pd.merge(
            benchmark_df[['Peak', 'Gene', 'label']],
            pred_clean,
            on=['Peak', 'Gene'],
            how='outer'
        )
        merged['label']   = merged['label'].fillna(0).astype(int)
        merged['predict'] = merged['predict'].fillna(0).astype(int)
        merged['value']   = merged['value'].fillna(0.0).astype(float)

        results = []
        for gene, group in merged.groupby('Gene'):
            y_true  = group['label'].values
            y_pred  = group['predict'].values
            y_score = group['value'].values

            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())

            prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom  = (1 + beta**2) * prec + beta**2 * rec
            fscore = (1 + beta**2) * prec * rec / denom if denom > 0 else 0.0

            # AUPRC：需要至少有一个正例才能计算
            try:
                if y_true.sum() > 0 and y_true.sum() < len(y_true):
                    auprc = average_precision_score(y_true, y_score)
                else:
                    auprc = np.nan
            except Exception:
                auprc = np.nan

            results.append({
                'Gene':      gene,
                'Method':    method_name,
                'Precision': prec,
                'Recall':    rec,
                'F_score':   fscore,
                'AUPRC':     auprc,
                'TP': tp, 'FP': fp, 'FN': fn,
                'N_peaks':   len(group),
            })

        return pd.DataFrame(results)

    per_gene_dfs = {}
    for method_name, pred_df in method_data.items():
        df = compute_per_gene(benchmark_peak_gene_df, pred_df, method_name, beta)
        per_gene_dfs[method_name] = df

    method_list = list(method_data.keys())
    score_cols  = [("Precision", "Precision"),
                   ("Recall",    "Recall"),
                   ("F_score",   label_f),
                   ("AUPRC",     "AUPRC")]

    # ============================================================
    # Step 2: Top-N 均值计算（含 AUPRC）
    # ============================================================
    def compute_topn_mean(per_gene_dfs, score_col, top_n_list, method_list):
        rows = []
        for method in method_list:
            if method not in per_gene_dfs:
                continue
            m_df = per_gene_dfs[method].dropna(subset=[score_col])
            m_df = m_df.sort_values(score_col, ascending=False)
            row  = {"Method": method}
            for n in top_n_list:
                topn       = m_df.head(n)
                row[f"Top{n}"] = topn[score_col].mean() if len(topn) > 0 else 0.0
            rows.append(row)
        return pd.DataFrame(rows).set_index("Method")

    topn_means = {
        col: compute_topn_mean(per_gene_dfs, col, top_n_list, method_list)
        for col, _ in score_cols
    }

    

    return per_gene_dfs, topn_means














########################################################################################################
#
# TF-Gene analysis
#
########################################################################################################


def dyg_tf_gene_result(data_path, model_result_path, jaspar_threshold):
    
    adata_atac = ad.read_h5ad(data_path + "process/atac_processed.h5ad")
    adata_rna = ad.read_h5ad(data_path + "process/rna_processed.h5ad")
    total_peak = set(adata_atac.var_names)

    ##################################################################
    ########################################################################################
    # load tf-region data
    
    jaspar_tf_region_file = data_path + "process/jaspar_data_processed.h5ad"
    jaspar_data = ad.read_h5ad(jaspar_tf_region_file)
    adata_region_tf = filter_jaspar_tf(jaspar_data, score_threshold= jaspar_threshold)


    coo_matrix = adata_region_tf.X.tocoo()
    tf_peak_df = pd.DataFrame({
        'Peak': adata_region_tf.obs_names[coo_matrix.row],
        'TF': adata_region_tf.var_names[coo_matrix.col],
        'value': coo_matrix.data
    })
    tf_peak_df = tf_peak_df[tf_peak_df['TF'].isin(set(adata_rna.var_names))]
    tf_peak_df = tf_peak_df[tf_peak_df["Peak"].isin(set(adata_atac.var_names))]

    dygmamba_tf_peak_df = tf_peak_df
    dygmamba_tf_peak_df = dygmamba_tf_peak_df.rename(columns={"value":"predict"})

    total_peak = set(adata_atac.var_names)
    dygmamba_tf_peak_df = dygmamba_tf_peak_df[dygmamba_tf_peak_df["Peak"].isin(total_peak)].copy()

    ########################################################################################
    # load dygmamba data 
    
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
    # get region-gene network 
    
    result_path = model_result_path + 'my_result_run0.npy'
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

    #################################################################################### 
    # get tf-gene network
    
    dygmamba_tf_peak_df = dygmamba_tf_peak_df.rename(columns={"predict":"value"})
    peak_gene_grn = peak_gene_df[peak_gene_df["Peak"].isin(total_peak)]
    merged_df = pd.merge(dygmamba_tf_peak_df, peak_gene_grn, on='Peak')

    tf_gene_grn = merged_df.groupby(['TF', 'Gene', 'ts']).agg(
        peak_num=('Peak', 'nunique'),   # 对 Peak 列做去重计数，新列名叫 peak_num
        avg_weight=('predict', 'mean'),  # 对 weight 列做均值，新列名叫 avg_weight
        total_weight=('predict', 'sum')  # (可选) 建议顺便算个总权重
    ).reset_index()


    tf_gene_grn.to_pickle(model_result_path + "new_tf_gene_grn_1224.pkl")

    #################################################################################### 
    # get average tf-gene network
    
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
    
    return tf_gene_grn, avg_active_tf_gene_grn, avg_global_tf_gene_grn








from analysis.assess_tf_gene import evaluate_predictability
from pdata.benchmark_data import glue_read_ctx_grn


def analysis_tf_gene_data(data_path, method_list, flag_corr, flag_precision, jaspar_threshold, beta = 1):
    
    if flag_corr:
        adata_atac = ad.read_h5ad(data_path + "process/atac_processed.h5ad")
        adata_rna = ad.read_h5ad(data_path + "process/rna_processed.h5ad")
        matrix_df = pd.DataFrame(adata_rna.X.toarray(), index=adata_rna.obs_names,
                                    columns=adata_rna.var_names)
    if flag_precision:
        # ── 金标准 ──────────────────────────────────────────────────────────────
        benchmark_tf_gene_grn = ad.read_h5ad(data_path + "process/tf_gene_network.h5ad")
        
        benchmark_tf_gene_df = adata_to_dataframe(benchmark_tf_gene_grn)

        benchmark_tf_gene_df = benchmark_tf_gene_df.rename(columns= {"obs":"TF", "var":"Gene"})

        benchmark_tf_gene_threshold = benchmark_tf_gene_df["value"].quantile(0.25)

        benchmark_tf_gene_df["label"] = (benchmark_tf_gene_df["value"] > benchmark_tf_gene_threshold).astype(int)

        benchmark_tf_gene_df.drop_duplicates(subset=['TF', 'Gene'], inplace=True)
    
    
    # ============================================================
    # 新增：提取 JASPAR 有效 TF 集合，所有方法取交集用
    # ============================================================
    jaspar_data     = ad.read_h5ad(data_path + "process/jaspar_data_processed.h5ad")
    jaspar_filtered = filter_jaspar_tf(jaspar_data, jaspar_threshold)
    jaspar_tfs      = set(jaspar_filtered.var_names)

    def apply_jaspar_tf_filter(df, tf_col='TF'):
        """只保留 TF 在 JASPAR 中有 motif 的行"""
        before = len(df)
        df = df[df[tf_col].isin(jaspar_tfs)].copy()
        
        return df
    
    
    correlation_result_dict = {}
    precision_result_dict = {}
    ##########################################################################
    
    ##########################################################################
    # dyg TF-Gene predictability
    if "DyGMamba" in method_list:
        
        dyg_result_path = data_path + "data_dyg/"
        avg_active_tf_gene_grn = pd.read_pickle(dyg_result_path + "average_active_tf_gene_grn.pkl")
        avg_global_tf_gene_grn = pd.read_pickle(dyg_result_path + "average_global_tf_gene_grn.pkl")
        
        if flag_corr:

            active_grn = avg_active_tf_gene_grn[["TF","Gene"]].copy()
            active_grn.rename(columns={'Gene':'Target'},inplace = True)

            active_result, trained_models = evaluate_predictability(matrix_df, active_grn)


            global_grn = avg_global_tf_gene_grn[["TF","Gene"]].copy()
            global_grn.rename(columns={'Gene':'Target'},inplace = True)

            global_result, global_trained_models = evaluate_predictability(matrix_df, global_grn)
            
            # correlation_result_dict.update({"Active": active_result, "Global": global_result})
            correlation_result_dict.update({"DyGMamba": active_result})
        
        if flag_precision:
            
            dyg_active_tf_gene_grn = avg_active_tf_gene_grn[["TF","Gene","avg_total_weight"]].copy()
            dyg_active_tf_gene_grn["predict_label"] = (dyg_active_tf_gene_grn["avg_total_weight"]>1).astype(int)
            dyg_active_tf_gene_grn.drop_duplicates(subset=['TF', 'Gene'], inplace=True)
            
            precision_result_dict.update({"DyGMamba": dyg_active_tf_gene_grn})

    ##########################################################################
    # GLUE TF-Gene predictability
    if "GLUE" in method_list:
        if flag_corr:
            glue_grn = glue_read_ctx_grn(data_path + "data_glue/pruned_grn.csv")
            glue_df = nx.to_pandas_edgelist(glue_grn, source="TF", target="Gene")

            glue_df  = apply_jaspar_tf_filter(glue_df)              # ← 新增
            
            glue_df.rename(columns={'Gene':'Target'},inplace = True)
            glue_result, glue_trained_models = evaluate_predictability(matrix_df, glue_df)
            
            correlation_result_dict.update({"GLUE": glue_result})
        
        if flag_precision:
            glue_data_path = data_path + "data_glue/"
            glue_df2 = glue_read_full_grn(glue_data_path + 'pruned_grn.csv')
            glue_df2 = glue_df2[['TF', 'Target', 'Importance', 'MotifID', 'NES', \
                'AUC', 'RankAtMax', 'MotifSimilarityQvalue', 'Annotation']]
            glue_grn2 = glue_df2[["TF","Target","Importance"]].rename(columns = {"Target":"Gene"})
            
            glue_grn2 = apply_jaspar_tf_filter(glue_grn2)           # ← 新增
            
            glue_grn2['predict_label'] = (glue_grn2["Importance"]>0).astype(int) 
            glue_grn2.drop_duplicates(subset=['TF', 'Gene'], inplace=True)
            
            precision_result_dict.update({"GLUE": glue_grn2})

    ##########################################################################
    # FigR TF-Gene predictability
    if "FigR" in method_list:
        figr_grn0 = pd.read_csv(data_path + "data_FigR/TF_Gene_Network.csv")
        
        if flag_corr:
            figr_df = figr_grn0[["TF","Target_Gene"]].copy()
            figr_df.rename(columns={'Target_Gene':'Target'},inplace = True)
            
            figr_df = apply_jaspar_tf_filter(figr_df)                        # ← 新增
                
            figr_result, figr_trained_models = evaluate_predictability(matrix_df, figr_df)
            
            correlation_result_dict.update({"FigR": figr_result})
        
        if flag_precision:
            figr_grn = figr_grn0[["TF","Target_Gene", "Correlation"]].copy()
            figr_grn.rename(columns={"Target_Gene": "Gene", "Correlation":"score"}, inplace=True)
            figr_grn["score"] = abs(figr_grn["score"])
            
            figr_grn = apply_jaspar_tf_filter(figr_grn)              # ← 新增
            
            figr_grn["predict_label"] = (figr_grn["score"]>0).astype(int)
            figr_grn.drop_duplicates(subset=['TF', 'Gene'], inplace=True)
        
            precision_result_dict.update({"FigR": figr_grn})

    ##########################################################################
    # Celloracle TF-Gene predictability
    if "CellOracle" in method_list:

        celloracle_grn = pd.read_csv(data_path + "data_celloracle/celloracle_results/grn_df_" + "cluster0" + ".csv")

        celloracle_grn.rename(columns={"source": "TF", "target": "Gene"}, inplace= True)

        if flag_corr:
            celloracle_df = celloracle_grn[["TF", "Gene"]].copy()
            celloracle_df.rename(columns={'Gene':'Target'},inplace = True)

            co_df = apply_jaspar_tf_filter(celloracle_df)                        # ← 新增
                
        
            celloracle_result, celloracle_trained_models = evaluate_predictability(matrix_df, co_df)
            
            correlation_result_dict.update({"CellOracle": celloracle_result})

        if flag_precision:
            celloracle_df2 = celloracle_grn[["TF", "Gene", "coef_abs"]].copy()
            celloracle_df2.rename(columns={"coef_abs": "score"}, inplace=True)
            
            celloracle_df2 = apply_jaspar_tf_filter(celloracle_df2)                  # ← 新增
            
            celloracle_df2["predict_label"] = (celloracle_df2["score"]>0).astype(int)
            celloracle_df2.drop_duplicates(subset=['TF', 'Gene'], inplace=True)
            
            precision_result_dict.update({"CellOracle": celloracle_df2})


    ##########################################################################
    # LINGER TF-Gene predictability
    
    if "LINGER" in method_list:

        linger_grn = pd.read_csv(data_path + "data_linger/train/cell_population_trans_regulatory.txt", sep='\t')
        linger_grn = linger_grn.rename(columns={'Unnamed: 0': 'Gene'})
        linger_df= linger_grn.melt(id_vars=['Gene'], var_name='TF', value_name='regulation')

        if flag_corr:
            linger_df = linger_df[["TF", "Gene"]].copy()
            linger_df.rename(columns={'Gene':'Target'},inplace = True)

            linger_df = apply_jaspar_tf_filter(linger_df)                       # ← 新增
            
            linger_result, linger_trained_models = evaluate_predictability(matrix_df, linger_df)
            
            correlation_result_dict.update({"LINGER": linger_result})
        
        if flag_precision:
            linger_df2= linger_grn.melt(id_vars=['Gene'], var_name='TF', value_name='regulation')
            
            linger_df2.rename(columns={"regulation": "score"}, inplace=True)

            linger_grn2 = linger_df2[["TF", "Gene", "score"]].copy()
            
            linger_grn2 = apply_jaspar_tf_filter(linger_grn2)         # ← 新增
            
            quantile_value = linger_grn2["score"].quantile(0.25)
            linger_grn2["predict_label"] = (linger_grn2["score"] > quantile_value).astype(int)
            linger_grn2.drop_duplicates(subset=['TF', 'Gene'], inplace=True)
            
            precision_result_dict.update({"LINGER": linger_grn2})

    ##########################################################################
    # GRaNIE TF-Gene predictability
    if "GRaNIE" in method_list:
        granie_grn = pd.read_csv(data_path + "GRaNIE_output/GRaNIE_tf_gene_Links.csv")
        
        if flag_corr:
            granie_df = granie_grn[["TF.name", "gene.name"]].copy()
            granie_df.rename(columns={"TF.name": "TF", "gene.name": "Target"}, inplace=True)
            
            granie_df = apply_jaspar_tf_filter(granie_df)             # ← 新增  
            
            granie_result, granie_trained_models = evaluate_predictability(matrix_df, granie_df)
            
            correlation_result_dict.update({"GRaNIE": granie_result})
        
        if flag_precision:
            granie_grn.dropna()
            granie_grn["score"] = granie_grn["TF_peak.r"].abs()* granie_grn['peak_gene.r'].abs()

            granie_grn2 = granie_grn[["TF.name", "gene.name", "score"]].copy()
            granie_grn2.rename(columns={"TF.name": "TF", "gene.name": "Gene"}, inplace=True)
            
            granie_grn2 = apply_jaspar_tf_filter(granie_grn2)         # ← 新增
            
            granie_grn_quantile = granie_grn2["score"].quantile(0.25)
            granie_grn2["predict_label"] = (granie_grn2["score"] > granie_grn_quantile).astype(int)
            granie_grn2.drop_duplicates(subset=['TF', 'Gene'], inplace=True)
            
            precision_result_dict.update({"GRaNIE": granie_grn2})

    ##########################################################################
    # Pando TF-Gene predictability
    if "Pando" in method_list:
        pando_grn = pd.read_csv(data_path + "data_pando/tf_gene_network.csv")

        if flag_corr:
            pando_df = pando_grn[["TF", "Gene"]].copy()
            pando_df.rename(columns={"Gene": "Target"}, inplace=True)
            
            pando_df = apply_jaspar_tf_filter(pando_df)                        # ← 新增
            
            pando_result, pando_trained_models = evaluate_predictability(matrix_df, pando_df)
            
            correlation_result_dict.update({"Pando": pando_result})
        
        if flag_precision:
            
            pando_grn["predict_label"] = (pando_grn["padj"]<0.04).astype(int)

            pando_grn2 = apply_jaspar_tf_filter(pando_grn)           # ← 新增
            
            pando_grn2 = pando_grn[["TF", "Gene", "predict_label"]].copy()
            pando_grn2.drop_duplicates(subset=['TF', 'Gene'], inplace=True)
            
            precision_result_dict.update({"Pando": pando_grn2})
            
    per_precision_result_df = {}
    if flag_precision:
        
        result_list = []
        for model_name, model_df in precision_result_dict.items():
            # All precision
            merged_data = pd.merge(benchmark_tf_gene_df, model_df, on = ["TF", "Gene"], how="outer").fillna(0)
            y_true = merged_data["label"]
            y_pred = merged_data["predict_label"]
            TN, FP, FN, TP = each_confusion_matrix(y_true, y_pred)
            precision, recall, FPR, AUC, f_beta = manual_assess(TN, FP, FN, TP, beta = beta )
            result_list.append({
                "Model": model_name,
                "Precision": precision,
                "Recall": recall,
                "FPR": FPR,
                "AUC": AUC,
                "F-beta": f_beta
            })
            
            per_tf = _compute_per_tf_metrics(benchmark_tf_gene_df, model_df, beta=beta)
            per_tf["Method"] = model_name
            per_precision_result_df.update({model_name: per_tf})

        precision_result_dataframe = pd.DataFrame(result_list)
        
        # per precision
        
        return correlation_result_dict, precision_result_dict, precision_result_dataframe, per_precision_result_df
    
    
    ##########################################################################
    precision_result_dataframe = pd.DataFrame([])
    
    return correlation_result_dict, precision_result_dict, precision_result_dataframe, per_precision_result_df

    
    
    
    
    
    

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_n_precision_mean(
    result_dict, output_path, method_colors,
    top_n=50, score_col='Precision', my_order=None
):
    """
    计算每个方法 Top-N Precision 的均值，并绘制柱状图
    """

    FONT_SIZE = 20
    sns.set_theme(style="whitegrid")

    if my_order is None:
        my_order = list(result_dict.keys())

    mean_list = []
    std_list = []
    labels = []

    # ===== 核心：计算 Top-N 均值 =====
    for method in my_order:
        if method not in result_dict:
            continue

        df = result_dict[method].copy()

        # 排序 + 取 Top N
        top_df = df.sort_values(by=score_col, ascending=False).head(top_n)

        mean_val = top_df[score_col].mean()
        std_val = top_df[score_col].std()

        mean_list.append(mean_val)
        std_list.append(std_val)
        labels.append(method)

    df_mean = pd.DataFrame({
        "Method": labels,
        "MeanPrecision": mean_list,
        "Std": std_list
    })
    y_max = df_mean["MeanPrecision"].max()

    # ===== 排序（更有说服力）=====
    df_mean = df_mean.sort_values(by="MeanPrecision", ascending=False)

    # ===== 开始绘图 =====
    plt.figure(figsize=(8, 6), dpi=300)

    ax = sns.barplot(
        data=df_mean,
        x="Method",
        y="MeanPrecision",
        palette=method_colors,
        edgecolor='black',
        linewidth=1.2
    )

    # # ===== 误差线（论文加分项）=====
    # ax.errorbar(
    #     x=np.arange(len(df_mean)),
    #     y=df_mean["MeanPrecision"],
    #     yerr=df_mean["Std"],
    #     fmt='none',
    #     ecolor='black',
    #     capsize=4,
    #     linewidth=1.2
    # )

    # ===== 数值标注 =====
    for i, v in enumerate(df_mean["MeanPrecision"]):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=FONT_SIZE-4)

    # ===== 美化 =====
    sns.despine()

    ax.set_ylabel(f"Mean Precision (Top {top_n})", fontsize=FONT_SIZE)
    ax.set_xlabel("")

    ax.set_xticklabels(df_mean["Method"], rotation=30, ha="right", fontsize=FONT_SIZE-2)
    ax.tick_params(axis='y', labelsize=FONT_SIZE-2)

    # Precision 一般在 0~1
    ax.set_ylim(0, y_max * 1.2)

    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.grid(False)

    plt.title(
        f"Comparison of Top-{top_n} TF {score_col} Across Methods",
        fontsize=15, pad=10
    )

    plt.tight_layout()

    save_path = os.path.join(
        output_path,
        f"Top_{top_n}_Mean_{score_col}_Barplot.png"
    )
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


    
    
from analysis.assess import glue_read_full_grn
from data_preprocess import adata_to_dataframe
from analysis.assess import manual_assess

def analysis_tf_gene_all_precision(result_dataframe, output_path, method_colors):

    #######################################
    # f-scpore figure plot

    FONT_SIZE = 20
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # 3. 绘制条形图
    # 按照 F-beta 从高到低排序会让图表更有说服力
    ax = sns.barplot(x='Model', y='F-beta', 
                     data=result_dataframe.sort_values('F-beta', ascending=False), 
                     palette=method_colors)
    
    sorted_df = result_dataframe.sort_values('F-beta', ascending=False)
    # 动态计算y轴范围
    y_max = sorted_df['F-beta'].dropna().max()  # 数据的最大值
    # 检查是否为有限值（不是Inf或NaN）
    if np.isfinite(y_max):
        y_max_label = y_max * 1.15
        plt.ylim(0, y_max_label)
    else:
        # 处理全是Inf或NaN的情况
        print(f"Warning: No valid finite values in column 'F-beta'. Setting y-axis limit to (0, 1).")
        plt.ylim(0, 1)

    # 在柱状图上方标注具体数值
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points', fontsize = FONT_SIZE-4)

    # 4. 添加细节
    plt.title('Benchmark of TF-Gene Comparison of F-beta Score', fontsize=15)
    plt.xlabel('Model Name', fontsize=FONT_SIZE)
    plt.ylabel('F-beta Score', fontsize= FONT_SIZE)
    plt.xticks(rotation=45, ha='right', fontsize = FONT_SIZE-2) # 防止模型名称太长重叠
    plt.tick_params(axis='y', labelsize=FONT_SIZE-2)
    
    plt.tight_layout()
    plt.savefig(output_path + 'Benchmark of TF-Gene Comparison of F-beta Score.png', 
                dpi=1200, bbox_inches='tight')
    
    
    ############################
    # precision igure plot

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ana_col = 'Precision'

    ax = sns.barplot(x='Model', y='Precision', 
                     data=result_dataframe.sort_values('Precision', ascending=False), 
                     palette=method_colors)

    sorted_df = result_dataframe.sort_values(ana_col, ascending=False)
    # 动态计算y轴范围
    y_max = sorted_df[ana_col].dropna().max()  # 数据的最大值
    # 检查是否为有限值（不是Inf或NaN）
    if np.isfinite(y_max):
        y_max_label = y_max * 1.15
        plt.ylim(0, y_max_label)
    else:
        # 处理全是Inf或NaN的情况
        print(f"Warning: No valid finite values in column {ana_col}. Setting y-axis limit to (0, 1).")
        plt.ylim(0, 1)

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points', fontsize = FONT_SIZE-4)
    
    plt.title('Benchmark of TF-Gene Comparison of Precision', fontsize=15)
    plt.xlabel('Model Name', fontsize=FONT_SIZE)
    plt.ylabel('Precision', fontsize=FONT_SIZE)
    plt.xticks(rotation=45, ha='right', fontsize = FONT_SIZE-2)
    plt.tick_params(axis='y', labelsize=FONT_SIZE-2) 
    plt.tight_layout()
    plt.savefig(output_path + 'Benchmark of TF-Gene Comparison of Precision.png', 
                dpi=1200, bbox_inches='tight')
    






def _compute_per_tf_metrics(benchmark_df, pred_df, beta=1.0):
    """
    按 TF 粒度计算 Precision、Recall、F-score。

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        金标准，必须包含列 ['TF', 'Gene', 'label']。
        label=1 表示存在真实调控关系。
    pred_df : pd.DataFrame
        方法预测结果，必须包含列 ['TF', 'Gene', 'predict_label']。
    beta : float
        F-beta 中的 beta 值；beta=1 即 F1，beta=0.5 更重视 Precision。

    Returns
    -------
    pd.DataFrame : 每行为一个 TF，包含 ['TF', 'Precision', 'Recall', 'F_score']。
    """
    merged = pd.merge(
        benchmark_df[['TF', 'Gene', 'label']],
        pred_df[['TF', 'Gene', 'predict_label']],
        on=['TF', 'Gene'],
        how='outer'
    ).fillna(0)

    records = []
    for tf, grp in merged.groupby('TF'):
        y_true = grp['label'].values.astype(int)
        y_pred = grp['predict_label'].values.astype(int)

        # 至少有一个正样本才计算
        if y_true.sum() == 0:
            continue

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = (1 + beta**2) * prec + beta**2 * rec
        fscore = (1 + beta**2) * prec * rec / denom if denom > 0 else 0.0

        records.append({'TF': tf, 'Precision': prec, 'Recall': rec, 'F_score': fscore})

    return pd.DataFrame(records)







def analysis_tf_gene_correlation_summary(result_dict, output_path, method_colors):
    """
    对 analysis_tf_gene_data() 返回的 result_dict，汇总每个方法的
    Correlation 统计量（median / mean），绘制柱状图用于方法间整体对比。

    此图补充了现有箱线图无法直接读出"方法整体水平高低"的不足。

    Parameters
    ----------
    result_dict : dict  {method_name: DataFrame(含 'Correlation' 列)}
    output_path : str   图片输出目录
    """
    FONT_SIZE = 20
    summary_rows = []
    for method_name, df in result_dict.items():
        corr_vals = df["Correlation"].dropna()
        summary_rows.append({
            "Method":          method_name,
            "Median_Corr":     corr_vals.median(),
            "Mean_Corr":       corr_vals.mean(),
            "Std_Corr":        corr_vals.std(),
            "N_TF":            len(corr_vals),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("Median_Corr", ascending=False)

    # ── 柱状图（Median Correlation） ─────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    ana_col = 'Median_Corr'
    plt.figure(figsize=(8, 6))
    # 左：Median
    bp = sns.barplot(data=summary_df, x="Method", y=ana_col,
                    palette= method_colors)
    
    plt.title("TF-Gene Predictability: Median Correlation", fontsize=13)
    plt.ylabel("Median Pearson Correlation", fontsize=FONT_SIZE)
    
    # 动态计算y轴范围
    y_max = summary_df[ana_col].dropna().max()  # 数据的最大值
    # 检查是否为有限值（不是Inf或NaN）
    if np.isfinite(y_max):
        y_max_label = y_max * 1.15
        plt.ylim(0, y_max_label)
    else:
        # 处理全是Inf或NaN的情况
        print(f"Warning: No valid finite values in column")
        plt.ylim(0, 1)
            
    plt.xlabel("", fontsize = FONT_SIZE)
    plt.tick_params(axis='x', rotation=35, labelsize = FONT_SIZE)
    plt.tick_params(axis='y', labelsize = FONT_SIZE)
    _annotate_bars(bp, fmt='.4f', fontsize = FONT_SIZE-4)
    plt.tight_layout()
    plt.savefig(output_path + "Benchmark of TF-Gene Overall Correlation Summary -Median.png",
                dpi=1200, bbox_inches="tight")
    
    # 右：Mean ± Std（误差棒）
    plt.figure(figsize=(8, 6))
    bp2 = sns.barplot(data=summary_df, x="Method", y="Mean_Corr",
                    palette=method_colors)
    # 手动添加误差棒
    for i, row in enumerate(summary_df.itertuples()):
        plt.errorbar(i, row.Mean_Corr, yerr=row.Std_Corr,
                        fmt='none', color='black', capsize=4, linewidth=1.2)
    plt.title("TF-Gene Predictability: Mean ± Std Correlation", fontsize=13)
    plt.ylabel("Mean Pearson Correlation", fontsize= FONT_SIZE)

        
    plt.xlabel("", fontsize = FONT_SIZE)
    plt.tick_params(axis='x', rotation=35, labelsize = FONT_SIZE)
    plt.tick_params(axis='y', labelsize = FONT_SIZE)

    plt.tight_layout()
    plt.savefig(output_path + "Benchmark of TF-Gene Overall Correlation Summary -overall.png",
                dpi=1200, bbox_inches="tight")


    return summary_df



def plot_top_n_per_tf_comparison(
    per_tf_dict,
    output_path,
    top_n       = 50,
    score_col   = "Precision",
    my_order    = None,
    palette     = "Set2",
    tag         = "",
):
    """
    对多种方法的 per-TF 指标取 Top N，绘制双面板图：
    左面板：Top N TF 的分布箱线图（+ strip plot）
    右面板：Top N TF 按排名排列的性能衰减折线图

    Parameters
    ----------
    per_tf_dict : dict
        {method_name: pd.DataFrame}
        每个 DataFrame 需包含列 ['TF', score_col]。
        可直接使用 analysis_tf_gene_per_tf_precision() 返回的中间数据。
    output_path : str
        图片保存目录。
    top_n : int
        每种方法取得分最高的前 N 个 TF。
    score_col : str
        排序与绘图所用的列名，取 'Precision' 或 'F_score'。
    my_order : list or None
        指定 X 轴方法顺序；None 则按各方法的 Top N 中位数降序自动排列。
    palette : str
        seaborn 配色方案。
    tag : str
        附加到文件名的标签，用于区分同一 score_col 不同 top_n 的输出。

    Returns
    -------
    pd.DataFrame : 合并后用于绘图的数据，含列 ['Method', score_col, 'Rank']。
    """

    # ── 1. 逐方法取 Top N ────────────────────────────────────────────────────
    combined_list = []

    order_keys = my_order if my_order is not None else list(per_tf_dict.keys())

    for method_name in order_keys:
        if method_name not in per_tf_dict:
            print(f"  [跳过] {method_name}：不在 per_tf_dict 中")
            continue

        df = per_tf_dict[method_name].copy()

        if score_col not in df.columns:
            print(f"  [跳过] {method_name}：找不到列 '{score_col}'")
            continue

        # 降序取前 top_n 个
        top_df          = df.sort_values(by=score_col, ascending=False).head(top_n).copy()
        top_df["Method"] = method_name
        top_df["Rank"]   = range(1, len(top_df) + 1)  # 1 = 最优 TF

        combined_list.append(top_df[["Method", score_col, "Rank", "TF"]])

    if not combined_list:
        print("  [错误] 没有任何有效数据，跳过绘图。")
        return None

    df_plot = pd.concat(combined_list, ignore_index=True)

    # ── 2. 确定方法顺序（按 Top N 中位数降序）────────────────────────────────
    order = (
        df_plot.groupby("Method")[score_col]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    # 保留用户指定顺序（但仅保留实际存在的方法）
    if my_order is not None:
        order = [m for m in my_order if m in order]

    # ── 3. 构造 X 轴标签（含实际 TF 数量）────────────────────────────────────
    x_labels = []
    for m in order:
        n_actual = df_plot[df_plot["Method"] == m]["TF"].nunique()
        x_labels.append(f"{m}\n(top {n_actual})")

    # ── 4. 双面板绘图 ─────────────────────────────────────────────────────────
    
    FONT_SIZE = 20
    
    sns.set_theme(style="ticks")

    # ── 左面板：Top N TF 分布箱线图 ──────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data      = df_plot,
        x         = "Method",
        y         = score_col,
        order     = order,
        palette   = palette,
        width     = 0.55,
        showfliers= False
    )
    sns.stripplot(
        data   = df_plot,
        x      = "Method",
        y      = score_col,
        order  = order,
        color  = "black",
        size   = 2.5,
        alpha  = 0.35,
        jitter = True
    )
    plt.title(
        f"TF-Gene Per-TF {score_col}: Top {top_n} TF Distribution",
        fontsize= FONT_SIZE,
    )
    plt.ylabel(score_col, fontsize= FONT_SIZE)
    plt.xlabel("", fontsize = FONT_SIZE)
    plt.tick_params(axis='y', labelsize=FONT_SIZE)
    plt.tick_params(axis='x', rotation=30, labelsize=FONT_SIZE)
    sns.despine()
        # ── 5. 保存 ──────────────────────────────────────────────────────────────
    fname = (
        f"Benchmark of TF-Gene Per-TF {score_col} Top{top_n}"
        + (f" {tag}" if tag else "") + "Distribution"
        + ".png"
    )
    fpath = os.path.join(output_path, fname)
    plt.savefig(fpath, dpi=1200, bbox_inches="tight")
    
    
    # ── 右面板：性能衰减折线图 ────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data      = df_plot,
        x         = "Rank",
        y         = score_col,
        hue       = "Method",
        hue_order = order,
        palette   = palette,
        lw        = 2.2
    )
    plt.title(
        f"TF-Gene Per-TF {score_col}: Top {top_n} TF Decay Curve",
        fontsize=13 )
    
    plt.xlabel("TF Rank (sorted by score, 1 = best)", fontsize= FONT_SIZE)
    plt.ylabel(score_col, fontsize= FONT_SIZE)
    plt.tick_params(labelsize = FONT_SIZE-2)
    plt.legend(loc = "upper left", bbox_to_anchor=(1, 1), ncol = 1, fontsize=FONT_SIZE-1)
    plt.grid(True, alpha=0.3)
    sns.despine()

    plt.title(
        f"Benchmark of TF-Gene Per-TF {score_col} — Top {top_n} TFs",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    # ── 5. 保存 ──────────────────────────────────────────────────────────────
    fname = (
        f"Benchmark of TF-Gene Per-TF {score_col} Top{top_n}"
        + (f" {tag}" if tag else "") + "Decay Curve"
        + ".png"
    )
    fpath = os.path.join(output_path, fname)
    plt.savefig(fpath, dpi=1200, bbox_inches="tight")
    # plt.close()
    print(f"  ✓ 已保存：{fname}")

    return df_plot



# ──────────────────────────────────────────────────────────────────────────────
# 完整流程函数：数据加载 → per-TF 指标计算 → Top N 绘图
# ──────────────────────────────────────────────────────────────────────────────

def analysis_tf_gene_per_tf_topn(per_tf_dict, output_path, method_colors,
                                top_n_list  = [20, 50, 100],
                                beta        = 1.0,
                                my_order    = None):
    """
    完整流程：
    1. 加载各方法的 TF-Gene GRN（与 analysis_tf_gene_all_precision 相同路径）
    2. 以 JASPAR 网络为金标准，逐 TF 计算 Precision / Recall / F_score
    3. 对 Precision 和 F_score 分别绘制多个 Top N 的双面板图

    Parameters
    ----------

    output_path : str   图片输出目录
    method_list : list  参与对比的方法名列表
    top_n_list  : list  要输出的 Top N 列表，如 [20, 50, 100]
    beta        : float F-beta 的 beta 值（默认 1.0 = F1）
    my_order    : list or None  方法显示顺序

    Returns
    -------
    dict : {method_name: pd.DataFrame}
        每个 DataFrame 含列 ['TF', 'Precision', 'Recall', 'F_score']
    """


    # ── 对 Precision 和 F_score 分别输出所有 Top N 图 ─────────────────────────
    label_f = f"F{beta}" if beta != 1.0 else "F1"

    for score_col in ["Precision", "F_score"]:
        col_label = score_col if score_col == "Precision" else label_f
        print(f"\n── 正在绘制 {col_label} 的 Top N 系列图 ──")

        for top_n in top_n_list:
            plot_top_n_per_tf_comparison(
                per_tf_dict = per_tf_dict,
                output_path = output_path,
                top_n       = top_n,
                score_col   = score_col,
                my_order    = my_order,
                tag         = col_label,
                palette     = method_colors,
            )


