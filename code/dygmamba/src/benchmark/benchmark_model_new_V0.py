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
import networkx as nx

sys.path.append('/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src/')


########################################################################################################
#
# TF recovery analysis
#
########################################################################################################

from analysis.assess_tf_recovery import calculate_recovery_metrics
from pdata.benchmark_data import glue_read_ctx_grn

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
    jaspar_tfs = set(jaspar_filtered.var_names)  # JASPAR 中有 motif 的 TF
    print(f"JASPAR 有效 TF 数量: {len(jaspar_tfs)}")
    
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
        
        # tf_peak_grn = ad.read_h5ad(data_path + "process/tf_peak_network.h5ad")

        # dyg_tfs = set(tf_peak_grn.var_names)
        
        dyg_tfs = set(tf_peak_df["TF"])

        dyg_x, dyg_y, dyg_raw_auc, dyg_norm_auc = \
            calculate_recovery_metrics(ground_truth_ranked, dyg_tfs, top_n=top_num)

        print(f"Top 40 AUC (Raw): {dyg_raw_auc:.2f}")
        print(f"Top 40 AUC (Normalized): {dyg_norm_auc:.4f}")

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

        # glue_tfs = set(df_edges["TF"])
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

        oracle_tf_gene = celloracle_grn[["TF", "Gene", "-logp"]].copy()

        # oracle_tfs = set(oracle_tf_gene["TF"])
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

        # figr_tfs = set(figr_grn["TF"])
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

        # linger_tfs = set(linger_grn.columns[1:])
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

        # granie_tfs = set(granie_grn["TF.name"])
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

        # pando_tfs = set(pando_grn["TF"])
        pando_tfs  = set(pando_grn["TF"]) & jaspar_tfs         # ← 取交集

        pando_x, pando_y, pando_raw_auc, pando_norm_auc = \
            calculate_recovery_metrics(ground_truth_ranked, pando_tfs, top_n=top_num)
            
        if max(pando_y) > max_num:
            max_num = max(pando_y)

        plt.plot(pando_x, pando_y, label=f'{"FigR"} \n(AUC={pando_norm_auc:.2f})', 
                color=method_colors['Pando'], linewidth=2, marker='o', markersize=4)


    ############################################################################
    FONT_SIZE = 20
    plt.title('TF Recovery Curve', fontsize=FONT_SIZE)
    plt.legend(loc = "upper left", bbox_to_anchor=(1.05, 1), ncol = 1, fontsize = FONT_SIZE-1)
    plt.grid(True, linestyle='--', alpha=0.3)
    max_tf = min(40, len(ground_truth_ranked))
    
    max_tf2 = min(max_tf, max_num)  # 确保 x 轴范围至少覆盖所有方法的 TF 数量
    print(f"Ground Truth TFs: {len(ground_truth_ranked)}, Max TF for x-axis: {max_tf}, Max TF in methods: {max_num}")
    plt.xlim(0, max_tf)
    plt.ylim(0, int(max_tf2*1.1)) # 或者是实际恢复的最大值
    plt.ylabel("Number of Top TFs Recovered", fontsize=FONT_SIZE)
    plt.xlabel("Rank TF", fontsize = FONT_SIZE)
    plt.tick_params(labelsize=FONT_SIZE)

    plt.savefig(output_path + 'Benchmark of TF Recovery Curve.png', dpi=1200, bbox_inches='tight')
    
    

########################################################################################################
#
# TF-Region analysis
#
########################################################################################################

from pdata.data_preprocess import filter_jaspar_tf, adata_to_dataframe
from pdata.data_preprocess import build_tf_peak_network
from analysis.assess_tf_region import calculate_tf_metrics
import dill

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
    jaspar_key_set = set(zip(
        jaspar_filtered.obs_names[coo.row],   # Peak
        jaspar_filtered.var_names[coo.col]    # TF
    ))
    print(f"JASPAR 过滤集合大小: {len(jaspar_key_set)} 个 (Peak, TF) 对")
    
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
        print(f"  JASPAR过滤: {before} → {after} 条边 (保留 {after/max(before,1)*100:.1f}%)")
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

        print("GLUE:")
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
            print("检测到冒号格式，正在统一为连字符格式...")
            # 2. 将冒号替换为连字符
            # 替换前：chr1:100-200 -> 替换后：chr1-100-200
            linger_df['Peak'] = linger_df['Peak'].str.replace(':', '-', regex=False)
            
        else:
            print("格式已是连字符（chr1-100-200），无需转换。")
        
        print("LINGER:")
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
            print("检测到冒号格式，正在统一为连字符格式...")
            # 2. 将冒号替换为连字符
            # 替换前：chr1:100-200 -> 替换后：chr1-100-200
            granie_df['Peak'] = granie_df['Peak'].str.replace(':', '-', regex=False)
            
        else:
            print("格式已是连字符（chr1-100-200），无需转换。")
            
        print("GRaNIE:")
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
            print("检测到冒号格式，正在统一为连字符格式...")
            # 2. 将冒号替换为连字符
            # 替换前：chr1:100-200 -> 替换后：chr1-100-200
            pando_df['Peak'] = pando_df['Peak'].str.replace(':', '-', regex=False)
            
        else:
            print("格式已是连字符（chr1-100-200），无需转换。")
        
        print("Pando:")
        pando_df = apply_jaspar_filter(pando_df)                  # ← 新增
        
        pando_results = calculate_tf_metrics(pando_df, unibind_tf_peak_grn, f_beta = f_score)
        
        pando_results['Method'] = 'Pando'
        
        method_result.update({"Pando": pando_results})
    
    return method_result
        
    


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_top_n_comparison(method_dict, output_file, top_n=200, score_col="Precision", palette="Set2"):
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
            
        # 2. 排序并提取前 N 个
        # 深拷贝一份以防修改原始数据，并按降序排序
        temp_df = df.sort_values(by=score_col, ascending=False).head(top_n).copy()
        
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 计算排序逻辑（按中位数从高到低排列箱线图）
    order = df_plot.groupby('Method')[score_col].median().sort_values(ascending=False).index
    
    # --- 图 1: Top N 分布对比 (Boxplot) ---
    sns.boxplot(data=df_plot, x='Method', y=score_col, 
                order=order, palette=palette, width=0.6, ax=ax1)
    sns.stripplot(data=df_plot, x='Method', y=score_col, 
                  order=order, color='black', size=2, alpha=0.3, jitter=True, ax=ax1)
    
    ax1.set_title(f"Top {top_n} Genes: {score_col} Distribution", fontsize=14)
    ax1.set_ylabel(score_col)
    ax1.set_xlabel("")
    ax1.tick_params(axis='x', rotation=45)

    # --- 图 2: 性能衰减曲线 (Line Plot) ---
    sns.lineplot(data=df_plot, x='Rank', y=score_col, 
                 hue='Method', hue_order=order, palette=palette, lw=2.5, ax=ax2)
    
    ax2.set_title(f"Top {top_n} Genes: {score_col} Decay Curve", fontsize=14)
    ax2.set_xlabel("Gene Rank (Sorted by score)")
    ax2.set_ylabel(score_col)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    sns.despine()
    plt.savefig(output_file, dpi=1200, bbox_inches='tight')
    
    return df_plot # 返回处理后的数据，方便后续查阅





def analysis_tf_region_overall(tf_region_result, output_path, method_colors, beta=1.0):
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
    for method_name, df in tf_region_result.items():
        df = df.dropna(subset=["Precision", "Recall", "F1"])

        # ── Macro 平均 ──────────────────────────────────────────────────────
        macro_prec = df["Precision"].mean()
        macro_rec  = df["Recall"].mean()
        denom_macro = (1 + beta**2) * macro_prec + beta**2 * macro_rec
        macro_f = ((1 + beta**2) * macro_prec * macro_rec / denom_macro
                if denom_macro > 0 else 0.0)

        # ── Micro 平均（需要原始 TP/FP/FN；若列不存在则跳过）──────────────
        if all(c in df.columns for c in ["TP", "FP", "FN"]):
            total_tp = df["TP"].sum()
            total_fp = df["FP"].sum()
            total_fn = df["FN"].sum()
            micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            denom_micro = (1 + beta**2) * micro_prec + beta**2 * micro_rec
            micro_f = ((1 + beta**2) * micro_prec * micro_rec / denom_micro
                    if denom_micro > 0 else 0.0)
        else:
            # TP/FP/FN 列不存在时，以 Macro 代替（保守估计）
            micro_prec, micro_rec, micro_f = macro_prec, macro_rec, macro_f
            print(f"  [警告] {method_name} 缺少 TP/FP/FN 列，Micro 平均以 Macro 代替。")

        rows.append({
            "Method":         method_name,
            "Macro_Precision": macro_prec,
            "Macro_Recall":    macro_rec,
            "Macro_F":         macro_f,
            "Micro_Precision": micro_prec,
            "Micro_Recall":    micro_rec,
            "Micro_F":         micro_f,
            "N_TF":            len(df),
        })
        print(f"[TF-Region Overall] {method_name} | "
            f"Macro Prec={macro_prec:.4f} F={macro_f:.4f} | "
            f"Micro Prec={micro_prec:.4f} F={micro_f:.4f} | N={len(df)}")

    overall_df = pd.DataFrame(rows)
    label_f = f"F{beta}" if beta != 1.0 else "F1"

    # ── 绘图：4 指标横向对比（2×2 子图）─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid")

    metrics = [
        ("Macro_Precision", f"Macro Precision", axes[0, 0]),
        (f"Macro_F",         f"Macro {label_f} Score", axes[0, 1]),
        ("Micro_Precision", f"Micro Precision", axes[1, 0]),
        (f"Micro_F",         f"Micro {label_f} Score", axes[1, 1]),
    ]

    for col, title, ax in metrics:
        sorted_df = overall_df.sort_values(col, ascending=False)
        bp = sns.barplot(data=sorted_df, x="Method", y=col, hue="Method", palette=method_colors, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(title, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=35)
        _annotate_bars(bp, fmt='.4f')

    plt.suptitle("Benchmark of TF-Region Overall Precision & F Score", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path + f"Benchmark of TF-Region Overall Precision and {label_f}.png",
                dpi=1200, bbox_inches="tight")
    # plt.close()
    print(f"✓ 已保存：TF-Region Overall Precision and {label_f}")

    # ── 补充：Macro Precision vs Macro F（分组柱状图）────────────────────────
    plot_df = overall_df.melt(
        id_vars="Method",
        value_vars=["Macro_Precision", f"Macro_F"],
        var_name="Metric", value_name="Score"
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {"Macro_Precision": "Macro Precision", "Macro_F": f"Macro {label_f}"}
    )

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=plot_df, x="Method", y="Score",
                    hue="Method", palette=method_colors)
    plt.title("TF-Region: Macro Precision vs F Score (per Method)", fontsize=13)
    plt.ylabel("Score", fontsize=11)
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path + "Benchmark of TF-Region Macro Precision vs F Score.png",
                dpi=1200, bbox_inches="tight")
    # plt.close()
    print("✓ 已保存：TF-Region Macro Precision vs F Score")

    return overall_df


########################################################################################################
#
# Region-Gene analysis
#
########################################################################################################

from analysis.assess_region_gene import evaluate_per_gene_correlation, evaluate_scenic_plus_correlation, region_gene_evaluate

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
            print("检测到冒号格式，正在统一为连字符格式...")
            # 2. 将冒号替换为连字符
            # 替换前：chr1:100-200 -> 替换后：chr1-100-200
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
            print("检测到冒号格式，正在统一为连字符格式...")
            # 2. 将冒号替换为连字符
            # 替换前：chr1:100-200 -> 替换后：chr1-100-200
            linger_grn['Peak'] = linger_grn['Peak'].str.replace(':', '-', regex=False)
            
        else:
            print("格式已是连字符（chr1-100-200），无需转换。")

        linger_grn.drop_duplicates(subset=["Peak", "Gene"], inplace=True)

        linger_gene_peak_df = linger_grn.copy()
        linger_gene_peak_df["predict"] = (linger_gene_peak_df["reg"]<0.05).astype(int)
        linger_gene_peak_df["value"] = -np.log10(linger_gene_peak_df["reg"])

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
            print("检测到冒号格式，正在统一为连字符格式...")
            # 2. 将冒号替换为连字符
            # 替换前：chr1:100-200 -> 替换后：chr1-100-200
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


def plot_top_n_methods_comparison(corr_df, output_file, method_colors, top_n=100, compare_method = 'Spearman_Rho'):
    """
    提取每种方法相关性最好的前 N 个基因并进行比较
    :param corr_df: 包含 [Method, Gene, Spearman_Rho] 的 DataFrame
    :param top_n: 每种方法选取的基因数量
    """
    # 1. 提取每种方法的前 N 个基因
    top_n_df = corr_df.groupby('Method').apply(
        lambda x: x.sort_values(compare_method, ascending=False).head(top_n)
    ).reset_index(drop=True)

    # 2. 为绘图计算排名（Rank 1 到 N）
    top_n_df['Rank'] = top_n_df.groupby('Method')[compare_method].rank(
        ascending=False, method='first'
    ).astype(int)

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 图 1: Top N 基因的相关性分布 (Boxplot) ---
    # 这展示了各方法“天花板”的平均水平
    method_order = top_n_df.groupby('Method')[compare_method].median().sort_values(ascending=False).index
    
    sns.boxplot(
        data=top_n_df, x='Method', y=compare_method, 
        order=method_order, palette=method_colors, ax=ax1
    )
    sns.stripplot(
        data=top_n_df, 
        x='Method', 
        y=compare_method, 
        order=method_order, 
        color='black', 
        alpha=0.3, 
        size=2, 
        jitter=True,
        ax=ax1
        )
    ax1.set_title(f'Distribution of Spearman Rho (Top {top_n} Genes)')
    ax1.set_ylabel('Spearman Rho')
    ax1.tick_params(axis='x', rotation=45)

    # --- 图 2: Top N 基因的相关性衰减曲线 (Line Plot) ---
    # 这展示了随着排名增加，高质量预测消失的速度
    sns.lineplot(
        data=top_n_df, x='Rank', y=compare_method, 
        hue='Method', hue_order=method_order, lw=2.5, ax=ax2, palette=method_colors)

    ax2.set_title(f'Performance Profile: Top {top_n} Genes Rank vs. Rho')
    ax2.set_xlabel('Gene Rank (Sorted by Rho)')
    ax2.set_ylabel('Spearman Rho')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=1200, bbox_inches='tight')


def plot_recovery_curves(corr_df, output_file, method_colors, compared_method='Spearman_Rho'):
    """
    恢复曲线：X轴是相关性阈值，Y轴是该方法在该质量下能覆盖的基因总数
    """
    thresholds = np.linspace(0, 0.9, 100) # 覆盖从低到高的相关性区间
    plot_list = []
    
    for method in corr_df['Method'].unique():
        m_vals = corr_df[corr_df['Method'] == method][compared_method].values
        for thr in thresholds:
            count = np.sum(m_vals > thr)
            plot_list.append({'Method': method, 'Threshold': thr, 'Gene_Count': count})
            
    pdf = pd.DataFrame(plot_list)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=pdf, x='Threshold', y='Gene_Count', hue='Method', palette=method_colors)
    plt.title('Gene Coverage vs Prediction Quality (Spearman Rho)')
    plt.ylabel('Number of Recovered Genes')
    plt.xlabel('Spearman Rho (Consistency with Benchmark)')
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
        print(merged.head())
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

def analysis_region_gene_precision(benchmark_peak_gene_df, method_data, output_path, method_colors, beta=1.0):
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
        print(f"{method_name}:{len(merged)}")
        merged['value']   = merged['value'].fillna(0.0)
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
    label_f   = f"F{beta}" if beta != 1.0 else "F1"

    # ── 绘图 1：4 指标柱状图（2×2 子图）─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid")

    for (col, title), ax in zip(
        [("Precision", "Precision"),
        ("Recall",    "Recall"),
        (f"F_score",  f"{label_f} Score"),
        ("AUPRC",     "AUPRC")],
        axes.flatten()
    ):
        sorted_df = result_df.sort_values(col, ascending=False)
        bp = sns.barplot(data=sorted_df, x="Method", y=col, palette=method_colors, ax=ax)
        ax.set_title(f"Region-Gene {title}", fontsize=12)
        ax.set_ylabel(title, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=35)
        _annotate_bars(bp, fmt='.4f')

    plt.suptitle("Benchmark of Region-Gene Precision, Recall, F Score & AUPRC",
                fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path + f"Benchmark of Region-Gene Precision Recall {label_f} AUPRC.png",
                dpi=1200, bbox_inches="tight")
    # plt.close()
    print(f"✓ 已保存：Region-Gene Precision Recall {label_f} AUPRC")

    # ── 绘图 2：PR 曲线（所有方法叠加）──────────────────────────────────────
    if pr_curve_data:
        plt.figure(figsize=(8, 6))
        for method_name, (rec_c, prec_c, auprc_val) in pr_curve_data.items():
            plt.plot(rec_c, prec_c, color = method_colors.get(method_name, 'blue'),
                    label=f"{method_name} (AUPRC={auprc_val:.3f})", linewidth=2)

        # 随机基线（正例比例）
        baseline = result_df["N_true"].iloc[0] / len(benchmark_peak_gene_df) \
            if len(result_df) > 0 else 0.5
        plt.axhline(y=baseline, color='gray', linestyle='--',
                    linewidth=1.2, label=f"Random (P={baseline:.3f})")

        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Benchmark of Region-Gene PR Curve", fontsize=14)
        plt.legend(loc="upper right", fontsize=9)
        plt.grid(True, alpha=0.3)
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_path + "Benchmark of Region-Gene PR Curve.png",
                    dpi=1200, bbox_inches="tight")
        # plt.close()
        print("✓ 已保存：Region-Gene PR Curve")

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
    plt.ylabel("Score", fontsize=11)
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path + f"Benchmark of Region-Gene Precision vs {label_f}.png",
                dpi=1200, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存：Region-Gene Precision vs {label_f}")

    return result_df



########################################################################################################
#
# TF-Gene analysis
#
########################################################################################################
from pdata.data_preprocess import filter_jaspar_tf

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

    print("*"*50)
    print(f"Peak-Gene: {peak_gene_df['Peak'].nunique()}, {peak_gene_df['Gene'].nunique()}, edge:{len(peak_gene_df)}")
    print("*"*50)

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

    # 查看结果
    print("*"*50)
    print(f"Dygmamba TF-Peak: {dygmamba_tf_peak_df['TF'].nunique()}, {dygmamba_tf_peak_df['Peak'].nunique()},\
        edge: {len(dygmamba_tf_peak_df)}")
    print(f"Peak-Gene: {peak_gene_df['Peak'].nunique()}, {peak_gene_df['Gene'].nunique()}, edge:{len(peak_gene_df)}")
    print(f"TF-Gene: {tf_gene_grn['TF'].nunique()}, {tf_gene_grn['Gene'].nunique()}, edge: {len(tf_gene_grn)}")

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

    print("*"*50)
    print(f"TF-Gene: {tf_gene_grn['TF'].nunique()}, {tf_gene_grn['Gene'].nunique()}, edge: {len(tf_gene_grn)}")
    print(f"avg active TF-Gene: {avg_active_tf_gene_grn['TF'].nunique()}, {avg_active_tf_gene_grn['Gene'].nunique()},\
        edge: {len(avg_active_tf_gene_grn)}")
    print(f"avg global TF-Gene: {avg_global_tf_gene_grn['TF'].nunique()}, {avg_global_tf_gene_grn['Gene'].nunique()},\
        edge: {len(avg_global_tf_gene_grn)}")
    print("*"*50)
    
    return tf_gene_grn, avg_active_tf_gene_grn, avg_global_tf_gene_grn








from analysis.assess_tf_gene import evaluate_predictability
from pdata.benchmark_data import glue_read_ctx_grn


def analysis_tf_gene_data(data_path, method_list, flag_corr, flag_precision, beta = 1):
    
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
            
            correlation_result_dict.update({"Active": active_result, "Global": global_result})
        
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

            glue_df.rename(columns={'Gene':'Target'},inplace = True)
            glue_result, glue_trained_models = evaluate_predictability(matrix_df, glue_df)
            
            correlation_result_dict.update({"GLUE": glue_result})
        
        if flag_precision:
            glue_data_path = data_path + "data_glue/"
            glue_df2 = glue_read_full_grn(glue_data_path + 'pruned_grn.csv')
            glue_df2 = glue_df2[['TF', 'Target', 'Importance', 'MotifID', 'NES', 'AUC', 'RankAtMax', 'MotifSimilarityQvalue', 'Annotation']]

            glue_grn2 = glue_df2[["TF","Target","Importance"]].rename(columns = {"Target":"Gene"})
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
            figr_result, figr_trained_models = evaluate_predictability(matrix_df, figr_df)
            
            correlation_result_dict.update({"FigR": figr_result})
        
        if flag_precision:
            figr_grn = figr_grn0[["TF","Target_Gene", "Correlation"]].copy()
            figr_grn.rename(columns={"Target_Gene": "Gene", "Correlation":"score"}, inplace=True)
            figr_grn["score"] = abs(figr_grn["score"])
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


            celloracle_result, celloracle_trained_models = evaluate_predictability(matrix_df, celloracle_df)
            
            correlation_result_dict.update({"CellOracle": celloracle_result})

        if flag_precision:
            celloracle_df2 = celloracle_grn[["TF", "Gene", "coef_abs"]].copy()
            celloracle_df2.rename(columns={"coef_abs": "score"}, inplace=True)
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

            linger_result, linger_trained_models = evaluate_predictability(matrix_df, linger_df)
            
            correlation_result_dict.update({"LINGER": linger_result})
        
        if flag_precision:
            linger_df2= linger_grn.melt(id_vars=['Gene'], var_name='TF', value_name='regulation')
            
            linger_df2.rename(columns={"regulation": "score"}, inplace=True)

            linger_grn2 = linger_df2[["TF", "Gene", "score"]].copy()
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
            granie_result, granie_trained_models = evaluate_predictability(matrix_df, granie_df)
            
            correlation_result_dict.update({"GRaNIE": granie_result})
        
        if flag_precision:
            granie_grn.dropna()
            granie_grn["score"] = granie_grn["TF_peak.r"].abs()* granie_grn['peak_gene.r'].abs()

            granie_grn2 = granie_grn[["TF.name", "gene.name", "score"]].copy()
            granie_grn2.rename(columns={"TF.name": "TF", "gene.name": "Gene"}, inplace=True)
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
            pando_result, pando_trained_models = evaluate_predictability(matrix_df, pando_df)
            
            correlation_result_dict.update({"Pando": pando_result})
        
        if flag_precision:
            pando_grn["predict_label"] = (pando_grn["padj"]<0.04).astype(int)

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
            print(f"{model_name} - Precision: {precision:.4f}, \
                Recall: {recall:.4f}, FPR: {FPR:.4f}, \
                    AUC: {AUC:.4f}, F-beta: {f_beta:.4f}")
            
            per_tf = _compute_per_tf_metrics(benchmark_tf_gene_df, model_df, beta=beta)
            per_tf["Method"] = model_name
            per_precision_result_df.update({model_name: per_tf})

        precision_result_dataframe = pd.DataFrame(result_list)
        
        # per precision
        
        return correlation_result_dict, precision_result_dict, precision_result_dataframe, per_precision_result_df
    
    
    ##########################################################################
    precision_result_dataframe = pd.DataFrame([])
    
    return correlation_result_dict, precision_result_dict, precision_result_dataframe, per_precision_result_df



def analysis_tf_gene_all_correlation(output_path, result_dict, method_colors):
    dfs_to_concat = []
    new_labels_dict = {}
    analys_label = 'Correlation'
    for method_name, df in result_dict.items():
        # 1. 直接在原 DataFrame 上添加/修改 Method 列
        df["Method"] = method_name
        
        # 2. 提取需要的列，并将其追加到列表中
        dfs_to_concat.append(df[[analys_label, 'Method']])
        
        new_labels_dict.update({method_name: f"{method_name}\n(n={len(df)})"})

    # 3. 将列表中的所有 DataFrame 拼接成一个
    df_combined = pd.concat(dfs_to_concat, ignore_index=True)

    plt.figure(figsize=(8, 6))

    # 定义顺序 (防止自动排序乱掉)
    my_order = ['Global', 'Active','GLUE', 'FigR', 'CellOracle', 
                'LINGER', 'GRaNIE', "Pando"]

    # 4. 绘制箱形图
    # Seaborn 会自动识别 'Method' 列中的组别，并独立计算每个组的箱子
    ax = sns.boxplot(data=df_combined, 
                    x='Method', 
                    y=analys_label, 
                    order=my_order, 
                    palette=method_colors,
                    showfliers=False) # 可选：不显示异常值点
    
    sns.stripplot(data=df_combined, x='Method', y=analys_label, 
                    order=my_order, color='black', size=2, alpha=0.3, jitter=True, ax=ax)
        
    # =================================================
    # 关键技巧：修改 X 轴标签，加上 (n=xxx)
    # =================================================
    new_labels = [new_labels_dict.get(label, label) for label in my_order]

    # 应用新标签
    ax.set_xticklabels(new_labels)

    plt.title("Benchmark of TF-Gene Predictability Comparison--Correlation", fontsize=14)
    plt.ylabel(analys_label)
    plt.xlabel("") # 清空 x 轴标题，因为标签里已经写了

    sns.despine() # 去掉边框
    plt.savefig(output_path + 'Benchmark of TF-Gene Predictability Comparison--Correlation.png', 
                dpi=1200, bbox_inches='tight')
    

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_top_n_tf_comparison(result_dict, output_path, method_colors, top_n=50, 
                             score_col='Correlation', my_order=None):
    """
    对比不同方法中表现最好的前 N 个 TF
    
    :param result_dict: 字典, key为方法名, value为对应的结果DataFrame
    :param top_n: 取前多少个TF
    :param score_col: 用于排序的列名，通常是 'Correlation' 或 'Precision'
    :param my_order: 指定绘图时的顺序
    """
    processed_list = []
    new_labels = []
    
    # 如果没指定顺序，默认按字典顺序
    if my_order is None:
        my_order = list(result_dict.keys())
        
    for method in my_order:
        if method not in result_dict:
            continue
            
        df = result_dict[method].copy()
        
        # 1. 核心操作：按得分降序排列，取前 N 个
        top_df = df.sort_values(by=score_col, ascending=False).head(top_n)
        top_df["Method"] = method
        processed_list.append(top_df[[score_col, 'Method']])
        
        # 2. 准备新的标签 (n=实际数量)
        actual_n = len(top_df)
        new_labels.append(f"{method}\n(top {actual_n})")
    
    # 合并数据
    df_combined = pd.concat(processed_list)
    
    # 3. 开始绘图
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="ticks")
    
    # 绘制箱线图
    ax = sns.boxplot(data=df_combined, 
                     x='Method', 
                     y=score_col, 
                     order=my_order, 
                     palette=method_colors,
                     showfliers=False)
    
    # 叠加散点
    sns.stripplot(data=df_combined, 
                  x='Method', 
                  y=score_col, 
                  order=my_order, 
                  color='black', 
                  size=3, 
                  alpha=0.4, 
                  jitter=True, 
                  ax=ax)
    
    # 4. 美化
    ax.set_xticklabels(new_labels)
    plt.title(f"Benchmark of TF-Gene Predictability Comparison--Correlation Top {top_n} TFs: {score_col}", fontsize=14)
    plt.ylabel(score_col)
    plt.xlabel("")
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path + f'Benchmark of TF-Gene Predictability Comparison--Top {top_n} TFs Correlation.png',
            dpi=1200, bbox_inches='tight')
    
    
    
    
    
from analysis.assess import glue_read_full_grn
from data_preprocess import adata_to_dataframe
from analysis.assess import each_confusion_matrix
from analysis.assess import manual_assess

def analysis_tf_gene_all_precision(result_dataframe, output_path, method_colors):

    #######################################
    # f-scpore figure plot

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(5, 6))

    # 3. 绘制条形图
    # 按照 F-beta 从高到低排序会让图表更有说服力
    ax = sns.barplot(x='Model', y='F-beta', 
                     data=result_dataframe.sort_values('F-beta', ascending=False), 
                     palette=method_colors)
    
    # 4. 添加细节
    plt.title('Benchmark of TF-Gene Comparison of F-beta Score', fontsize=15)
    plt.xlabel('Model Name', fontsize=12)
    plt.ylabel('F-beta Score', fontsize=12)
    plt.xticks(rotation=45) # 防止模型名称太长重叠

    # 在柱状图上方标注具体数值
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    plt.tight_layout()
    plt.savefig(output_path + 'Benchmark of TF-Gene Comparison of F-beta Score.png', 
                dpi=1200, bbox_inches='tight')
    
    
    ############################
    # precision igure plot

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(5, 6))

    ax = sns.barplot(x='Model', y='Precision', 
                     data=result_dataframe.sort_values('Precision', ascending=False), 
                     palette=method_colors)

    plt.title('Benchmark of TF-Gene Comparison of Precision', fontsize=15)
    plt.xlabel('Model Name', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.xticks(rotation=45) 

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

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


def _annotate_bars(ax, fmt='.4f'):
    """在柱状图每个柱子顶端标注数值。"""
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), fmt),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 9), textcoords='offset points', fontsize=8
        )


def analysis_tf_gene_per_tf_precision(method_grn_dict, output_path, method_colors, beta=1.0):
    """
    计算每个 TF 粒度的 Precision 与 F-score，绘制箱线图，与各方法对比。

    金标准：JASPAR TF-Gene motif 网络（process/tf_gene_network.h5ad）。
    阈值逻辑与 analysis_tf_gene_all_precision() 保持一致（25th 百分位）。

    Parameters
    ----------
    data_path   : str   数据根目录
    output_path : str   图片输出目录
    method_list : list  参与对比的方法名列表
    beta        : float F-beta 的 beta（默认 1.0 = F1）
    """

    # ── 逐方法计算每 TF 的 Precision / F_score ──────────────────────────────
    all_prec_dfs  = []
    all_fscore_dfs = []
    all_rec_dfs   = []
    
    for method_name, per_tf in method_grn_dict.items():
        all_prec_dfs.append(per_tf[["TF", "Precision", "Method"]])
        all_fscore_dfs.append(per_tf[["TF", "F_score", "Method"]])
        all_rec_dfs.append(per_tf[["TF", "Recall", "Method"]])

    prec_combined   = pd.concat(all_prec_dfs,   ignore_index=True)
    fscore_combined = pd.concat(all_fscore_dfs, ignore_index=True)
    rec_combined    = pd.concat(all_rec_dfs,    ignore_index=True)

    # ── 绘图：Precision 箱线图 ────────────────────────────────────────────────
    method_order_prec = (prec_combined.groupby("Method")["Precision"]
                        .median().sort_values(ascending=False).index.tolist())

    sns.set_theme(style="ticks")
    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(data=prec_combined, x="Method", y="Precision",
                    order=method_order_prec, palette = method_colors, width=0.55, showfliers=False)
    
    sns.stripplot(data=prec_combined, x="Method", y="Precision",
                order=method_order_prec, color="black", size=2.5, alpha=0.35,
                jitter=True, ax=ax)
    plt.title("Benchmark of TF-Gene Per-TF Precision Comparison", fontsize=14)
    plt.ylabel("Precision (per TF)", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path + "Benchmark of TF-Gene Per-TF Precision Boxplot.png",
                dpi=1200, bbox_inches="tight")
    # plt.close()
    print("✓ 已保存：TF-Gene Per-TF Precision Boxplot")

    # ── 绘图：F_score 箱线图 ─────────────────────────────────────────────────
    method_order_f = (fscore_combined.groupby("Method")["F_score"]
                    .median().sort_values(ascending=False).index.tolist())

    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(data=fscore_combined, x="Method", y="F_score",
                    order=method_order_f, palette = method_colors, width=0.55, showfliers=False)
    
    sns.stripplot(data=fscore_combined, x="Method", y="F_score",
                order=method_order_f, color="black", size=2.5, alpha=0.35,
                jitter=True, ax=ax)
    label_str = f"F{beta}" if beta != 1.0 else "F1"
    plt.title(f"Benchmark of TF-Gene Per-TF {label_str} Score Comparison", fontsize=14)
    plt.ylabel(f"{label_str} Score (per TF)", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path + f"Benchmark of TF-Gene Per-TF {label_str} Boxplot.png",
                dpi=1200, bbox_inches="tight")
    # plt.close()
    print(f"✓ 已保存：TF-Gene Per-TF {label_str} Boxplot")

    return prec_combined, fscore_combined, rec_combined






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

    print("\n[TF-Gene Correlation Summary]")
    print(summary_df.to_string(index=False))

    # ── 柱状图（Median Correlation） ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.set_theme(style="whitegrid")

    # 左：Median
    ax0 = sns.barplot(data=summary_df, x="Method", y="Median_Corr",
                    palette= method_colors, ax=axes[0])
    
    axes[0].set_title("TF-Gene Predictability: Median Correlation", fontsize=13)
    axes[0].set_ylabel("Median Pearson Correlation", fontsize=11)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=35)
    _annotate_bars(ax0)

    # 右：Mean ± Std（误差棒）
    ax1 = sns.barplot(data=summary_df, x="Method", y="Mean_Corr",
                    palette=method_colors, ax=axes[1])
    # 手动添加误差棒
    for i, row in enumerate(summary_df.itertuples()):
        axes[1].errorbar(i, row.Mean_Corr, yerr=row.Std_Corr,
                        fmt='none', color='black', capsize=4, linewidth=1.2)
    axes[1].set_title("TF-Gene Predictability: Mean ± Std Correlation", fontsize=13)
    axes[1].set_ylabel("Mean Pearson Correlation", fontsize=11)
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=35)

    plt.suptitle("Benchmark of TF-Gene Overall Correlation Summary", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path + "Benchmark of TF-Gene Overall Correlation Summary.png",
                dpi=1200, bbox_inches="tight")
    # plt.close()
    print("✓ 已保存：TF-Gene Overall Correlation Summary")

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
    sns.set_theme(style="ticks")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── 左面板：Top N TF 分布箱线图 ──────────────────────────────────────────
    sns.boxplot(
        data      = df_plot,
        x         = "Method",
        y         = score_col,
        order     = order,
        palette   = palette,
        width     = 0.55,
        showfliers= False,
        ax        = ax1,
    )
    sns.stripplot(
        data   = df_plot,
        x      = "Method",
        y      = score_col,
        order  = order,
        color  = "black",
        size   = 2.5,
        alpha  = 0.35,
        jitter = True,
        ax     = ax1,
    )
    ax1.set_xticklabels(x_labels, fontsize=10)
    ax1.set_title(
        f"TF-Gene Per-TF {score_col}: Top {top_n} TF Distribution",
        fontsize=13,
    )
    ax1.set_ylabel(score_col, fontsize=11)
    ax1.set_xlabel("")
    sns.despine(ax=ax1)

    # ── 右面板：性能衰减折线图 ────────────────────────────────────────────────
    sns.lineplot(
        data      = df_plot,
        x         = "Rank",
        y         = score_col,
        hue       = "Method",
        hue_order = order,
        palette   = palette,
        lw        = 2.2,
        ax        = ax2,
    )
    ax2.set_title(
        f"TF-Gene Per-TF {score_col}: Top {top_n} TF Decay Curve",
        fontsize=13,
    )
    ax2.set_xlabel("TF Rank (sorted by score, 1 = best)", fontsize=11)
    ax2.set_ylabel(score_col, fontsize=11)
    ax2.grid(True, alpha=0.3)
    sns.despine(ax=ax2)

    plt.suptitle(
        f"Benchmark of TF-Gene Per-TF {score_col} — Top {top_n} TFs",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    # ── 5. 保存 ──────────────────────────────────────────────────────────────
    fname = (
        f"Benchmark of TF-Gene Per-TF {score_col} Top{top_n}"
        + (f" {tag}" if tag else "")
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


