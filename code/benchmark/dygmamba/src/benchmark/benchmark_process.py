import sys
import os
import pickle
import pandas as pd
import anndata as ad
import numpy as np
import seaborn as sns
import networkx as nx
from tqdm import tqdm
import scipy.sparse
from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

sys.path.append('/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src/')

from analysis.assess_tf_recovery import calculate_recovery_metrics
from analysis.assess_tf_gene import evaluate_predictability
from analysis.assess_tf_region import calculate_tf_metrics

from pdata.benchmark_data import glue_read_ctx_grn

from pdata.data_preprocess import filter_jaspar_tf, adata_to_dataframe
from pdata.data_preprocess import build_tf_peak_network
from pdata.data_read import read_unibind_file
from analysis.assess_region_gene import evaluate_per_gene_correlation, evaluate_scenic_plus_correlation
from analysis.assess_region_gene import region_gene_evaluate
import dill


##############################################################################
# TF recovery
#
##############################################################################
def TF_Recovery_benchmark(data_path):

    output_path = data_path + "benchmark/"

    os.makedirs(output_path, exist_ok=True )

    ###########################################################
    # Unibind data processing
    benchmark_df = pd.read_pickle(output_path + "count_region_df.pkl")

    ground_truth_ranked = benchmark_df.sort_values(by="PeakCount", ascending=False)["TF"].tolist()

    ##########################################################################
    # DygMamba TF recovery
    dyg_tf_peak_grn = ad.read_h5ad(data_path + "process/tf_peak_network.h5ad")

    dyg_tfs = set(dyg_tf_peak_grn.var_names)

    dyg_x, dyg_y, dyg_raw_auc, dyg_norm_auc = calculate_recovery_metrics(ground_truth_ranked, dyg_tfs, top_n=40)

    print("*"*20 + " DYGMAMBA " + "*"*20)
    print(f"Top 40 AUC (Raw): {dyg_raw_auc:.2f}")
    print(f"Top 40 AUC (Normalized): {dyg_norm_auc:.4f}")

    plt.figure(figsize=(6, 6))

    # 绘制我们的方法的曲线
    plt.plot(dyg_x, dyg_y, label=f'{"DyGMAMBA"} (AUC={dyg_norm_auc:.2f})', color='dodgerblue', linewidth=2, marker='o', markersize=4)

    ##########################################################################
    # GLUE TF recovery

    print("*"*20 + " GLUE " + "*"*20)
    
    glue_grn = glue_read_ctx_grn(data_path + "data_glue/pruned_grn.csv")

    df_edges = nx.to_pandas_edgelist(glue_grn, source="TF", target="Gene")

    glue_tfs = set(df_edges["TF"])

    glue_x, glue_y, glue_raw_auc, glue_norm_auc = calculate_recovery_metrics(ground_truth_ranked, glue_tfs, top_n=40)

    plt.plot(glue_x, glue_y, label=f'{"GLUE"} (AUC={glue_norm_auc:.2f})', color='teal', linewidth=2, marker='o', markersize=4)

    ##########################################################################
    # CellOracle TF recovery

    print("*"*20 + " CellOracle " + "*"*20)
    
    celloracle_grn = pd.read_csv(data_path + "data_celloracle/celloracle_results/grn_df_" + "cluster0" + ".csv")

    celloracle_grn.rename(columns={"source": "TF", "target": "Gene"}, inplace= True)

    oracle_tf_gene = celloracle_grn[["TF", "Gene", "-logp"]].copy()

    oracle_tfs = set(oracle_tf_gene["TF"])

    oracle_x, oracle_y, oracle_raw_auc, oracle_norm_auc = calculate_recovery_metrics(ground_truth_ranked, oracle_tfs, top_n=40)

    plt.plot(oracle_x, oracle_y, label=f'{"CellOracle"} (AUC={oracle_norm_auc:.2f})', color='lightpink', linewidth=2, marker='o', markersize=4)


    ##########################################################################
    # FigR TF recovery

    figr_grn = pd.read_csv(data_path + "data_FigR/TF_Gene_Network.csv")

    figr_tfs = set(figr_grn["TF"])

    figr_x, figr_y, figr_raw_auc, figr_norm_auc = calculate_recovery_metrics(ground_truth_ranked, figr_tfs, top_n=40)

    plt.plot(figr_x, figr_y, label=f'{"FigR"} (AUC={figr_norm_auc:.2f})', 
            color='darkorange', linewidth=2, marker='o', markersize=4)

    ##########################################################################
    # Linger TF recovery

    linger_grn = pd.read_csv(data_path + "data_linger/train/cell_population_TF_RE_binding.txt", sep='\t')

    linger_tfs = set(linger_grn.columns[1:])

    linger_x, linger_y, linger_raw_auc, linger_norm_auc = calculate_recovery_metrics(ground_truth_ranked, linger_tfs, top_n=40)

    plt.plot(linger_x, linger_y, label=f'{"LINGER"} (AUC={linger_norm_auc:.2f})', 
            color='plum', linewidth=2, marker='o', markersize=4)

    ##########################################################################
    # GRaNIE TF recovery

    granie_grn = pd.read_csv(data_path + "GRaNIE_output/GRaNIE_tf_gene_Links.csv")

    granie_tfs = set(granie_grn["TF.name"])

    granie_x, granie_y, granie_raw_auc, granie_norm_auc = calculate_recovery_metrics(ground_truth_ranked, granie_tfs, top_n=40)

    plt.plot(granie_x, granie_y, label=f'{"GRaNIE"} (AUC={granie_norm_auc:.2f})', 
            color='sienna', linewidth=2, marker='o', markersize=4)
    
    ##########################################################################
    # Pando TF recovery

    pando_grn = pd.read_csv(data_path + "data_pando/tf_gene_network.csv")

    pando_tfs = set(pando_grn["TF"])

    pando_x, pando_y, pando_raw_auc, pando_norm_auc = calculate_recovery_metrics(ground_truth_ranked, pando_tfs, top_n=40)

    plt.plot(pando_x, pando_y, label=f'{"Pando"} (AUC={pando_norm_auc:.2f})', 
            color='mediumpurple', linewidth=2, marker='o', markersize=4)
    
    ############################################################################
    plt.title('TF Recovery Curve', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 40)
    plt.ylim(0, 40) # 或者是实际恢复的最大值

    plt.savefig(output_path + 'Benchmark_TF_Recovery_Curve.png', dpi=300, bbox_inches='tight')



def TF_Recovery_tau_benchmark():
    
    data_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/"

    output_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/benchmark/"

    os.makedirs(output_path, exist_ok=True )

    ###########################################################
    # Unibind data processing
    benchmark_df = pd.read_pickle(output_path + "count_region_df.pkl")

    ground_truth_ranked = benchmark_df.sort_values(by="PeakCount", ascending=False)["TF"].tolist()
    
    
##########################################################################
#
#
##########################################################################


def TF_Gene_Benchmark(data_path):
    
    output_path = data_path + "benchmark/"
    os.makedirs(output_path, exist_ok=True )

    adata_atac = ad.read_h5ad(data_path + "process/atac_processed.h5ad")
    adata_rna = ad.read_h5ad(data_path + "process/rna_processed.h5ad")
    matrix_df = pd.DataFrame(adata_rna.X.toarray(), index=adata_rna.obs_names,
                                columns=adata_rna.var_names)

    ##########################################################################
    # dyg TF-Gene predictability
    
    dyg_result_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/data_dyg/"
    avg_active_tf_gene_grn = pd.read_pickle(dyg_result_path + "average_active_tf_gene_grn.pkl")
    avg_global_tf_gene_grn = pd.read_pickle(dyg_result_path + "average_global_tf_gene_grn.pkl")

    active_grn = avg_active_tf_gene_grn[["TF","Gene"]].copy()
    active_grn.rename(columns={'Gene':'Target'},inplace = True)

    active_result, trained_models = evaluate_predictability(matrix_df, active_grn)


    global_grn = avg_global_tf_gene_grn[["TF","Gene"]].copy()
    global_grn.rename(columns={'Gene':'Target'},inplace = True)

    global_result, global_trained_models = evaluate_predictability(matrix_df, global_grn)

    ##########################################################################
    # GLUE TF-Gene predictability

    glue_grn = glue_read_ctx_grn(data_path + "data_glue/pruned_grn.csv")
    glue_df = nx.to_pandas_edgelist(glue_grn, source="TF", target="Gene")

    glue_df.rename(columns={'Gene':'Target'},inplace = True)
    glue_result, glue_trained_models = evaluate_predictability(matrix_df, glue_df)

    ##########################################################################
    # FigR TF-Gene predictability

    figr_grn = pd.read_csv(data_path + "data_FigR/TF_Gene_Network.csv")
    figr_df = figr_grn[["TF","Target_Gene"]].copy()
    figr_df.rename(columns={'Target_Gene':'Target'},inplace = True)

    figr_result, figr_trained_models = evaluate_predictability(matrix_df, figr_df)

    ##########################################################################
    # Celloracle TF-Gene predictability

    celloracle_grn = pd.read_csv(data_path + "data_celloracle/celloracle_results/grn_df_" + "cluster0" + ".csv")

    celloracle_grn.rename(columns={"source": "TF", "target": "Gene"}, inplace= True)

    celloracle_df = celloracle_grn[["TF", "Gene"]].copy()
    celloracle_df.rename(columns={'Gene':'Target'},inplace = True)


    celloracle_result, celloracle_trained_models = evaluate_predictability(matrix_df, celloracle_df)


    ##########################################################################
    # LINGER TF-Gene predictability

    linger_grn = pd.read_csv(data_path + "data_linger/train/cell_population_trans_regulatory.txt", sep='\t')
    linger_grn = linger_grn.rename(columns={'Unnamed: 0': 'Gene'})
    linger_df= linger_grn.melt(id_vars=['Gene'], var_name='TF', value_name='regulation')

    linger_df = linger_df[["TF", "Gene"]].copy()
    linger_df.rename(columns={'Gene':'Target'},inplace = True)

    linger_result, linger_trained_models = evaluate_predictability(matrix_df, linger_df)

    ##########################################################################
    # GRaNIE TF-Gene predictability

    granie_grn = pd.read_csv(data_path + "GRaNIE_output/GRaNIE_tf_gene_Links.csv")
    granie_df = granie_grn[["TF.name", "gene.name"]].copy()
    granie_df.rename(columns={"TF.name": "TF", "gene.name": "Target"}, inplace=True)
    granie_result, granie_trained_models = evaluate_predictability(matrix_df, granie_df)

    ##########################################################################
    # Pando TF-Gene predictability

    pando_grn = pd.read_csv(data_path + "data_pando/tf_gene_network.csv")

    pando_df = pando_grn[["TF", "Gene"]].copy()
    pando_df.rename(columns={"Gene": "Target"}, inplace=True)
    pando_result, pando_trained_models = evaluate_predictability(matrix_df, pando_df)
    
    ##########################################################################
    #
    #
    ##########################################################################

    global_result["Method"] = "Global"
    active_result["Method"] = "Active"
    glue_result["Method"] = "GLUE"
    figr_result["Method"] = "FigR"
    celloracle_result["Method"] = "CellOracle"
    linger_result["Method"] = "LINGER"
    granie_result["Method"] = "GRaNIE"
    pando_result["Method"] = "Pando"
    df_combined = pd.concat([global_result[['Correlation', 'Method']], 
                            active_result[['Correlation', 'Method']],
                            glue_result[['Correlation', 'Method']],
                            figr_result[['Correlation', 'Method']],
                            celloracle_result[['Correlation', 'Method']],
                            linger_result[['Correlation', 'Method']],
                            granie_result[['Correlation', 'Method']],
                            pando_result[['Correlation', 'Method']]])

    plt.figure(figsize=(6, 5))

    # 定义顺序 (防止自动排序乱掉)
    my_order = ['Global', 'Active','GLUE', 'FigR', 'CellOracle', 
                'LINGER', 'GRaNIE', "Pando"]

    # 4. 绘制箱形图
    # Seaborn 会自动识别 'Method' 列中的组别，并独立计算每个组的箱子
    ax = sns.boxplot(data=df_combined, 
                    x='Method', 
                    y='Correlation', 
                    order=my_order, 
                    palette="Set2",
                    showfliers=False) # 可选：不显示异常值点

    # =================================================
    # 关键技巧：修改 X 轴标签，加上 (n=xxx)
    # =================================================
    # 计算每组的数量
    n_A = len(global_result)
    n_B = len(active_result)

    # 创建新的标签列表
    new_labels = [f"Global \n(n={n_A})", f"Active \n(n={n_B})",
                  f"GLUE \n(n={len(glue_result)})", f"FigR \n(n={len(figr_result)})", 
                  f"CellOracle \n(n={len(celloracle_result)})", f"LINGER \n(n={len(linger_result)})", 
                  f"GRaNIE \n(n={len(granie_result)})", f"Pando \n(n={len(pando_result)})"]

    # 应用新标签
    ax.set_xticklabels(new_labels)

    # 5. 美化
    plt.title("TF-Gene Predictability Comparison", fontsize=14)
    plt.ylabel("Correlation")
    plt.xlabel("") # 清空 x 轴标题，因为标签里已经写了

    sns.despine() # 去掉边框
    
    plt.savefig(output_path + "Benchmark_TF_Gene_boxplot_comparison.png", dpi=300, bbox_inches='tight')


    
##########################################################################
#
#
##########################################################################


def TF_Region_Benchmark(data_path):

    output_path = data_path + "benchmark/"

    unibind_df_file = data_path + "benchmark/unibind_df.pkl"

    os.makedirs(output_path, exist_ok=True )

    adata_atac = ad.read_h5ad(data_path + "process/atac_processed.h5ad")

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

    ####################################################################
    # dyg TF-Region
    jaspar_tf_region_file = data_path + "process/jaspar_data_processed.h5ad"
    jaspar_data = ad.read_h5ad(jaspar_tf_region_file)
    adata_region_tf = filter_jaspar_tf(jaspar_data)

    coo_matrix = adata_region_tf.X.tocoo()
    dyg_tf_peak_df = pd.DataFrame({
        'Peak': adata_region_tf.obs_names[coo_matrix.row],
        'TF': adata_region_tf.var_names[coo_matrix.col],
        'value': coo_matrix.data
    })
    dyg_tf_peak_df = dyg_tf_peak_df[dyg_tf_peak_df["Peak"].isin(set(adata_atac.var_names))]

    dyg_results = calculate_tf_metrics(dyg_tf_peak_df, unibind_tf_peak_grn)

    print("*"*20 + " DYGMAMBA " + "*"*20)
    print(dyg_results)

    # dyg_results.to_csv(output_path + "tf_region_jaspar_unibind_results.csv", index=False)
    
    
    ####################################################################
    # GLUE TF-Region
    glue_data_path = data_path + "data_glue/"
    with open(glue_data_path + "peak2tf.pkl", "rb") as f:
        tf_peak_nx = dill.load(f)
    glue_tf_peak_df = nx.to_pandas_edgelist(tf_peak_nx) 
    glue_tf_peak_df["value"] = 1
    glue_tf_peak_df.rename(columns={'source':'Peak', 'target':'TF'}, inplace=True)

    glue_results = calculate_tf_metrics(glue_tf_peak_df, unibind_tf_peak_grn)
    print("*"*20 + " GLUE " + "*"*20)
    print(glue_results)
    
    ##########################################################################
    # LINGER TF-Region predictability

    linger_data_file = data_path + "data_linger/train/cell_population_TF_RE_binding.txt"
    
    linger_grn = pd.read_csv(linger_data_file, sep='\t')
    linger_grn = linger_grn.rename(columns={'Unnamed: 0': 'Peak'})
    linger_df= linger_grn.melt(id_vars=['Peak'], var_name='TF', value_name='regulation')
    if linger_df['Peak'].str.contains(':').any():
        print("检测到冒号格式，正在统一为连字符格式...")
        # 2. 将冒号替换为连字符
        # 替换前：chr1:100-200 -> 替换后：chr1-100-200
        linger_df['Peak'] = linger_df['Peak'].str.replace(':', '-', regex=False)
        
    else:
        print("格式已是连字符（chr1-100-200），无需转换。")
        
    linger_results = calculate_tf_metrics(linger_df, unibind_tf_peak_grn)


    ##########################################################################
    # GRaNIE TF-Region predictability
    
    granie_data_file = data_path + "GRaNIE_output/GRaNIE_tf_region_Links.csv"

    granie_grn = pd.read_csv(granie_data_file)

    granie_df = granie_grn[["TF.name", "peak.ID", "TF_peak.r", "TF_peak.fdr"]].copy()
    granie_df.rename(columns={"TF.name": "TF", "peak.ID": "Peak"}, inplace=True)

    if granie_df['Peak'].str.contains(':').any():
        print("检测到冒号格式，正在统一为连字符格式...")
        # 2. 将冒号替换为连字符
        # 替换前：chr1:100-200 -> 替换后：chr1-100-200
        granie_df['Peak'] = granie_df['Peak'].str.replace(':', '-', regex=False)
        
    else:
        print("格式已是连字符（chr1-100-200），无需转换。")
        
    granie_results = calculate_tf_metrics(granie_df, unibind_tf_peak_grn)


    ##########################################################################
    # Pando TF-Region predictability
    
    pando_data_file = data_path + "data_pando/tf_region_network.csv"

    pando_grn = pd.read_csv(pando_data_file)

    pando_df = pando_grn.copy()
    pando_df.rename(columns={"Region": "Peak"}, inplace=True)

    if pando_df['Peak'].str.contains(':').any():
        print("检测到冒号格式，正在统一为连字符格式...")
        # 2. 将冒号替换为连字符
        # 替换前：chr1:100-200 -> 替换后：chr1-100-200
        pando_df['Peak'] = pando_df['Peak'].str.replace(':', '-', regex=False)
        
    else:
        print("格式已是连字符（chr1-100-200），无需转换。")
        
    pando_results = calculate_tf_metrics(pando_df, unibind_tf_peak_grn)

    ####################################################################
    # Total results processing and plotting
    dyg_results['Method'] = 'DYG'
    glue_results['Method'] = 'GLUE'
    linger_results['Method'] = 'LINGER'
    granie_results['Method'] = 'GRaNIE'
    pando_results['Method'] = 'Pando'

    # 拼接在一起 (只取 F1 和 Method 列)
    df_combined = pd.concat([dyg_results[['F1', 'Method']], 
                            glue_results[['F1', 'Method']],
                            linger_results[['F1', 'Method']],
                            granie_results[['F1', 'Method']],
                            pando_results[['F1', 'Method']]])


    # ==========================================
    # 3. 绘制箱形图
    # ==========================================
    # 设置绘图风格
    sns.set_theme(style="ticks")
    plt.figure(figsize=(6, 5))

    x_axis_label = "Method"
    y_axis_label = "F1"
    # 画图
    ax = sns.boxplot(data=df_combined, 
                    x=x_axis_label,    # x轴放分组标签
                    y=y_axis_label,        # y轴放数值
                    palette="Set2", # 配色方案
                    width=0.5)     # 箱子宽度

    # (可选) 添加抖动散点，展示原始数据点分布
    sns.stripplot(data=df_combined, x=x_axis_label, y=y_axis_label, 
                color='black', size=3, alpha=0.5, jitter=True)

    # ==========================================
    # 4. 美化与保存
    # ==========================================
    plt.title("TF-Region " + y_axis_label + " Score Comparison")
    plt.ylabel(y_axis_label + "Score")
    plt.xlabel("") # 去掉 x 轴的 'Method' 字样，因为标签已经很清楚了
    sns.despine()  # 去掉上方和右侧的边框，更符合学术规范

    # 保存图片

    plt.savefig(output_path + "Benchmark_TF_Region_f1_boxplot_comparison.png", dpi=300, bbox_inches='tight')

#######################################################################################
#
#
#######################################################################################

def Region_Gene_Benchmark(data_path):

    output_path = data_path + "benchmark/"
    
    os.makedirs(output_path, exist_ok=True )

    adata_atac = ad.read_h5ad(data_path + "process/atac_processed.h5ad")
    adata_rna = ad.read_h5ad(data_path + "process/rna_processed.h5ad")
    total_peak = set(adata_atac.var_names)
    total_gene = set(adata_rna.var_names)

    ##################################################################
    benchmark_peak_gene_df = pd.read_pickle(data_path + "process/peak_gene_df.pkl")
    benchmark_peak_gene_df = benchmark_peak_gene_df.rename(columns= {"PeakID":"Peak", "gene_name":"Gene","hic_score":"value"})
    benchmark_peak_gene_df["label"] = (benchmark_peak_gene_df["value"]>0).astype(int)
    benchmark_peak_gene_df = benchmark_peak_gene_df[benchmark_peak_gene_df["Peak"].isin(total_peak)].copy()
    benchmark_peak_gene_df = benchmark_peak_gene_df[benchmark_peak_gene_df["Gene"].isin(total_gene)].copy()
    
    df_hic = benchmark_peak_gene_df.copy()
    df_hic = df_hic.rename(columns={"Peak":"region", "Gene":"gene"})
    
    ############################################################################################
    
    Markrer_Genes = adata_rna.var["highly_variable_rank"].copy()
    Markrer_Genes = Markrer_Genes.sort_values()
    top_marker_genes = list(Markrer_Genes.index[:100])
    
    ############################################################################################
    # DYGMAMBA peak-gene grn
    
    dyg_result_path = data_path + "data_dyg/"
    
    Node_id = pd.read_pickle(dyg_result_path + "node_id.pkl")
    graph_df = pd.read_pickle(dyg_result_path + "Graph_df.pkl")
    graph_df["Unnamed"] = graph_df.index
    name_list = ["Unnamed", "source_node", "target_node", "time", "label", "edge_idx"]
    New_Graph = graph_df[name_list].copy()
    New_Graph.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx']

    result_path = dyg_result_path + 'my_result_run0.npy'
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

    dyg_peak_gene_result = region_gene_evaluate(benchmark_peak_gene_df, avg_active_peak_gene_grn, 
                                                model_name="DYGMAMBA", output_path=output_path)

    df_pred = avg_active_peak_gene_grn.copy()
    df_pred = df_pred.rename(columns={"Peak":"region", "Gene":"gene", "avg_ts_weight":"value"})

    dyg_results = evaluate_per_gene_correlation(df_pred, df_hic, marker_genes=top_marker_genes, min_links=3)
    
    dyg_corr_score, dyg_plot_data = evaluate_scenic_plus_correlation(df_pred, df_hic, 
                                                             marker_genes=top_marker_genes)
    print("*"*50)
    print("*"*20 + " DYGMAMBA " + "*"*20)
    print(dyg_corr_score)
    print("*"*50)
    print(dyg_results)
    print("*"*50)
    print(dyg_peak_gene_result)
    #######################################################################################
    #
    #
    #######################################################################################
    # GLUE peak-gene grn

    glue_data_path = data_path + "data_glue/"

    with open(glue_data_path + "gene2peak.pkl", "rb") as f:
        gene_peak_nx = dill.load(f)
    glue_gene_peak_df = nx.to_pandas_edgelist(gene_peak_nx)

    glue_gene_peak_df.rename(columns={'source':'Gene', 'target':'Peak', 'weight':'predict'}, inplace=True)
    glue_gene_peak_df["predict"] = (glue_gene_peak_df["predict"]>0.9).astype(int)
    glue_gene_peak_df["value"] = glue_gene_peak_df["score"]
    glue_pred = glue_gene_peak_df[["Peak", "Gene", "value"]].copy()
    glue_pred = glue_pred.rename(columns={"Peak":"region", "Gene":"gene"})

    glue_peak_gene_result = region_gene_evaluate(benchmark_peak_gene_df, glue_gene_peak_df, 
                                                model_name="GLUE", output_path=output_path)  
    
    glue_corr_score, glue_plot_data = evaluate_scenic_plus_correlation(glue_pred, df_hic, 
                                                                       marker_genes=top_marker_genes)


    #######################################################################################
    # Pando peak-gene grn

    pando_data_file = data_path + "data_pando/region_gene_network.csv"
    
    pando_grn = pd.read_csv(pando_data_file)
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

    #######################################################################################
    # FigR peak-gene grn
    
    figr_data_file = data_path + "data_FigR/Region_Gene_Network.csv"
    
    figr_grn = pd.read_csv(figr_data_file)
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

    #######################################################################################
    # LINGER peak-gene grn
    
    linger_data_file = data_path + "data_linger/train/cell_population_cis_regulatory.txt"

    linger_grn = pd.read_csv(linger_data_file, 
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

    #######################################################################################
    # GRaNIE peak-gene grn
    
    granie_data_file = data_path + "GRaNIE_output/GRaNIE_region_gene_Links.csv" 

    granie_grn = pd.read_csv(granie_data_file)

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
    #######################################################################################
    #######################################################################################
    # results analysis

    results = pd.DataFrame({"GRaNIE":abs(granie_corr_score), "FigR":abs(figr_corr_score), 
                            "LINGER":abs(linger_corr_score), "Pando":abs(pando_corr_score), 
                            "GLUE":abs(glue_corr_score), "DYGMAMBA":abs(dyg_corr_score)}, index=[0])
    
    plot_data = results.T.reset_index()
    plot_data.columns = ['Method', 'Correlation']

    # 绘图
    plt.figure(figsize=(6, 6))
    sns.set_theme(style="ticks")
    sns.set_style("whitegrid") # 设置清爽的白色网格背景

    # 使用你喜欢的配色，比如 'viridis' 或 'magma'
    ax = sns.barplot(x='Method', y='Correlation', data=plot_data, palette='coolwarm')

    # 在柱子上方自动添加数值标签
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    plt.title("Region-Gene Benchmarking", fontsize=15)
    plt.ylim(0, max(plot_data['Correlation']) * 1.2) # 给上方数值留点空间
    plt.show()
    sns.despine()  # 去掉上方和右侧的边框，更符合学术规范

    # 保存图片
    plt.savefig(output_path + "Benchmark_Region_Gene_boxplot_comparison.png", 
                dpi=300, bbox_inches='tight')
    
    
    
    
    
##########################################################################
#
#
##########################################################################

if __name__ == "__main__":
    
    data_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/HepG2/"
    
    TF_Recovery_benchmark(data_path)
    
    TF_Gene_Benchmark(data_path)
    
    TF_Region_Benchmark(data_path)
    
    Region_Gene_Benchmark(data_path)
   
