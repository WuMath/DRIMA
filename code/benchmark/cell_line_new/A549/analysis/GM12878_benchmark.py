import sys
import code
import os
import pickle

import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tqdm import tqdm
import scipy.sparse
from scipy.sparse import csr_matrix
from gtfparse import read_gtf
from collections import defaultdict
from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

import networkx as nx
import dill

sys.path.append('/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src/')

from pdata.data_read import read_unibind_file
from benchmark.benchmark_model import analysis_tf_recovery

from benchmark.benchmark_model import analysis_tf_gene_data, analysis_tf_gene_all_correlation, \
    plot_top_n_tf_comparison, dyg_tf_gene_result
    
from benchmark.benchmark_model import analysis_tf_gene_all_precision


from benchmark.benchmark_model import analysis_tf_region

from benchmark.benchmark_model import plot_top_n_comparison

from benchmark.benchmark_model import analysis_region_gene, evaluate_r2g_benchmark, \
    plot_recovery_curves, plot_top_n_methods_comparison

from benchmark.benchmark_model import analysis_tf_gene_per_tf_precision, \
    analysis_tf_gene_correlation_summary, \
    analysis_tf_region_overall, \
    analysis_region_gene_precision
    
from benchmark.benchmark_model import plot_top_n_per_tf_comparison
from benchmark.benchmark_model import analysis_tf_gene_per_tf_topn

cell_type = "GM12878"
unibind_file = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/GM12878/unibind/GM12878_UniBind.tar.gz"



if __name__ == "__main__":
    


    flag_tf_recovery = True
    flag_tf_region = True
    flag_tf_gene = True
    flag_region_gene = True

    data_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/" + cell_type + "/"

    output_path = data_path + "benchmark/"

    os.makedirs(output_path, exist_ok=True )

    adata_atac = ad.read_h5ad(data_path + "process/atac_processed.h5ad")

    adata_rna = ad.read_h5ad(data_path + "process/rna_processed.h5ad")

    unibind_df_file = data_path + "benchmark/unibind_df.pkl"

    model_result_path = data_path + "data_dyg/"

    method_list = ["DyGMamba", "CellOracle", "FigR", "GLUE", "LINGER", "GRaNIE", "Pando"]

    method_colors = {
        "Global": "#E64B35",      # 比如：红色
        "Active": "#4DBBD5",      # 比如：天蓝色
        "DyGMamba": "#9467BD",
        "GLUE": "#00A087",        # 比如：绿色
        "FigR": "#3C5488",        # 比如：深蓝色
        "CellOracle": "#F39B7F",  # 比如：橙色
        "LINGER": "#8491B4",      # 比如：灰蓝色
        "GRaNIE": "#91D1C2",      # 比如：青色
        "Pando": "#DC0000"        # 比如：深红色
    }
    #############################################################################
    #
    # Unibind data processing
    #
    #############################################################################

    unibind_df, count_region_df = read_unibind_file(unibind_file)

    unibind_df["CellLine"] = unibind_df["CellLine"].str.split('_').str[0]
    count_region_df["CellLine"] = count_region_df["CellLine"].str.split('_').str[0]

    print(unibind_df.head())
    print(count_region_df.head())

    unibind_df.to_pickle(output_path + "unibind_df.pkl")
    count_region_df.to_pickle(output_path + "count_region_df.pkl")


    ##############################################################################
    #
    # TF recovery
    #
    ##############################################################################
    if flag_tf_recovery == True:
        analysis_tf_recovery(data_path, output_path, method_list, jaspar_threshold= 0, method_colors=method_colors)



    ##############################################################################
    #
    # TF-Region
    #
    ##############################################################################
    if flag_tf_region == True:
        ###################################
        # Benchmark

        f_score = 0.1

        tf_region_result = analysis_tf_region(unibind_df_file, data_path, f_score, method_list, jaspar_threshold=0)


        ###################################
        # 

        # # 1. 准备数据字典
        # tf_region_result = {
        #     "DYGMAMBA": dyg_results,
        #     "GLUE": glue_results,
        #     "LINGER": linger_results,
        #     "GRaNIE": granie_results,
        #     "Pando": pando_results
        # }

        # 2. 调用函数
        # 例如：对比前 10 个基因的 Precision
        top_results = plot_top_n_comparison(method_dict=tf_region_result, 
            output_file=output_path + 'Benchmark of TF-Region Comparison of Top 10 Precision.png',
            top_n=10, score_col="Precision", palette = method_colors)

        # 2. 调用函数
        # 例如：对比前 50 个基因的 fscore
        top_results = plot_top_n_comparison(method_dict= tf_region_result, 
            output_file= output_path + 'Benchmark of TF-Region Comparison of Top 50 F1 Score.png',
            top_n= 50, score_col= "F1", palette = method_colors)


        ###########################
        # Boxplot
        ###########################
        # precisioin

        analysis_type = "Precision"
        tf_region_df = []

        for method_name, df in tf_region_result.items():

            tf_region_df.append(df[[analysis_type, 'Method']])

        tf_region_df_combined = pd.concat(tf_region_df, ignore_index=True)

        # ==========================
        # @ 绘制箱形图
        # ==========================
        sns.set_theme(style="ticks")
        plt.figure(figsize=(6, 5))

        ax = sns.boxplot(data=tf_region_df_combined, x='Method',    # x轴放分组标签
                        y=analysis_type,        # y轴放数值
                        palette=method_colors, # 配色方案
                        width=0.5)     # 箱子宽度

        sns.stripplot(data=tf_region_df_combined, x='Method', y=analysis_type, 
                    color='black', size=3, alpha=0.5, jitter=True)

        plt.title("Benchmark of TF-Region Precision Comparison")
        plt.ylabel("Precision")
        plt.xlabel("") # 去掉 x 轴的 'Method' 字样，因为标签已经很清楚了
        sns.despine()  # 去掉上方和右侧的边框，更符合学术规范

        plt.savefig(output_path + 'Benchmark of TF-Region Comparison of Precision.png', dpi=1200, bbox_inches='tight')


        ################################
        # F1 score

        analysis_type = "F1"
        tf_region_df = []

        for method_name, df in tf_region_result.items():

            tf_region_df.append(df[[analysis_type, 'Method']])

        tf_region_df_combined = pd.concat(tf_region_df, ignore_index=True)

        # =============================
        # @ 绘制箱形图
        # =============================
        sns.set_theme(style="ticks")
        plt.figure(figsize=(6, 5))

        ax = sns.boxplot(data=tf_region_df_combined, 
                        x='Method',    # x轴放分组标签
                        y=analysis_type,        # y轴放数值
                        palette=method_colors, # 配色方案
                        width=0.5)     # 箱子宽度

        sns.stripplot(data=tf_region_df_combined, x='Method', y=analysis_type, 
                    color='black', size=3, alpha=0.5, jitter=True)

        plt.title("Benchmark of TF-Region F1 Score Comparison")
        plt.ylabel("F1 Score")
        plt.xlabel("") # 去掉 x 轴的 'Method' 字样，因为标签已经很清楚了
        sns.despine()  # 去掉上方和右侧的边框，更符合学术规范

        plt.savefig(output_path + 'Benchmark of TF-Region Comparison of F1 Score.png', dpi=1200, bbox_inches='tight')
        
        # 补充分析 3（直接复用已有 tf_region_result）
        analysis_tf_region_overall(tf_region_result, output_path, method_colors, beta=1.0)



    ##############################################################################
    #
    # Region-Gene
    #
    ##############################################################################
    if flag_region_gene == True:
        
        region_gene_results, region_gene_method_data, benchmark_peak_gene_df = analysis_region_gene(data_path, output_path, method_list)

        #################################
        # Correlation

        plot_data = region_gene_results.T.reset_index()
        plot_data.columns = ['Method', 'Correlation']

        # 2. 绘图
        plt.figure(figsize=(6, 6))
        sns.set_style("whitegrid") # 设置清爽的白色网格背景

        ax = sns.barplot(x='Method', y='Correlation', data=plot_data, hue = "Method", palette=method_colors)

        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.3f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')

        plt.title("Benchmark pf Region-Gene correlation comparsion", fontsize=15)
        plt.ylim(0, max(plot_data['Correlation']) * 1.2) # 给上方数值留点空间

        plt.savefig(output_path + 'Benchmark of Region-Gene Correlation Comparison.png', dpi=1200, bbox_inches='tight')


        #############################
        # score analysis

        corr_df, metrics_df = evaluate_r2g_benchmark(benchmark_peak_gene_df, region_gene_method_data)

        corr_df['Abs_Spearman_Rho'] = corr_df['Spearman_Rho'].abs()

        # 调用
        plot_recovery_curves(corr_df, 
                            output_path + 'Benchmark of Recovery Curve of Region-Gene Prediction.png',
                            method_colors,
                            compared_method='Abs_Spearman_Rho')

        plot_top_n_methods_comparison(corr_df, 
                                    output_path + 'Benchmark of Region-Gene Comparison of Top 200 Spearman Rho.png',
                                    method_colors, top_n=200, compare_method='Abs_Spearman_Rho')
        
        # 补充分析 4（直接复用已有 method_data 和 benchmark_peak_gene_df）
        
        analysis_region_gene_precision(benchmark_peak_gene_df, region_gene_method_data, output_path, method_colors, beta=1.0)
        
        
    ##############################################################################
    #
    # TF-Gene
    #
    ##############################################################################
    if flag_tf_gene == True:
        
        beta = 1.0
        
        tf_gene_grn, avg_active_tf_gene_grn, avg_global_tf_gene_grn = dyg_tf_gene_result(data_path, model_result_path, jaspar_threshold=0)

        tf_gene_correlation_result_dict, tf_gene_precision_result_dict, \
            precision_result_df, tf_gene_per_precision_dict = analysis_tf_gene_data( data_path, method_list, 
                                                                flag_corr = True, flag_precision = True, 
                                                                beta = beta)

        analysis_tf_gene_all_correlation(output_path, tf_gene_correlation_result_dict, method_colors)

        ###########################
        # Top 100 correlation

        # 1. 定义绘图顺序
        my_order = ['Global', 'Active', 'GLUE', 'FigR', 'CellOracle', 'LINGER', 'GRaNIE', 'Pando']

        plot_top_n_tf_comparison(tf_gene_correlation_result_dict, 
                                output_path, method_colors, top_n=100, 
                                score_col='Correlation', my_order=my_order)

        #############################
        # Total correlation

        analysis_tf_gene_all_precision(precision_result_df, output_path, method_colors)
        

        tf_gene_prec, tf_gene_fscore, tf_gene_rec =\
            analysis_tf_gene_per_tf_precision(tf_gene_per_precision_dict, output_path, method_colors, beta=beta)

        # 补充分析 2（直接复用已有 result_dict）
        analysis_tf_gene_correlation_summary(tf_gene_correlation_result_dict, output_path, method_colors)

        analysis_tf_gene_per_tf_topn(tf_gene_per_precision_dict, output_path = output_path,
                                    top_n_list  = [20, 50, 100], beta        = beta,
                                    method_colors = method_colors)
    
