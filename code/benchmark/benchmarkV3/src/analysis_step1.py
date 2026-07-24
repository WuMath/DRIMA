"""
step1_fix_excel.py
对每个细胞系重跑分析，修复 Excel 只保存 TF-gene 的问题
运行一次即可，之后直接用 benchmark_plot.py 绘图
"""
import sys, os
sys.path.append('/home/wuyan/dygmamba_project/model/dygmamba/src/')
import re, warnings
import pandas as pd
import numpy as np
import anndata as ad
warnings.filterwarnings('ignore')

from benchmark.benchmark_model_new import (
    analysis_tf_region, calc_overall_metrics,
    analysis_region_gene, analysis_region_gene_precision,
    analysis_tf_gene_data, dyg_tf_gene_result_inherit,
    analysis_tf_recovery, read_unibind_file
)



def find_unibind_file(ct, data_root):
    unibind_dir = f"{data_root}{ct}/unibind/"
    bench_dir   = f"{data_root}{ct}/benchmarkV7/"
    # 优先用已处理的 pkl
    pkl = bench_dir + "unibind_df.pkl"
    if os.path.exists(pkl):
        return pkl, 'pkl'
    # 找 tar.gz
    import glob
    candidates = glob.glob(f"{unibind_dir}*.tar.gz")
    if candidates:
        return candidates[0], 'tar'
    return None, None

def safe_sheet(name, used, max_len=31):
    name = re.sub(r'[:\\/*?\[\]]', '_', str(name))[:max_len]
    base, i = name, 1
    while name in used:
        sfx = f"_{i}"; name = base[:max_len-len(sfx)] + sfx; i += 1
    used.add(name)
    return name


######################################################################
# 参数
######################################################################
data_root   = "/home/wuyan/dygmamba_project/data/cell_line_drima/"
CELL_TYPES  = ["GM12878","HepG2","IMR90","K562","MCF7","A549","H1","HELA","SK"]
METHOD_LIST = ["DyGMamba","CellOracle","FigR","GLUE","LINGER","GRaNIE","Pando"]
BETA        = 0.1

UNIBIND_FILES = {
    "GM12878": "GM12878_UniBind_search_*.tar.gz",
    "HepG2":   "HepG2_UniBind_search_*.tar.gz",
    "IMR90":   "IMR90_UniBind_search_*.tar.gz",
    "K562":    "K562_UniBind_search_*.tar.gz",
    "MCF7":    "MCF7_UniBind_search_*.tar.gz",
    "A549":    "A549_UniBind_search_*.tar.gz",
    "H1":      "H1_UniBind_search_*.tar.gz",
    "HELA":    "HELA_UniBind_search_*.tar.gz",
    "SK":      "SK_UniBind_search_*.tar.gz",
}

COLORS = {
    "DyGMamba":  "#9467BD",
    "GLUE":      "#00A087",
    "FigR":      "#3C5488",
    "CellOracle":"#F39B7F",
    "LINGER":    "#8491B4",
    "GRaNIE":    "#91D1C2",
    "Pando":     "#DC0000",
}

# 参数（与 notebook 保持一致）


THRESHOLD_LIST = {
    "GM12878":{"unibind_threshold": 92, "jaspar_threshold": 4.8, "jaspar_score_threshold": 9,
               "jaspar_q_threshold": 0.4, "hic_threshold": 160, "hic_threshold_lower": 20,
               "self_threshold": 0.88},
    "HepG2":{"unibind_threshold": 92, "jaspar_threshold": 4.8, "jaspar_score_threshold": 9,
               "jaspar_q_threshold": 0.4, "hic_threshold": 160, "hic_threshold_lower": 20,
               "self_threshold": 0.88},
    "IMR90":{"unibind_threshold": 92, "jaspar_threshold": 4.8, "jaspar_score_threshold": 9,
               "jaspar_q_threshold": 0.4, "hic_threshold": 160, "hic_threshold_lower": 20,
               "self_threshold": 0.88},
    "K562":{"unibind_threshold": 92, "jaspar_threshold": 4.8, "jaspar_score_threshold": 9,
               "jaspar_q_threshold": 0.4, "hic_threshold": 160, "hic_threshold_lower": 20,
               "self_threshold": 0.88},
    "MCF7":{"unibind_threshold": 92, "jaspar_threshold": 4.8, "jaspar_score_threshold": 9,
               "jaspar_q_threshold": 0.4, "hic_threshold": 160, "hic_threshold_lower": 20,
               "self_threshold": 0.88},
    "A549":{"unibind_threshold": 92, "jaspar_threshold": 4.8, "jaspar_score_threshold": 9,
               "jaspar_q_threshold": 0.4, "hic_threshold": 140, "hic_threshold_lower": 30,
               "self_threshold": 0.7},
    "H1":{"unibind_threshold": 92, "jaspar_threshold": 4.8, "jaspar_score_threshold": 9,
            "jaspar_q_threshold": 0.4, "hic_threshold": 1, "hic_threshold_lower": 0,
            "self_threshold": 0.88},
    "HELA":{"unibind_threshold": 92, "jaspar_threshold": 4.8, "jaspar_score_threshold": 9,
            "jaspar_q_threshold": 0.4, "hic_threshold": 1, "hic_threshold_lower": 0,
            "self_threshold": 0.88},
    "SK":{"unibind_threshold": 92, "jaspar_threshold": 4.8, "jaspar_score_threshold": 9,
               "jaspar_q_threshold": 0.4, "hic_threshold": 1, "hic_threshold_lower": 0,
               "self_threshold": 0.88}
}

######################################################################
# 对每个细胞系运行全部分析并保存完整 Excel
######################################################################
for ct in CELL_TYPES:
    print(f"\n{'='*55}\n处理: {ct}\n{'='*55}")

    unibind_threshold      = THRESHOLD_LIST.get(ct).get("unibind_threshold")
    jaspar_threshold       = THRESHOLD_LIST.get(ct).get("jaspar_threshold")
    jaspar_score_threshold = THRESHOLD_LIST.get(ct).get("jaspar_score_threshold")
    jaspar_q_threshold     = THRESHOLD_LIST.get(ct).get("jaspar_q_threshold")
    
    hic_threshold          = THRESHOLD_LIST.get(ct).get("hic_threshold")
    hic_threshold_lower    = THRESHOLD_LIST.get(ct).get("hic_threshold_lower")
    self_threshold         = THRESHOLD_LIST.get(ct).get("self_threshold")
    
    data_path  = f"{data_root}{ct}/"
    bench_path = f"{data_path}benchmarkV7/"
    excel_file = bench_path + "benchmark_all_results.xlsx"
    os.makedirs(bench_path, exist_ok=True)

    # ── Step A: JASPAR 预处理 ─────────────────────────────────────
    pred_tf_peak_path = data_path + "process/pred_tf_peak.pkl"
    jaspar_pkl_path   = data_path + "process/jaspar_df.pkl"

    if os.path.exists(pred_tf_peak_path):
        print(f"  ✓ pred_tf_peak.pkl 已存在")
    elif os.path.exists(jaspar_pkl_path):
        try:
            jaspar_data = pd.read_pickle(jaspar_pkl_path)
            jaspar_data['sequence_name'] = \
                jaspar_data['sequence_name'].str.replace(':', '-')

            jtp = jaspar_data[
                ['TF_Symbol','sequence_name','score','p-value','q-value']
            ].copy()
            jtp.rename(columns={'TF_Symbol':'TF','sequence_name':'Peak'},
                       inplace=True)
            jtp.drop_duplicates(subset=['TF','Peak'], keep='first',
                                inplace=True)
            jtp = jtp[(jtp['q-value'] < jaspar_q_threshold) &
                      (jtp['score']   > jaspar_score_threshold)]
            jtp['value'] = -np.log10(jtp['p-value'])
            mask = (jtp['value'] > jaspar_threshold) & (jtp['value'] < 8)
            jtp.loc[mask, ['TF','Peak','value']].copy().to_pickle(
                pred_tf_peak_path)
            print(f"  ✓ pred_tf_peak.pkl 生成: "
                  f"{mask.sum()} pairs, "
                  f"{jtp.loc[mask,'TF'].nunique()} TFs")
        except Exception as e:
            print(f"  [JASPAR 错误] {e}")
    else:
        print(f"  [警告] 找不到 jaspar_df.pkl，TF-Region 可能失败")
        print(f"         路径: {jaspar_pkl_path}")

    # ── Step B: UniBind 处理 ──────────────────────────────────────
    unibind_path, utype = find_unibind_file(ct, data_root)
    if unibind_path is None:
        print(f"  [跳过] 找不到 UniBind 数据")
        continue

    used_sheets = set()
    result_store = {}   # sheet_name → DataFrame
    
    # ── Step  TF-Recovery ────────────────────────────────────
    try:
        num_tf_result, tf_set_result = analysis_tf_recovery(
            data_path, bench_path, METHOD_LIST,
            method_colors=COLORS)

        # tf_set_result: dict{method: set of recovered TFs}
        # 转为 DataFrame
        tf_recovery_df = pd.DataFrame(
            {k: pd.Series(list(v)) for k, v in tf_set_result.items()}
        )
        result_store['TF_recovery_set'] = tf_recovery_df

        # num_tf_result: dict{method: list of cumulative counts}
        # 转为 DataFrame（每行是一个 rank，每列是一个方法）
        num_tf_df = pd.DataFrame(num_tf_result)
        num_tf_df.index.name = 'Rank'
        num_tf_df = num_tf_df.reset_index()
        result_store['TF_recovery_num'] = num_tf_df

        # count_region_df（ground truth 排名）
        count_region_pkl = bench_path + "count_region_df.pkl"
        if os.path.exists(count_region_pkl):
            ground_truth_ranked = pd.read_pickle(count_region_pkl)\
                .sort_values(by="PeakCount", ascending=False)
            result_store['TF_recovery_rank'] = ground_truth_ranked

        print(f"  ✓ TF-Recovery: {len(tf_set_result)} methods, "
              f"recovered TFs: "
              f"{ {m: len(v) for m, v in tf_set_result.items()} }")
    except Exception as e:
        print(f"  [TF-Recovery 错误] {e}")

    # ── TF-Region ─────────────────────────────────────────────────
    try:
        if utype == 'pkl':
            unibind_df_file = unibind_path
        else:
            # 从 tar 读取并保存 pkl
            from benchmark.benchmark_model_new import read_unibind_file
            unibind_df, count_region_df = read_unibind_file(unibind_path)
            unibind_df["CellLine"] = unibind_df["CellLine"].str.split('_').str[0]
            count_region_df["CellLine"] = count_region_df["CellLine"].str.split('_').str[0]
            unibind_df = unibind_df[(unibind_df['score']>92)&(unibind_df['score']<95)]
            count_region_df = count_region_df[count_region_df['TF'].isin(set(unibind_df['TF']))]
            unibind_df_file = bench_path + "unibind_df.pkl"
            unibind_df.to_pickle(unibind_df_file)
            count_region_df.to_pickle(bench_path + "count_region_df.pkl")

        tf_region_result, _ = analysis_tf_region(
            unibind_df_file, data_path, BETA, METHOD_LIST)
        all_tf_region_result = calc_overall_metrics(tf_region_result, beta=BETA)

        # per-TF（加 Method 列）
        rows = []
        for method, df in tf_region_result.items():
            if len(df) == 0: continue
            d = df.copy(); d['Method'] = method; rows.append(d)
        if rows:
            result_store['TF_region_per'] = pd.concat(rows, ignore_index=True)
        result_store['TF_region_all'] = all_tf_region_result
        print(f"  ✓ TF-Region: {len(tf_region_result)} methods")
    except Exception as e:
        print(f"  [TF-Region 错误] {e}")

    # ── Region-Gene ────────────────────────────────────────────────
    try:
        rg_total_corr, rg_per_corr_df, rg_method_data, bench_peak_gene_df = \
            analysis_region_gene(data_path, bench_path, METHOD_LIST,
                                hic_threshold= hic_threshold, hic_threshold_lower= hic_threshold_lower,
                                self_threshold= self_threshold)

        rg_per_corr_df['Abs_Spearman_Rho'] = rg_per_corr_df['Spearman_Rho'].abs()

        rg_total_prec, rg_per_prec_dict, _ = analysis_region_gene_precision(
            bench_peak_gene_df, rg_method_data, bench_path,
            {"DyGMamba":"#9467BD"}, beta=BETA)

        rows = []
        for method, df in rg_per_prec_dict.items():
            if len(df) == 0: continue
            d = df.copy(); d['Method'] = method; rows.append(d)

        # total_corr → DataFrame
        if hasattr(rg_total_corr, 'T'):
            plot_data = rg_total_corr.T.reset_index()
            plot_data.columns = ['Method','Spearman_Total']
            result_store['Region_gene_total_corr'] = plot_data
        result_store['Region_gene_per_corr']       = rg_per_corr_df
        result_store['Region_gene_total_precision'] = rg_total_prec
        if rows:
            result_store['Region_gene_per_precision'] = pd.concat(rows, ignore_index=True)
        print(f"  ✓ Region-Gene: {rg_per_corr_df['Method'].nunique()} methods, "
            f"{len(rg_per_corr_df)} records")
    except Exception as e:
        print(f"  [Region-Gene 错误] {e}")

    # ── TF-Gene ────────────────────────────────────────────────────
    try:
        model_result_path = f"{data_path}data_dyg/"
        _ = dyg_tf_gene_result_inherit(data_path, model_result_path)

        corr_dict, _, prec_df, per_prec_dict = analysis_tf_gene_data(
            data_path, METHOD_LIST, flag_corr=True, flag_precision=True, beta=BETA)

        rows_corr, rows_prec = [], []
        for method, df in corr_dict.items():
            if len(df) == 0: continue
            d = df.copy(); d['Method'] = method; rows_corr.append(d)
        for method, df in per_prec_dict.items():
            if len(df) == 0: continue
            d = df.copy(); d['Method'] = method; rows_prec.append(d)

        if rows_corr:
            result_store['TF_gene_per_corr']      = pd.concat(rows_corr, ignore_index=True)
        if rows_prec:
            result_store['TF_gene_per_precision'] = pd.concat(rows_prec, ignore_index=True)
        if prec_df is not None and len(prec_df) > 0:
            result_store['TF_gene_total_precision'] = prec_df
        print(f"  ✓ TF-Gene: {len(corr_dict)} methods")
    except Exception as e:
        print(f"  [TF-Gene 错误] {e}")

    # ── 统一写入 Excel ─────────────────────────────────────────────
    if result_store:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            for sheet_name, df in result_store.items():
                sn = safe_sheet(sheet_name, used_sheets)
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    df.to_excel(writer, sheet_name=sn, index=False)
                    print(f"    saved sheet: {sn}  ({len(df)} rows)")
        print(f"  ✓ 完整 Excel 已保存: {excel_file}")
    else:
        print(f"  [警告] {ct} 无数据写入")

print("\n所有细胞系处理完成！")

