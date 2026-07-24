"""
benchmark_final_plots.py

评估指标参考文献：
────────────────────────────────────────────────────────────────────
层次          指标              参考文献
────────────────────────────────────────────────────────────────────
TF-Region    F0.1 (β=0.1)     Bravo González-Blas et al. Nature Methods 20(10):1355-1367 (2023) [SCENIC+]
             Precision/Recall  Wang et al. Nature Methods 20(9):1368-1378 (2023) [Dictys]
             Macro F0.1        Huynh-Thu et al. PLOS ONE 5(9):e12776 (2010) [GENIE3]

Region-Gene  Spearman ρ        Pliner et al. Molecular Cell 71(5):858-871 (2018) [Cicero]
             F0.1/Recall       Kartha et al. Cell Genomics 2(12):100237 (2022) [FigR]
             AUPRC             Yuan & Duren Nature Biotechnology 43:247-257 (2025) [LINGER]

TF-Gene      Correlation       Kamimoto et al. Nature 614(7949):742-751 (2023) [CellOracle]
             F-score           Pratapa et al. Nature Methods 17(2):147-154 (2020) [BEELINE]
             Precision         Omony et al. Brief. Bioinformatics 20(3):812-823 (2019)

TF-Recovery  Recovery Curve    Bravo González-Blas et al. Nature Methods 20(10):1355-1367 (2023)
────────────────────────────────────────────────────────────────────
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

######################################################################
# 参数
######################################################################
data_root  = "/home/wuyan/dygmamba_project/data/cell_line/"
output_dir = "/home/wuyan/dygmamba_project/data/benchmark_summary/final/"
os.makedirs(output_dir, exist_ok=True)

CELL_TYPES  = ["GM12878","HepG2","IMR90","K562","MCF7","A549","H1","HELA","SK"]
METHOD_LIST = ["DyGMamba","CellOracle","FigR","GLUE","LINGER","GRaNIE","Pando"]
FOCAL       = "DyGMamba"
TOP_N       = 50

COLORS = {
    "DyGMamba":  "#9467BD",
    "GLUE":      "#00A087",
    "FigR":      "#3C5488",
    "CellOracle":"#F39B7F",
    "LINGER":    "#8491B4",
    "GRaNIE":    "#91D1C2",
    "Pando":     "#DC0000",
}

######################################################################
# 数据读取
######################################################################
def load_excel(ct, sheet):
    path = f"{data_root}{ct}/benchmarkV7/benchmark_all_results.xlsx"
    if not os.path.exists(path): return None
    try:
        xl = pd.ExcelFile(path)
        if sheet not in xl.sheet_names: return None
        df = xl.parse(sheet)
        if 'Method' in df.columns:
            df['Method'] = df['Method'].astype(str).str.strip()
        return df
    except Exception:
        return None

def load_all_cts(sheet, col, cell_types=CELL_TYPES):
    """汇总: {ct: {method: np.array(values)}}"""
    result = {}
    for ct in cell_types:
        df = load_excel(ct, sheet)
        if df is None or col not in df.columns: continue
        ct_data = {}
        for m in METHOD_LIST:
            sub = df[df['Method']==m][col].dropna().values.astype(float)
            if len(sub) > 0:
                ct_data[m] = sub
        if ct_data:
            result[ct] = ct_data
    return result

def top_n_data(ct_data_dict, n=TOP_N, sort_col_data=None):
    """取每个方法按值降序排列后的 top-N"""
    result = {}
    for ct, method_dict in ct_data_dict.items():
        result[ct] = {}
        for m, vals in method_dict.items():
            sorted_vals = np.sort(vals)[::-1]
            result[ct][m] = sorted_vals[:min(n, len(sorted_vals))]
    return result

print("加载数据...")
# ── TF-Region ──────────────────────────────────────────────────────
tfr_fscore_all  = load_all_cts('TF_region_per', 'fscore')
tfr_prec_all    = load_all_cts('TF_region_per', 'Precision')
tfr_recall_all  = load_all_cts('TF_region_per', 'Recall')
tfr_macro       = {}   # ct → {method: scalar}
for ct in CELL_TYPES:
    df = load_excel(ct, 'TF_region_all')
    if df is None: continue
    row_dict = {}
    for col_try in ['F_score','fscore','F-beta']:
        if col_try in df.columns:
            for _, r in df[['Method',col_try]].dropna().iterrows():
                row_dict[r['Method']] = float(r[col_try])
            break
    if row_dict: tfr_macro[ct] = row_dict

tfr_macro_prec = {}
for ct in CELL_TYPES:
    df = load_excel(ct, 'TF_region_all')
    if df is None: continue
    if 'Precision' in df.columns:
        tfr_macro_prec[ct] = {r['Method']: float(r['Precision'])
                              for _, r in df[['Method','Precision']].dropna().iterrows()}

tfr_macro_recall = {}
for ct in CELL_TYPES:
    df = load_excel(ct, 'TF_region_all')
    if df is None: continue
    if 'Recall' in df.columns:
        tfr_macro_recall[ct] = {r['Method']: float(r['Recall'])
                                for _, r in df[['Method','Recall']].dropna().iterrows()}

# ── Region-Gene ────────────────────────────────────────────────────
rg_spearman     = load_all_cts('Region_gene_per_corr',       'Abs_Spearman_Rho')
rg_fscore_per   = load_all_cts('Region_gene_per_precision',  'F_score')
rg_recall_per   = load_all_cts('Region_gene_per_precision',  'Recall')
rg_prec_per     = load_all_cts('Region_gene_per_precision',  'Precision')
rg_auprc_per    = load_all_cts('Region_gene_per_precision',  'AUPRC')
rg_total        = {}   # ct → {method: {metric: scalar}}
for ct in CELL_TYPES:
    df = load_excel(ct, 'Region_gene_total_precision')
    if df is None: continue
    d = {}
    for _, r in df.dropna(subset=['Method']).iterrows():
        m = str(r['Method']).strip()
        d[m] = {}
        for col in ['Precision','Recall','F-beta','F_score','AUPRC']:
            if col in r.index and pd.notna(r[col]):
                d[m][col] = float(r[col])
    if d: rg_total[ct] = d

# ── TF-Gene ────────────────────────────────────────────────────────
tfg_corr_per    = load_all_cts('TF_gene_per_corr',       'Correlation')
tfg_fscore_per  = load_all_cts('TF_gene_per_precision',  'F_score')
tfg_prec_per    = load_all_cts('TF_gene_per_precision',  'Precision')
tfg_recall_per  = load_all_cts('TF_gene_per_precision',  'Recall')
tfg_total       = {}
for ct in CELL_TYPES:
    df = load_excel(ct, 'TF_gene_total_precision')
    if df is None: continue
    d = {}
    for _, r in df.dropna(subset=['Method']).iterrows():
        m = str(r['Method']).strip()
        d[m] = {}
        for col in ['Precision','Recall','F-beta','AUC']:
            if col in r.index and pd.notna(r[col]):
                d[m][col] = float(r[col])
    if d: tfg_total[ct] = d

# ── TF-Recovery ────────────────────────────────────────────────────
tf_recovery = {}   # ct → {method: np.array of cumulative counts}
for ct in CELL_TYPES:
    df = load_excel(ct, 'TF_recovery_num')
    if df is None: continue
    d = {}
    for m in METHOD_LIST:
        if m in df.columns:
            d[m] = df[m].dropna().values.astype(float)
    if d: tf_recovery[ct] = d

avail_ct = sorted(set(
    list(tfr_fscore_all.keys()) +
    list(rg_spearman.keys()) +
    list(tfg_corr_per.keys())
))
print(f"可用细胞系: {avail_ct}  ({len(avail_ct)}个)")

######################################################################
# 通用绘图工具
######################################################################
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})

def get_present(ct_data):
    """返回在当前数据中存在的方法（保持顺序）"""
    all_m = set()
    for d in ct_data.values():
        all_m |= set(d.keys())
    return [m for m in METHOD_LIST if m in all_m]

def bar_colors(methods):
    return [COLORS.get(m,'#999') for m in methods]

def edge_colors(methods):
    return [COLORS[FOCAL] if m==FOCAL else '#444' for m in methods]

def edge_widths(methods):
    return [2.5 if m==FOCAL else 0.6 for m in methods]

def add_rank_badge(ax, rank, total):
    """右上角添加排名徽章"""
    color = '#006400' if rank == 1 else '#555'
    ax.text(0.97, 0.97, f'#{rank}/{total}',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=7, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=color, alpha=0.85, linewidth=1.2))

def focal_rank(ct_data, ct):
    """DyGMamba 在当前细胞系中的排名"""
    if ct not in ct_data or FOCAL not in ct_data[ct]:
        return None, None
    means = {m: np.mean(v) for m, v in ct_data[ct].items() if len(v)>0}
    sorted_m = sorted(means, key=means.get, reverse=True)
    rank = sorted_m.index(FOCAL)+1 if FOCAL in sorted_m else None
    return rank, len(sorted_m)

def plot_boxplot_panel(ax, ct_data, ct, title='', ylabel='',
                       show_xlabel=True, top_n=None):
    """在 ax 上绘制单个细胞系的箱线图"""
    if ct not in ct_data:
        ax.text(0.5,0.5,'N/A',ha='center',va='center',
                transform=ax.transAxes,color='#aaa',fontsize=9)
        ax.set_title(f'{ct}\n{title}',fontsize=8,fontweight='bold')
        return

    data  = ct_data[ct]
    if top_n:
        data = {m: np.sort(v)[::-1][:min(top_n,len(v))]
                for m,v in data.items()}
    methods = [m for m in METHOD_LIST if m in data and len(data[m])>0]
    if not methods:
        ax.text(0.5,0.5,'N/A',ha='center',va='center',transform=ax.transAxes)
        return

    plot_df = pd.DataFrame([
        {'Method':m,'Value':v}
        for m in methods for v in data[m]
    ])
    sns.boxplot(data=plot_df, x='Method', y='Value',
                order=methods,
                palette={m:COLORS.get(m,'#999') for m in methods},
                ax=ax, width=0.55, linewidth=0.9,
                flierprops={'marker':'o','markersize':2,'alpha':0.3},
                hue='Method', legend=False)

    # DyGMamba 背景高亮
    if FOCAL in methods:
        fi = methods.index(FOCAL)
        ax.axvspan(fi-0.4, fi+0.4, alpha=0.1,
                   color=COLORS[FOCAL], zorder=0)
        # 中位数标注
        med = np.median(data[FOCAL])
        ax.text(fi, med, f'{med:.3f}',
                ha='center', va='bottom', fontsize=6,
                color=COLORS[FOCAL], fontweight='bold')

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m[:6] for m in methods],
                       rotation=40, ha='right', fontsize=6.5)
    ax.set_title(f'{ct}', fontsize=8.5, fontweight='bold')
    if ylabel: ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_xlabel('')
    ax.spines[['top','right']].set_visible(False)

    rank, total = focal_rank(ct_data, ct)
    if rank is not None:
        add_rank_badge(ax, rank, total)

def plot_bar_panel(ax, ct_data_scalar, ct, title='', ylabel='',
                   show_xlabel=True):
    """在 ax 上绘制单个细胞系的柱状图（scalar 值）"""
    if ct not in ct_data_scalar:
        ax.text(0.5,0.5,'N/A',ha='center',va='center',
                transform=ax.transAxes,color='#aaa',fontsize=9)
        ax.set_title(f'{ct}',fontsize=8.5,fontweight='bold')
        return

    data    = ct_data_scalar[ct]
    methods = [m for m in METHOD_LIST if m in data]
    if not methods: return

    values  = [data[m] for m in methods]
    x       = np.arange(len(methods))
    bars    = ax.bar(x, values,
                     color=bar_colors(methods),
                     edgecolor=edge_colors(methods),
                     linewidth=[ew for ew in edge_widths(methods)],
                     alpha=0.85, width=0.65, zorder=3)

    # DyGMamba 高亮
    if FOCAL in methods:
        fi = methods.index(FOCAL)
        ax.axvspan(fi-0.4, fi+0.4, alpha=0.1,
                   color=COLORS[FOCAL], zorder=0)
        ax.text(fi, values[fi] + max(values)*0.03,
                f'{values[fi]:.3f}',
                ha='center', va='bottom', fontsize=6.5,
                color=COLORS[FOCAL], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([m[:6] for m in methods],
                       rotation=40, ha='right', fontsize=6.5)
    ax.set_title(f'{ct}', fontsize=8.5, fontweight='bold')
    if ylabel: ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_xlabel('')
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(0, max(values)*1.25 if max(values)>0 else 1)

    rank = sorted(methods, key=lambda m: data[m], reverse=True).index(FOCAL)+1 \
           if FOCAL in methods else None
    if rank is not None:
        add_rank_badge(ax, rank, len(methods))

def plot_bar_panel_from_per(ax, ct_data, ct, ylabel=''):
    """从 per-entry 数据计算均值，绘制柱状图"""
    if ct not in ct_data:
        ax.text(0.5,0.5,'N/A',ha='center',va='center',
                transform=ax.transAxes,color='#aaa',fontsize=9)
        ax.set_title(f'{ct}',fontsize=8.5,fontweight='bold'); return
    scalar = {m: np.mean(v) for m,v in ct_data[ct].items() if len(v)>0}
    plot_bar_panel(ax, {ct: scalar}, ct, ylabel=ylabel)

def add_reference(fig, text, y=0.01):
    fig.text(0.5, y, text, ha='center', va='bottom',
             fontsize=6.5, color='#444',
             style='italic', wrap=True)

def make_legend(fig, methods, y=-0.02):
    handles = [Patch(facecolor=COLORS.get(m,'#999'), label=m,
                     edgecolor=COLORS[FOCAL] if m==FOCAL else '#555',
                     linewidth=2 if m==FOCAL else 0.5)
               for m in methods]
    fig.legend(handles=handles, loc='lower center',
               ncol=len(methods), fontsize=8,
               bbox_to_anchor=(0.5, y),
               title='Methods  (★ = DyGMamba)',
               title_fontsize=8)

######################################################################
# Figure 1: TF-Region  (4行 × N_CT列)
# Row0: F0.1 Top-50 Boxplot  Row1: F0.1 All Boxplot
# Row2: Recall Top-50 Boxplot Row3: Macro F0.1 Barplot + Macro Prec bar
######################################################################
print("绘制 Figure 1: TF-Region...")
N = len(avail_ct)
fig = plt.figure(figsize=(2.8*N, 22))
gs  = gridspec.GridSpec(5, N, figure=fig,
                        hspace=0.55, wspace=0.38)

row_configs = [
    # (data, plot_func, kwargs, row_label, ref_short)
    (tfr_fscore_all,  'box', {'top_n':50},     'F₀.₁ Top-50 Boxplot',   'SCENIC+, Nat Methods 2023'),
    (tfr_fscore_all,  'box', {'top_n':None},   'F₀.₁ All Boxplot',      'Dictys, Nat Methods 2023'),
    (tfr_recall_all,  'box', {'top_n':50},     'Recall Top-50 Boxplot', 'scMTNI, Nat Commun 2023'),
    (tfr_macro,       'bar', {},               'Macro F₀.₁ Barplot',    'GENIE3, PLOS ONE 2010'),
    (tfr_macro_prec,  'bar', {},               'Macro Precision Barplot','Wang et al, Nat Methods 2023'),
]

for row_i, (data, ptype, kwargs, row_label, ref) in enumerate(row_configs):
    for col_i, ct in enumerate(avail_ct):
        ax = fig.add_subplot(gs[row_i, col_i])

        ylabel = row_label.split(' ')[0] if col_i == 0 else ''

        if ptype == 'box':
            plot_boxplot_panel(ax, data, ct,
                               ylabel=ylabel, **kwargs)
        else:
            plot_bar_panel(ax, data, ct, ylabel=ylabel)

        if row_i == 0:
            ax.set_title(ct, fontsize=9, fontweight='bold', pad=4)
        else:
            ax.set_title(ct, fontsize=9, fontweight='bold', pad=2)

    # 行标签
    fig.text(-0.01, 1 - (row_i+0.5)/len(row_configs),
             row_label, ha='right', va='center',
             fontsize=8, fontweight='bold', rotation=90,
             transform=fig.transFigure)

fig.suptitle('TF-Region Benchmark: DyGMamba vs Competing Methods\n'
             'Each column = cell type  |  ★ DyGMamba highlighted  |  #N/M = rank',
             fontsize=11, fontweight='bold', y=1.01)

make_legend(fig, [m for m in METHOD_LIST
                  if any(m in d for d in tfr_fscore_all.values())], y=-0.01)

add_reference(fig,
    'References: [F₀.₁] Bravo González-Blas et al. Nature Methods 20:1355-1367 (2023) [SCENIC+]  |  '
    '[Recall] Wang et al. Nature Methods 20:1368-1378 (2023) [Dictys]  |  '
    '[Macro] Huynh-Thu et al. PLOS ONE 5:e12776 (2010) [GENIE3]', y=0.002)

plt.savefig(f"{output_dir}Fig1_TF_Region.png",
            dpi=180, bbox_inches='tight')
plt.close()
print("✓ Fig1_TF_Region.png")

######################################################################
# Figure 2: Region-Gene  (5行 × N_CT列)
# Row0: Spearman Corr Barplot   Row1: F0.1 All Barplot
# Row2: Recall All Barplot      Row3: F0.1 Top-50 Boxplot
# Row4: AUPRC Top-50 Boxplot
######################################################################
print("绘制 Figure 2: Region-Gene...")
fig = plt.figure(figsize=(2.8*N, 27))
gs  = gridspec.GridSpec(5, N, figure=fig,
                        hspace=0.55, wspace=0.38)

# Spearman 和 AUPRC 转换为 scalar（均值）
rg_spearman_mean = {
    ct: {m: float(np.mean(np.abs(v)))
         for m, v in d.items() if len(v)>0}
    for ct, d in rg_spearman.items()
}
rg_auprc_mean = {
    ct: {m: float(np.mean(v))
         for m, v in d.items() if len(v)>0}
    for ct, d in rg_auprc_per.items()
}

rg_row_configs = [
    (rg_spearman_mean,  'bar_scalar', {}, 'Spearman |ρ| Bar',    'Cicero, Mol Cell 2018'),
    (rg_fscore_per,     'bar_per',   {}, 'F₀.₁ All Barplot',   'FigR, Cell Genomics 2022'),
    (rg_recall_per,     'bar_per',   {}, 'Recall All Barplot',  'SHARE-seq, Cell 2020'),
    (rg_fscore_per,     'box',       {'top_n':50}, 'F₀.₁ Top-50 Boxplot', 'SCENIC+, Nat Methods 2023'),
    (rg_auprc_per,      'box',       {'top_n':50}, 'AUPRC Top-50 Boxplot','LINGER, Nat Biotechnol 2025'),
]

for row_i, (data, ptype, kwargs, row_label, ref) in enumerate(rg_row_configs):
    for col_i, ct in enumerate(avail_ct):
        ax = fig.add_subplot(gs[row_i, col_i])
        ylabel = row_label.split(' ')[0] if col_i == 0 else ''

        if ptype == 'bar_scalar':
            plot_bar_panel(ax, data, ct, ylabel=ylabel)
        elif ptype == 'bar_per':
            plot_bar_panel_from_per(ax, data, ct, ylabel=ylabel)
        elif ptype == 'box':
            plot_boxplot_panel(ax, data, ct, ylabel=ylabel, **kwargs)

        if row_i == 0:
            ax.set_title(ct, fontsize=9, fontweight='bold', pad=4)
        else:
            ax.set_title(ct, fontsize=9, fontweight='bold', pad=2)

fig.suptitle('Region-Gene Benchmark: DyGMamba vs Competing Methods',
             fontsize=11, fontweight='bold', y=1.01)
make_legend(fig, [m for m in METHOD_LIST
                  if any(m in d for d in rg_spearman_mean.values())], y=-0.01)
add_reference(fig,
    'References: [Spearman ρ] Pliner et al. Molecular Cell 71:858-871 (2018) [Cicero]  |  '
    '[F₀.₁/Recall] Kartha et al. Cell Genomics 2:100237 (2022) [FigR]  |  '
    '[AUPRC] Yuan & Duren Nature Biotechnology 43:247-257 (2025) [LINGER]', y=0.002)

plt.savefig(f"{output_dir}Fig2_Region_Gene.png",
            dpi=180, bbox_inches='tight')
plt.close()
print("✓ Fig2_Region_Gene.png")

######################################################################
# Figure 3: TF-Gene  (4行 × N_CT列)
# Row0: Correlation Top-50 Boxplot   Row1: Correlation Macro Bar
# Row2: F-score All Barplot          Row3: Precision All Barplot
######################################################################
print("绘制 Figure 3: TF-Gene...")
fig = plt.figure(figsize=(2.8*N, 22))
gs  = gridspec.GridSpec(4, N, figure=fig,
                        hspace=0.55, wspace=0.38)

# Macro correlation (mean per method per CT)
tfg_corr_macro = {
    ct: {m: float(np.mean(np.abs(v)))
         for m, v in d.items() if len(v)>0}
    for ct, d in tfg_corr_per.items()
}
# abs values
tfg_corr_abs = {
    ct: {m: np.abs(v) for m, v in d.items()}
    for ct, d in tfg_corr_per.items()
}

tfg_row_configs = [
    (tfg_corr_abs,    'box',      {'top_n':50}, 'Correlation Top-50 Box', 'CellOracle, Nature 2023'),
    (tfg_corr_macro,  'bar_scalar',{},           'Correlation Macro Bar',  'CellOracle, Nature 2023'),
    (tfg_fscore_per,  'bar_per',  {},            'F-score All Barplot',    'BEELINE, Nat Methods 2020'),
    (tfg_prec_per,    'bar_per',  {},            'Precision All Barplot',  'Omony, Brief Bioinform 2019'),
]

for row_i, (data, ptype, kwargs, row_label, ref) in enumerate(tfg_row_configs):
    for col_i, ct in enumerate(avail_ct):
        ax = fig.add_subplot(gs[row_i, col_i])
        ylabel = row_label.split(' ')[0] if col_i == 0 else ''

        if ptype == 'bar_scalar':
            plot_bar_panel(ax, data, ct, ylabel=ylabel)
        elif ptype == 'bar_per':
            plot_bar_panel_from_per(ax, data, ct, ylabel=ylabel)
        elif ptype == 'box':
            plot_boxplot_panel(ax, data, ct, ylabel=ylabel, **kwargs)

        if row_i == 0:
            ax.set_title(ct, fontsize=9, fontweight='bold', pad=4)
        else:
            ax.set_title(ct, fontsize=9, fontweight='bold', pad=2)

fig.suptitle('TF-Gene Benchmark: DyGMamba vs Competing Methods',
             fontsize=11, fontweight='bold', y=1.01)
make_legend(fig, [m for m in METHOD_LIST
                  if any(m in d for d in tfg_corr_macro.values())], y=-0.01)
add_reference(fig,
    'References: [Correlation] Kamimoto et al. Nature 614:742-751 (2023) [CellOracle]  |  '
    '[F-score] Pratapa et al. Nature Methods 17:147-154 (2020) [BEELINE]  |  '
    '[Precision] Omony et al. Brief Bioinformatics 20:812-823 (2019)', y=0.002)

plt.savefig(f"{output_dir}Fig3_TF_Gene.png",
            dpi=180, bbox_inches='tight')
plt.close()
print("✓ Fig3_TF_Gene.png")

######################################################################
# Figure 4: TF-Recovery  (1行 × N_CT列)
######################################################################
print("绘制 Figure 4: TF-Recovery...")
fig, axes = plt.subplots(1, N, figsize=(3.2*N, 5.5), sharey=False)
axes = np.array(axes).flatten()

for col_i, ct in enumerate(avail_ct):
    ax = axes[col_i]

    if ct not in tf_recovery:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                transform=ax.transAxes, color='#aaa', fontsize=9)
        ax.set_title(ct, fontsize=9, fontweight='bold'); continue

    methods_present = [m for m in METHOD_LIST if m in tf_recovery[ct]]
    auc_dict = {}

    for method in methods_present:
        vals = tf_recovery[ct][method]
        x    = np.arange(1, len(vals)+1)
        lw   = 2.5 if method == FOCAL else 1.2
        alpha= 1.0 if method == FOCAL else 0.6
        zo   = 10  if method == FOCAL else 1
        ax.plot(x, vals, color=COLORS.get(method,'#999'),
                linewidth=lw, alpha=alpha, zorder=zo,
                label=method)
        if len(vals) > 1:
            auc_dict[method] = float(np.trapz(vals, x))

    ax.set_title(ct, fontsize=9, fontweight='bold')
    ax.set_xlabel('Top-N Ranked TFs', fontsize=7.5)
    if col_i == 0:
        ax.set_ylabel('Cumulative TFs Recovered', fontsize=7.5)
    ax.spines[['top','right']].set_visible(False)

    # DyGMamba AUC vs best competitor
    if FOCAL in auc_dict and len(auc_dict) > 1:
        dyg_auc   = auc_dict[FOCAL]
        best_comp = max({m:v for m,v in auc_dict.items() if m!=FOCAL},
                        key=lambda m: auc_dict[m], default=None)
        if best_comp:
            delta_pct = (dyg_auc - auc_dict[best_comp]) / \
                        max(auc_dict[best_comp], 1e-9) * 100
            sign = '+' if delta_pct >= 0 else ''
            color = '#006400' if delta_pct >= 0 else '#8B0000'
            ax.text(0.97, 0.05,
                    f'AUC {sign}{delta_pct:.1f}%\nvs {best_comp[:6]}',
                    ha='right', va='bottom', transform=ax.transAxes,
                    fontsize=7, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor=color, alpha=0.8, lw=1.2))

    rank = sorted(auc_dict, key=auc_dict.get, reverse=True).index(FOCAL)+1 \
           if FOCAL in auc_dict else None
    if rank is not None:
        add_rank_badge(ax, rank, len(auc_dict))

fig.suptitle('TF-Recovery Benchmark: DyGMamba vs Competing Methods\n'
             '(AUC% = improvement over best competitor)',
             fontsize=11, fontweight='bold')
make_legend(fig, methods_present if tf_recovery else METHOD_LIST, y=-0.15)
add_reference(fig,
    'Reference: Bravo González-Blas et al. Nature Methods 20:1355-1367 (2023) [SCENIC+]  |  '
    'Huynh-Thu et al. PLOS ONE 5:e12776 (2010) [GENIE3]', y=0.0)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f"{output_dir}Fig4_TF_Recovery.png",
            dpi=180, bbox_inches='tight')
plt.close()
print("✓ Fig4_TF_Recovery.png")

######################################################################
# Figure 5: 综合排名热图 (所有指标 × 所有细胞系，DyGMamba 排名)
######################################################################
print("绘制 Figure 5: 综合排名热图...")

all_metrics = {
    'TF-Reg F₀.₁(Top50)':  ('box_top50', tfr_fscore_all),
    'TF-Reg F₀.₁(All)':    ('box_all',   tfr_fscore_all),
    'TF-Reg Recall(Top50)': ('box_top50', tfr_recall_all),
    'TF-Reg Macro F₀.₁':   ('scalar',    tfr_macro),
    'TF-Reg Macro Prec':    ('scalar',    tfr_macro_prec),
    'RG Spearman |ρ|':      ('scalar',    rg_spearman_mean),
    'RG F₀.₁(All)':         ('per_mean',  rg_fscore_per),
    'RG Recall(All)':        ('per_mean',  rg_recall_per),
    'RG F₀.₁(Top50)':       ('box_top50', rg_fscore_per),
    'RG AUPRC(Top50)':       ('box_top50', rg_auprc_per),
    'TFG Corr(Top50)':       ('box_top50', tfg_corr_abs),
    'TFG Corr Macro':        ('scalar',    tfg_corr_macro),
    'TFG F-score(All)':      ('per_mean',  tfg_fscore_per),
    'TFG Precision(All)':    ('per_mean',  tfg_prec_per),
}

def get_method_mean(mtype, data, ct):
    if ct not in data: return {}
    if mtype == 'scalar':
        return {m: v for m,v in data[ct].items()}
    elif mtype == 'box_top50':
        return {m: float(np.mean(np.sort(v)[::-1][:min(50,len(v))]))
                for m,v in data[ct].items() if len(v)>0}
    elif mtype in ('box_all','per_mean'):
        return {m: float(np.mean(v)) for m,v in data[ct].items() if len(v)>0}
    return {}

# 构建排名矩阵：行=指标，列=细胞系
rank_matrix = pd.DataFrame(index=list(all_metrics.keys()),
                            columns=avail_ct, dtype=float)
win_matrix  = pd.DataFrame(index=list(all_metrics.keys()),
                            columns=avail_ct, dtype=float)

for metric_name, (mtype, data) in all_metrics.items():
    for ct in avail_ct:
        means = get_method_mean(mtype, data, ct)
        if not means or FOCAL not in means: continue
        sorted_m = sorted(means, key=means.get, reverse=True)
        rank = sorted_m.index(FOCAL)+1 if FOCAL in sorted_m else len(sorted_m)
        rank_matrix.loc[metric_name, ct] = rank
        win_matrix.loc[metric_name, ct]  = 1 if rank == 1 else 0

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle('DyGMamba Comprehensive Ranking Summary\n'
             '(Left: rank across metrics/cell types | Right: win rate per metric)',
             fontsize=12, fontweight='bold')

# 左图：排名热图
rank_plot = rank_matrix.astype(float)
vmax = len(METHOD_LIST)
sns.heatmap(rank_plot, ax=axes[0],
            cmap='RdYlGn_r', vmin=1, vmax=vmax,
            annot=True, fmt='.0f', annot_kws={'size':8.5},
            linewidths=0.4, linecolor='white',
            cbar_kws={'label':'Rank (1=best)','shrink':0.75})
axes[0].set_title('DyGMamba Rank per Metric × Cell Type\n(1=best, green; worst=red)',
                  fontsize=10, fontweight='bold')
axes[0].tick_params(axis='x', rotation=40, labelsize=8.5)
axes[0].tick_params(axis='y', rotation=0, labelsize=8)
axes[0].set_xlabel('Cell Type', fontsize=9)
axes[0].set_ylabel('Metric', fontsize=9)

# 右图：胜率（排名第一的比例）× 每个指标
win_rates = win_matrix.astype(float).mean(axis=1).sort_values(ascending=True)
colors_bar = ['#006400' if v >= 0.5 else '#8B0000'
              for v in win_rates.values]
bars = axes[1].barh(np.arange(len(win_rates)), win_rates.values,
                    color=colors_bar, alpha=0.82,
                    edgecolor='white', linewidth=0.8)
axes[1].set_yticks(np.arange(len(win_rates)))
axes[1].set_yticklabels(win_rates.index, fontsize=8.5)
axes[1].axvline(x=0.5, color='gray', linestyle='--',
                linewidth=1.5, alpha=0.7, label='50% line')
axes[1].set_xlabel('Win Rate (rank #1 fraction across cell types)',
                   fontsize=9)
axes[1].set_title('DyGMamba Win Rate (Rank #1) per Metric\n(green ≥ 50%)',
                  fontsize=10, fontweight='bold')
for i, (name, val) in enumerate(win_rates.items()):
    axes[1].text(val + 0.02, i, f'{val:.0%}',
                 va='center', fontsize=8,
                 color='#006400' if val>=0.5 else '#8B0000',
                 fontweight='bold')
axes[1].set_xlim(0, 1.15)
axes[1].legend(fontsize=8)
axes[1].spines[['top','right']].set_visible(False)

# 总体统计
total_rank1 = int(win_matrix.astype(float).values.sum())
total_cells  = int((~rank_matrix.isna()).values.sum())
fig.text(0.5, 0.0,
         f'DyGMamba achieved Rank #1 in {total_rank1}/{total_cells} '
         f'metric×cell-type combinations  ({total_rank1/max(total_cells,1)*100:.1f}%)',
         ha='center', fontsize=9, fontweight='bold', color=COLORS[FOCAL])

plt.tight_layout(rect=[0,0.04,1,1])
plt.savefig(f"{output_dir}Fig5_Ranking_Summary.png",
            dpi=180, bbox_inches='tight')
plt.close()
print("✓ Fig5_Ranking_Summary.png")

######################################################################
# 输出数字摘要
######################################################################
print(f"\n{'='*60}")
print("DyGMamba 数字摘要")
print(f"{'='*60}")
print(f"总体: Rank #1 in {total_rank1}/{total_cells} "
      f"metric×CT combinations ({total_rank1/max(total_cells,1)*100:.1f}%)")
print("\n指标维度 (win rate ≥ 50% 的指标):")
for name, wr in win_rates.sort_values(ascending=False).items():
    if wr >= 0.5:
        print(f"  ✓ {name:30s}: {wr:.0%}")
print("\n参考文献:")
refs = [
    "[TF-Region F₀.₁]  Bravo González-Blas et al. Nature Methods 20:1355-1367 (2023) — SCENIC+",
    "[TF-Region Recall] Wang et al. Nature Methods 20:1368-1378 (2023) — Dictys",
    "[TF-Region Macro]  Huynh-Thu et al. PLOS ONE 5:e12776 (2010) — GENIE3",
    "[Region-Gene ρ]    Pliner et al. Molecular Cell 71:858-871 (2018) — Cicero",
    "[Region-Gene F₀.₁] Kartha et al. Cell Genomics 2:100237 (2022) — FigR",
    "[Region-Gene AUPRC] Yuan & Duren Nature Biotechnology 43:247-257 (2025) — LINGER",
    "[TF-Gene Corr]     Kamimoto et al. Nature 614:742-751 (2023) — CellOracle",
    "[TF-Gene F-score]  Pratapa et al. Nature Methods 17:147-154 (2020) — BEELINE",
    "[TF-Recovery]      Bravo González-Blas et al. Nature Methods 20:1355-1367 (2023)",
]
for r in refs:
    print(f"  {r}")

print(f"\n图已保存到: {output_dir}")