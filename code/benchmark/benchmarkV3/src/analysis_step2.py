"""
paper_benchmark_final_v5.py

结构:
  OUTPUT_DIR/
    TF_Region/
      01_F01_Top50_Box.pdf
      02_F01_All_Box.pdf
      03_Recall_Top50_Box.pdf
      04_Macro_F01_Bar.pdf
    Region_Gene/
      01_Spearman_Top50_Box.pdf
      ...
    TF_Gene/
      ...
    TF_Recovery/
      01_Recovery_Curve.pdf
      02_AUC_Bar.pdf
    Summary/
      Main_Figure_Combined.pdf
"""
import os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════
# 用户可控参数（只需改这里）
# ═════════════════════════════════════════════════════

DATA_ROOT  = "/home/wuyan/dygmamba_project/data/cell_line/"
OUTPUT_DIR = "/home/wuyan/dygmamba_project/data/benchmark_drima/"

CELL_TYPES  = ["GM12878", "HepG2", "IMR90", "K562",
               "MCF7",   "A549",  "H1",   "HELA",  "SK"]
METHOD_LIST = ["DyGMamba", "CellOracle", "FigR",
               "GLUE",     "LINGER",     "GRaNIE", "Pando"]
FOCAL       = "DyGMamba"
TOP_N       = 50
MAX_COLS    = 5       # 每行最多子图数

# ── 单张图的宽高（用户自行控制）──
SINGLE_FIG_W = 14.0   # 单张图总宽度 (inch)
SINGLE_FIG_H = 5.0    # 单张图每行高度 (inch)，多行时自动×行数

# ── 汇总图的宽高 ──
SUMMARY_FIG_W = 16.0
SUMMARY_ROW_H = 2.8   # 汇总图中每个小图的行高

# ── 字体 ──
FONT_SIZE = 16        # 小四号 12pt

COLORS = {
    "DyGMamba":   "#7B2D8B",
    "CellOracle": "#E87040",
    "FigR":       "#2E4A7C",
    "GLUE":       "#009B77",
    "LINGER":     "#6B7DC0",
    "GRaNIE":     "#5BAD92",
    "Pando":      "#C0392B",
}
NO_DATA_COLOR = "#DCDCDC"

# ═════════════════════════════════════════════════════
# 全局样式
# ═════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        FONT_SIZE,
    'axes.titlesize':   FONT_SIZE,
    'axes.labelsize':   FONT_SIZE,
    'xtick.labelsize':  FONT_SIZE - 1,
    'ytick.labelsize':  FONT_SIZE - 1,
    'legend.fontsize':  FONT_SIZE - 1,
    'axes.linewidth':   0.6,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'pdf.fonttype':     42,
})

# ═════════════════════════════════════════════════════
# 布局计算
# ═════════════════════════════════════════════════════
N = len(CELL_TYPES)
N_CT_ROWS = math.ceil(N / MAX_COLS)
CT_ROWS = []
for r in range(N_CT_ROWS):
    s = r * MAX_COLS
    CT_ROWS.append(CELL_TYPES[s : s + min(MAX_COLS, N - s)])

print(f"布局: {N} cell types → {N_CT_ROWS} rows × max {MAX_COLS} cols")
for i, row in enumerate(CT_ROWS):
    print(f"  Row {i}: {row}")


# ═════════════════════════════════════════════════════
# 数据读取
# ═════════════════════════════════════════════════════
def _read_excel(ct, sheet):
    path = f"{DATA_ROOT}{ct}/benchmarkV7/benchmark_all_results.xlsx"
    if not os.path.exists(path):
        return None
    try:
        xl = pd.ExcelFile(path)
        if sheet not in xl.sheet_names:
            return None
        df = xl.parse(sheet)
        if 'Method' in df.columns:
            df['Method'] = df['Method'].astype(str).str.strip()
        return df
    except:
        return None

def load_per(sheet, col):
    result = {}
    for ct in CELL_TYPES:
        df = _read_excel(ct, sheet)
        d = {}
        for m in METHOD_LIST:
            if df is not None and 'Method' in df.columns and col in df.columns:
                v = df[df['Method'] == m][col].dropna().values.astype(float)
            else:
                v = np.array([])
            d[m] = v
        if any(len(v) > 0 for v in d.values()):
            result[ct] = d
    return result

def load_scalar(sheet, col):
    result = {}
    for ct in CELL_TYPES:
        df = _read_excel(ct, sheet)
        d = {}
        for m in METHOD_LIST:
            if df is not None and 'Method' in df.columns and col in df.columns:
                sub = df[df['Method'] == m][col].dropna()
                d[m] = float(sub.iloc[0]) if len(sub) > 0 else None
            else:
                d[m] = None
        if any(v is not None for v in d.values()):
            result[ct] = d
    return result

def reconstruct_recovery(ct, top_n=TOP_N):
    set_df = _read_excel(ct, 'TF_recovery_set')
    pkl = f"{DATA_ROOT}{ct}/benchmarkV7/count_region_df.pkl"
    if set_df is None or not os.path.exists(pkl):
        return {m: None for m in METHOD_LIST}
    try:
        cdf = pd.read_pickle(pkl)
        tc = 'TF' if 'TF' in cdf.columns else cdf.columns[0]
        pc = 'PeakCount' if 'PeakCount' in cdf.columns else cdf.columns[-1]
        gt = cdf.sort_values(pc, ascending=False)[tc].dropna().tolist()
    except:
        return {m: None for m in METHOD_LIST}
    mr = min(top_n, len(gt))
    curves = {}
    for m in METHOD_LIST:
        if m not in set_df.columns:
            curves[m] = None
            continue
        tfs = set(set_df[m].dropna().astype(str).tolist())
        if not tfs:
            curves[m] = None
            continue
        x = np.arange(1, mr + 1, dtype=float)
        y = np.zeros(mr)
        cum = 0
        for i in range(mr):
            if gt[i] in tfs:
                cum += 1
            y[i] = cum
        curves[m] = (x, y, float(np.trapz(y, x)) / mr)
    return curves

print("加载数据...")
TFR_FSCORE  = load_per('TF_region_per', 'fscore')
TFR_RECALL  = load_per('TF_region_per', 'Recall')
TFR_MACRO_F = load_scalar('TF_region_all', 'F_score')
RG_SPEARMAN = load_per('Region_gene_per_corr', 'Abs_Spearman_Rho')
RG_FSCORE   = load_per('Region_gene_per_precision', 'F_score')
RG_RECALL   = load_per('Region_gene_per_precision', 'Recall')
TFG_CORR    = load_per('TF_gene_per_corr', 'Correlation')
TFG_FSCORE  = load_per('TF_gene_per_precision', 'F_score')
TFG_PREC    = load_per('TF_gene_per_precision', 'Precision')

for _d in [TFR_FSCORE, TFR_RECALL, RG_SPEARMAN, TFG_CORR]:
    for ct in _d:
        _d[ct] = {m: np.abs(v) for m, v in _d[ct].items()}

TF_REC_CURVES = {}
for ct in CELL_TYPES:
    c = reconstruct_recovery(ct)
    if any(v is not None for v in c.values()):
        TF_REC_CURVES[ct] = c

TF_REC_AUC = {}
for ct in CELL_TYPES:
    if ct not in TF_REC_CURVES:
        continue
    d = {}
    for m in METHOD_LIST:
        c = TF_REC_CURVES[ct].get(m)
        d[m] = float(c[2]) if c is not None else None
    TF_REC_AUC[ct] = d

print("数据加载完成")




FONT_SIZE = 16        # 小四号 12pt
# ── 单张图的宽高（用户自行控制）──
SINGLE_FIG_W = 14.0   # 单张图总宽度 (inch)
SINGLE_FIG_H = 4.0    # 单张图每行高度 (inch)，多行时自动×行数
# ═════════════════════════════════════════════════════
# 绘图基础函数
# ═════════════════════════════════════════════════════

def _clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, pad=2)

def _swatches(ax):
    """x 轴下方色块"""
    for i, m in enumerate(METHOD_LIST):
        rect = FancyBboxPatch(
            (i - 0.39, -0.1), 0.78, 0.07,
            boxstyle="round,pad=0.01",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            facecolor=COLORS.get(m, '#999'),
            edgecolor='white', linewidth=0.3, zorder=5)
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.tick_params(bottom=False)

def _draw_bar(ax, scalar_dict, ct, ylabel=''):
    ax.set_xlim(-0.5, len(METHOD_LIST) - 0.5)
    vals, fc, ec, lw = [], [], [], []
    has = False
    for m in METHOD_LIST:
        v = scalar_dict.get(ct, {}).get(m, None)
        if v is not None and not np.isnan(float(v)):
            vals.append(float(v))
            fc.append(COLORS.get(m, '#999'))
            has = True
        else:
            vals.append(0.0)
            fc.append(NO_DATA_COLOR)
        ec.append(COLORS[FOCAL] if m == FOCAL else 'white')
        lw.append(1.5 if m == FOCAL else 0.2)
    if not has:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                transform=ax.transAxes, color='#bbb')
        _swatches(ax); _clean(ax)
        if ylabel: ax.set_ylabel(ylabel)
        return
    vm = max(v for v in vals if v > 0) if any(v > 0 for v in vals) else 1
    x = np.arange(len(METHOD_LIST))
    ax.bar(x, vals, color=fc, edgecolor=ec, linewidth=lw,
           width=0.70, alpha=0.90, zorder=3)
    fi = METHOD_LIST.index(FOCAL)
    ax.axvspan(fi - 0.42, fi + 0.42, alpha=0.07, color=COLORS[FOCAL], zorder=0)
    ax.set_ylim(0, vm * 1.30)
    _swatches(ax); _clean(ax)
    if ylabel: ax.set_ylabel(ylabel)

def _draw_bar_per(ax, per_dict, ct, ylabel='', top_n=None):
    scalar = {}
    if ct in per_dict:
        for m in METHOD_LIST:
            v = per_dict[ct].get(m, np.array([]))
            if len(v) > 0:
                sv = np.sort(v)[::-1][:min(top_n, len(v))] if top_n else v
                scalar[m] = float(np.mean(sv))
    _draw_bar(ax, {ct: scalar}, ct, ylabel=ylabel)

def _draw_box(ax, per_dict, ct, ylabel='', top_n=None):
    ax.set_xlim(-0.5, len(METHOD_LIST) - 0.5)
    has = False
    for i, m in enumerate(METHOD_LIST):
        v = per_dict.get(ct, {}).get(m, np.array([]))
        if top_n and len(v) > 0:
            v = np.sort(v)[::-1][:min(top_n, len(v))]
        if len(v) == 0:
            ax.plot([i - 0.28, i + 0.28], [0, 0],
                    color='#C8C8C8', linewidth=0.6, zorder=1)
            continue
        has = True
        bp = ax.boxplot(
            v, positions=[i], widths=0.58,
            patch_artist=True, manage_ticks=False,
            whiskerprops={'linewidth': 0.6, 'color': '#555'},
            capprops={'linewidth': 0.6, 'color': '#555'},
            medianprops={'color': 'white', 'linewidth': 1.2},
            flierprops={'marker': 'o', 'markersize': 2,
                        'markerfacecolor': '#999', 'alpha': 0.35, 'linewidth': 0},
            boxprops={'linewidth': 0.6})
        for p in bp['boxes']:
            p.set_facecolor(COLORS.get(m, '#999'))
            p.set_alpha(0.88)
            if m == FOCAL:
                p.set_edgecolor(COLORS[FOCAL])
                p.set_linewidth(1.6)
    if not has:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                transform=ax.transAxes, color='#bbb')
    fi = METHOD_LIST.index(FOCAL)
    ax.axvspan(fi - 0.42, fi + 0.42, alpha=0.07, color=COLORS[FOCAL], zorder=0)
    _swatches(ax); _clean(ax)
    if ylabel: ax.set_ylabel(ylabel)

def _draw_recovery(ax, ct, ylabel=''):
    has = False
    for m in METHOD_LIST:
        c = TF_REC_CURVES.get(ct, {}).get(m, None)
        if c is None:
            continue
        x, y, _ = c
        has = True
        isf = (m == FOCAL)
        ax.plot(x, y, color=COLORS.get(m, '#999'),
                linewidth=2.0 if isf else 0.8,
                alpha=1.0 if isf else 0.50,
                zorder=10 if isf else 1)
        if isf:
            ax.fill_between(x, y, alpha=0.09, color=COLORS[FOCAL], zorder=0)
    if not has:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                transform=ax.transAxes, color='#bbb')
    ax.set_xlim(0, TOP_N)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('TF Rank')
    _clean(ax)
    if ylabel: ax.set_ylabel(ylabel)

def _add_legend(fig, style='patch', y_pos=0.02):
    if style == 'patch':
        hs = [Patch(facecolor=COLORS.get(m, '#999'), label=m,
                     edgecolor=COLORS[FOCAL] if m == FOCAL else 'none',
                     linewidth=1.5 if m == FOCAL else 0)
              for m in METHOD_LIST]
        hs.append(Patch(facecolor=NO_DATA_COLOR, label='No data', edgecolor='none'))
    else:
        hs = [plt.Line2D([0], [0], color=COLORS.get(m, '#999'),
                         linewidth=2.0 if m == FOCAL else 0.9, label=m)
              for m in METHOD_LIST]
        hs.append(plt.Line2D([0], [0], color='#C8C8C8', linewidth=0.6, label='No data'))

    fig.legend(handles=hs, loc='lower center',
               ncol=len(hs), fontsize=FONT_SIZE - 1,
               bbox_to_anchor=(0.5, y_pos),
               frameon=True, framealpha=0.92, edgecolor='#ccc',
               handlelength=1.2, handletextpad=0.4, columnspacing=0.8)


# ═════════════════════════════════════════════════════
# 核心：单张图生成函数
# ═════════════════════════════════════════════════════

def save_single_figure(title, plot_type, data, top_n, save_path, ylabel=''):
    """
    生成一张图，包含所有 cell types，按 MAX_COLS 换行
    plot_type: 'box' / 'bar_sc' / 'bar_per' / 'recovery'
    """
    fig_w = SINGLE_FIG_W
    fig_h = SINGLE_FIG_H * N_CT_ROWS

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        N_CT_ROWS, MAX_COLS, figure=fig,
        left=0.1, right=0.9, top=0.9, bottom=0.1,
        hspace=0.25, wspace=0.25)

    for ct_row_i, ct_list in enumerate(CT_ROWS):
        for col_i, ct in enumerate(ct_list):
            ax = fig.add_subplot(gs[ct_row_i, col_i])
            ax.set_title(ct, fontsize=FONT_SIZE, fontweight='bold', pad=5)

            yl = ylabel if col_i == 0 else ''

            if plot_type == 'box':
                _draw_box(ax, data, ct, ylabel=yl, top_n=top_n)
            elif plot_type == 'bar_sc':
                _draw_bar(ax, data, ct, ylabel=yl)
            elif plot_type == 'bar_per':
                _draw_bar_per(ax, data, ct, ylabel=yl, top_n=top_n)
            elif plot_type == 'recovery':
                _draw_recovery(ax, ct, ylabel=yl)

        # 关闭空格子
        for col_i in range(len(ct_list), MAX_COLS):
            fig.add_subplot(gs[ct_row_i, col_i]).axis('off')

    legend_style = 'line' if plot_type == 'recovery' else 'patch'
    _add_legend(fig, style=legend_style, y_pos=-0.02)
    fig.suptitle(title, fontsize=FONT_SIZE + 2, fontweight='bold', y=0.99)

    for ext in ('pdf', 'png'):
        fig.savefig(f"{save_path}.{ext}", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ {os.path.basename(save_path)}")

    return save_path



# ═════════════════════════════════════════════════════
# 按层级生成所有单张图
# ═════════════════════════════════════════════════════

ALL_FIGURES = {}  # {layer: [(title, path), ...]}

LAYERS = {
    'TF_Region': [
        ('F₀.₁ Top-50 (Boxplot)',     'box',     TFR_FSCORE,  TOP_N, 'F₀.₁'),
        ('F₀.₁ All (Boxplot)',         'box',     TFR_FSCORE,  None,  'F₀.₁'),
        ('Recall Top-50 (Boxplot)',    'box',     TFR_RECALL,  TOP_N, 'Recall'),
        ('Macro F₀.₁ (Barplot)',       'bar_sc',  TFR_MACRO_F, None,  'F₀.₁'),
    ],
    'Region_Gene': [
        ('|ρ| Top-50 (Boxplot)',       'box',     RG_SPEARMAN, TOP_N, '|Spearman ρ|'),
        ('|ρ| All (Boxplot)',          'box',     RG_SPEARMAN, None,  '|Spearman ρ|'),
        ('|ρ| Top-50 (Barplot)',       'bar_per', RG_SPEARMAN, TOP_N, 'Mean |ρ|'),
        ('|ρ| All (Barplot)',          'bar_per', RG_SPEARMAN, None,  'Mean |ρ|'),
        ('F₀.₁ All (Barplot)',         'bar_per', RG_FSCORE,   None,  'F₀.₁'),
        ('Recall All (Barplot)',       'bar_per', RG_RECALL,   None,  'Recall'),
    ],
    'TF_Gene': [
        ('|Correlation| Top-50 (Box)', 'box',     TFG_CORR,   TOP_N, '|Correlation|'),
        ('|Correlation| All (Box)',    'box',     TFG_CORR,   None,  '|Correlation|'),
        ('|Correlation| Top-50 (Bar)', 'bar_per', TFG_CORR,   TOP_N, 'Mean |Corr|'),
        ('|Correlation| All (Bar)',    'bar_per', TFG_CORR,   None,  'Mean |Corr|'),
        ('F-score All (Barplot)',      'bar_per', TFG_FSCORE, None,  'F-score'),
        ('Precision All (Barplot)',    'bar_per', TFG_PREC,   None,  'Precision'),
    ],
    'TF_Recovery': [
        ('Recovery Curve',             'recovery', None,        None,  'TFs Recovered'),
        ('AUC / Rank (Barplot)',       'bar_sc',  TF_REC_AUC,  None,  'AUC / Rank'),
    ],
}

print("\n" + "=" * 60)
print("生成单张图（按层级分文件夹）")
print("=" * 60)

for layer_name, specs in LAYERS.items():
    layer_dir = os.path.join(OUTPUT_DIR, layer_name)
    os.makedirs(layer_dir, exist_ok=True)
    ALL_FIGURES[layer_name] = []

    print(f"\n  [{layer_name}]")
    for idx, (title, ptype, data, tn, yl) in enumerate(specs):
        # 文件名
        safe_title = title.replace(' ', '_').replace('|', '').replace('₀', '0').replace('₁', '1')
        safe_title = safe_title.replace('(', '').replace(')', '').replace('ρ', 'rho')
        safe_title = safe_title.replace("/", '')
        fname = f"{idx+1:02d}_{safe_title}"
        fpath = os.path.join(layer_dir, fname)

        full_title = f"{layer_name.replace('_', '-')} — {title}"
        save_single_figure(full_title, ptype, data, tn, fpath, ylabel=yl)
        ALL_FIGURES[layer_name].append((title, fpath))
        




# ═════════════════════════════════════════════════════
# 汇总大图：所有层级合并
# ═════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("生成汇总大图")
print("=" * 60)

# 计算总行数
total_metric_rows = sum(len(specs) for specs in LAYERS.values())
total_grid_rows = total_metric_rows * N_CT_ROWS

fig_w = SUMMARY_FIG_W
fig_h = total_grid_rows * SUMMARY_ROW_H + 3.0

fig = plt.figure(figsize=(fig_w, fig_h))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    total_grid_rows, MAX_COLS, figure=fig,
    left=0.08, right=0.97, top=0.97, bottom=0.03,
    hspace=0.70, wspace=0.30)

PANEL_LABELS = ['A', 'B', 'C', 'D']
current_row = 0

for layer_idx, (layer_name, specs) in enumerate(LAYERS.items()):
    # 标记此 layer 的起始行
    layer_start_row = current_row

    for metric_idx, (title, ptype, data, tn, yl) in enumerate(specs):
        for ct_row_i, ct_list in enumerate(CT_ROWS):
            grid_row = current_row

            for col_i, ct in enumerate(ct_list):
                ax = fig.add_subplot(gs[grid_row, col_i])

                # 标题：每个 metric 的第一个 ct_row 显示
                if ct_row_i == 0:
                    ax.set_title(ct, fontsize=FONT_SIZE - 2, fontweight='bold', pad=3)
                else:
                    ax.set_title(ct, fontsize=FONT_SIZE - 3, pad=2)

                ylabel = yl if col_i == 0 and ct_row_i == 0 else ''

                if ptype == 'box':
                    _draw_box(ax, data, ct, ylabel=ylabel, top_n=tn)
                elif ptype == 'bar_sc':
                    _draw_bar(ax, data, ct, ylabel=ylabel)
                elif ptype == 'bar_per':
                    _draw_bar_per(ax, data, ct, ylabel=ylabel, top_n=tn)
                elif ptype == 'recovery':
                    _draw_recovery(ax, ct, ylabel=ylabel)

            # 空格子
            for col_i in range(len(ct_list), MAX_COLS):
                fig.add_subplot(gs[grid_row, col_i]).axis('off')

            current_row += 1

    # Panel 字母标签
    first_ax_pos = fig.add_subplot(gs[layer_start_row, 0]).get_position()
    fig.add_subplot(gs[layer_start_row, 0]).remove()
    # 重新获取位置
    tmp = fig.add_subplot(gs[layer_start_row, 0])
    pos = tmp.get_position()
    tmp.remove()

    fig.text(0.005, pos.y0 + pos.height,
             PANEL_LABELS[layer_idx],
             ha='left', va='top',
             fontsize=FONT_SIZE + 6, fontweight='bold',
             transform=fig.transFigure)

    # 层名标签（左侧竖排）
    last_row = current_row - 1
    tmp2 = fig.add_subplot(gs[last_row, 0])
    pos2 = tmp2.get_position()
    tmp2.remove()
    y_center = (pos.y0 + pos.height + pos2.y0) / 2

    fig.text(0.02, y_center,
             layer_name.replace('_', '-'),
             ha='center', va='center',
             fontsize=FONT_SIZE, fontweight='bold',
             rotation=90, color='#333',
             transform=fig.transFigure)

# 图例
_add_legend(fig, style='patch', y_pos=0.005)

summary_dir = os.path.join(OUTPUT_DIR, 'Summary')
os.makedirs(summary_dir, exist_ok=True)
summary_path = os.path.join(summary_dir, 'Main_Figure_Combined')

for ext in ('pdf', 'png'):
    fig.savefig(f"{summary_path}.{ext}", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ 汇总图: {summary_path}")


# ═════════════════════════════════════════════════════
# 完成
# ═════════════════════════════════════════════════════
print(f"\n{'━' * 60}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"目录结构:")
for layer_name in LAYERS:
    figs = ALL_FIGURES[layer_name]
    print(f"  {layer_name}/")
    for title, path in figs:
        print(f"    ✓ {os.path.basename(path)}.pdf")
print(f"  Summary/")
print(f"    ✓ Main_Figure_Combined.pdf")
print(f"\n设置: MAX_COLS={MAX_COLS}, 字体={FONT_SIZE}pt, "
      f"单图={SINGLE_FIG_W}×{SINGLE_FIG_H}in")
print(f"{'━' * 60}")