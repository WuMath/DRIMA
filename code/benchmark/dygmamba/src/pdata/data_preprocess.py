import numpy as np
import anndata as ad
import pandas as pd
import pickle
import scanpy as sc
import pyranges as pr
import anndata as ad
import scipy.sparse as sp_sparse
from scipy import sparse
import pybedtools
from tqdm import tqdm
from sklearn.neighbors import RadiusNeighborsTransformer

import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import muon.atac as ma  # 用于 TF-IDF


def find_zero_sum_elements(adata):
    """
    统计 AnnData 中行和为 0 的 Gene (obs) 和列和为 0 的 Peak (var)
    """
    
    # -------------------------------------------------------
    # 1. 计算和 (兼容稀疏矩阵和稠密矩阵)
    # -------------------------------------------------------
    # axis=0: 对列求和 -> 得到每个 Peak (var) 的总调控数
    # axis=1: 对行求和 -> 得到每个 Gene (obs) 的总调控数
    
    if sp_sparse.issparse(adata.X):
        # 稀疏矩阵 sum 后返回的是 np.matrix 对象 (二维)，需要转为一维数组
        peak_sums = np.array(adata.X.sum(axis=0)).flatten()
        gene_sums = np.array(adata.X.sum(axis=1)).flatten()
    else:
        # 稠密矩阵直接求和
        peak_sums = np.sum(adata.X, axis=0)
        gene_sums = np.sum(adata.X, axis=1)

    # -------------------------------------------------------
    # 2. 提取名称
    # -------------------------------------------------------
    # 找出和为 0 的索引对应的名称
    zero_peaks = adata.var_names[peak_sums == 0].tolist()
    zero_genes = adata.obs_names[gene_sums == 0].tolist()
    
    return zero_peaks, zero_genes





def adata_to_dataframe(adata):
    
    if sp.issparse(adata.X):
        coo_matrix = adata.X.tocoo()
    else:
        coo_matrix = sp.coo_matrix(adata.X)
    

    # 创建一个 DataFrame 来存储 TF-Peak 的连接
    df = pd.DataFrame({
        'obs': adata.obs_names[coo_matrix.row],
        'var': adata.var_names[coo_matrix.col],
        'value': coo_matrix.data
    })
    
    return df



####################################################################################################
#***************************----------------------------------------*******************************#
#***************************   JASPAR TF peak 数据预处理函数   *******************************#
#***************************----------------------------------------*******************************#


def filter_jaspar_tf(jaspar_tf_peak, score_threshold=0):
    '''
    input data: 
        jaspar_tf_peak: is a anndata with tf as vars, peak as obs
    '''
    if sp_sparse.issparse(jaspar_tf_peak.X):
        nan_mask = np.isnan(jaspar_tf_peak.X.data)
        if np.any(nan_mask):
            print(f"  > 发现了 {np.sum(nan_mask)} 个 NaN 值, 将其设为 0 ...")
            jaspar_tf_peak.X.data[nan_mask] = 0.0
    else:
        if np.any(np.isnan(jaspar_tf_peak.X)):
            print(f"  > 发现了 NaN 值, 将其设为 0 ...")
            jaspar_tf_peak.X = np.nan_to_num(jaspar_tf_peak.X, nan=0.0)

    jaspar_tf_peak.X = (jaspar_tf_peak.X > score_threshold).astype(int)
    jaspar_tf_peak.X = sp_sparse.csr_matrix(jaspar_tf_peak.X)

    # ************* 1. Peak Filter (Rows) **************************

    peak_counts = jaspar_tf_peak.X.getnnz(axis=1)
    keep_peaks_mask = peak_counts > 0

    print(f"\n步骤 1: 过滤 Peaks (行)")
    print(f"  > 找到 {np.sum(keep_peaks_mask)} / {jaspar_tf_peak.n_obs} 个 peaks 至少有 1 个 TF 结合。")
    jaspar_tf_peak = jaspar_tf_peak[keep_peaks_mask, :].copy()

    # ************* 2. TF Filter (Columns) **************************

    tf_counts = jaspar_tf_peak.X.getnnz(axis=0)
    keep_tfs_mask = tf_counts > 0

    print(f"\n步骤 2: 过滤 TFs (列)")
    print(f"  > 找到 {np.sum(keep_tfs_mask)} / {jaspar_tf_peak.n_vars} 个 TFs 至少结合 1 个 peak。")

    jaspar_tf_peak = jaspar_tf_peak[:, keep_tfs_mask].copy()

    print(f"  > 最终形状: {jaspar_tf_peak.shape}")

    return jaspar_tf_peak


def dataframe_to_anndata_sparse(df, obs_col='sequence_name', var_col='TF_Symbol', value_col=None):
    """
    将长格式 DataFrame 转换为 AnnData (Obs=Region, Var=TF)。
    
    参数:
    value_col: 如果为 None，则生成 0/1 二值矩阵（表示有无结合）。
               如果是 'score'，则矩阵中填充 FIMO 的分数。
    """
    print("正在转换数据格式...")
    
    # 1. 将字符串转换为 Categorical 类型，获取整数索引
    # 这是最快的方法，避免了手动建立字典映射
    obs_cat = df[obs_col].astype('category')
    var_cat = df[var_col].astype('category')
    
    row_indices = obs_cat.cat.codes
    col_indices = var_cat.cat.codes
    
    # 2. 准备矩阵的填充值 (Data)
    if value_col and value_col in df.columns:
        # 如果指定了分数列，就用分数填充
        data = df[value_col].values
    else:
        # 否则填充 1 (表示存在)
        data = np.ones(len(df), dtype=np.float32)
        
    # 3. 构建稀疏矩阵 (COO 格式 -> CSR 格式)
    # shape = (Regions数量, TFs数量)
    n_obs = len(obs_cat.cat.categories)
    n_vars = len(var_cat.cat.categories)
    
    # 注意：如果同一个 TF 在同一个 Peak 里出现多次（FIMO常见情况），
    # 稀疏矩阵转换时默认会把它们相加 (sum)。
    # 如果你只想要“有/无” (Binary)，后续需要处理一下。
    sparse_mat = sparse.coo_matrix((data, (row_indices, col_indices)), 
                                   shape=(n_obs, n_vars)).tocsr()
    
    # 如果是二值矩阵，确保最大值是 1 (处理重复 hit)
    if value_col is None:
        sparse_mat.data = np.where(sparse_mat.data > 0, 1, 0)

    # 4. 创建 AnnData
    adata = ad.AnnData(X=sparse_mat)
    
    # 5. 赋值索引名称
    adata.obs_names = obs_cat.cat.categories # Regions
    adata.var_names = var_cat.cat.categories # TFs
    
    print(f"转换完成！AnnData shape: {adata.shape}")
    print(f"Obs (Regions): {adata.n_obs}, Var (TFs): {adata.n_vars}")
    
    return adata

####################################################################################################
#***************************----------------------------------------*******************************#
#***************************   ATAC peak 数据预处理函数   *******************************#
#***************************----------------------------------------*******************************#
####################################################################################################


def filter_atac_base(adata_atac):
    """
    参数:
    adata: AnnData 对象 (行=细胞, 列=Peak, var_names格式如 "chr1:100-200")
    tf_df: TF ChIP-seq 的 DataFrame (必须包含列: [chrom, start, end, peak_name])
    gene_df: 基因位置的 DataFrame (必须包含列: [chrom, start, end, gene_name])
    gene_window: 定义结合的距离窗口
    """
    
    print(f"初始 Peak 数量: {adata_atac.n_vars}")

    # (1) 选择标准染色体上的 Peaks
    standard_chroms = set([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'])
    pattern = r'^(?P<chrom>.*?)[:\-](\d+)-(\d+)$'

    extracted_df = adata_atac.var_names.to_series().str.extract(pattern)

    chr_mask = extracted_df['chrom'].isin(standard_chroms)

    # standard_chroms = tuple([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'])
    # peak_chroms = adata_atac.var_names.to_series().str.split(r'[:\-]', expand=True)[0]
    # chr_mask = peak_chroms.isin(standard_chroms)

    adata_atac = adata_atac[:, chr_mask].copy()

    # (2) 依据consencus peak 计数过滤

    if hasattr(adata_atac.X, "tocsr"):
        # 稀疏矩阵求和后是一个 matrix 对象，需要展平
        peak_sums = np.array(adata_atac.X.sum(axis=0)).flatten()
    else:
        peak_sums = adata_atac.X.sum(axis=0)

    expr_mask = peak_sums > 0

    adata_atac = adata_atac[:, expr_mask].copy()

    return adata_atac

# ==========================================
# 工具函数: 提取坐标
# ==========================================

def get_atac_bed(adata):
    """
    从 AnnData 提取 Peak 坐标并转换为 PyRanges 对象
    支持格式: 
      1. chr1:100-200 (标准)
      2. chr1-100-200 (你的需求)
      3. chr1_100_200 (常见变体)
    """
    df = None

    # --- 情况 1: 从列中读取 (如果 metadata 已经有坐标) ---
    # 检查常见的列名变体
    if 'chrom' in adata.var.columns and 'start' in adata.var.columns:
        df = adata.var[['chrom', 'start', 'end']].copy()

    elif 'Chromosome' in adata.var.columns and 'Start' in adata.var.columns:
        df = adata.var[['Chromosome', 'Start', 'End']].copy()

    elif 'chr' in adata.var.columns and 'start' in adata.var.columns:
        df = adata.var[['chr', 'start', 'end']].copy()
        df.columns = ['Chromosome', 'Start', 'End']
    # --- 情况 2: 从索引 (var_names) 解析 ---
    else:
        df = pd.DataFrame(index=adata.var_names)
        try:
            # 【核心修改】使用正则表达式分割
            # r'[:\-_]' 表示：匹配 冒号(:) 或 连字符(-) 中的任意一个
            # expand=True 会把分割后的结果直接变成多列
            split_df = df.index.to_series().str.split(r'[:\-]', expand=True)
            
            # 只有当分割出至少3列时才处理 (chr, start, end)
            if split_df.shape[1] >= 3:
                df['chrom'] = split_df[0]
                df['start'] = split_df[1].astype(int)
                df['end'] = split_df[2].astype(int)
            else:
                raise ValueError("分割后的列数不足3列")
                
        except Exception as e:
            print(f"❌ 解析 ATAC peak 名称失败: {e}")
            print(f"你的 Peak 格式示例: {df.index[0]}")
            return None

    # --- 标准化列名 (PyRanges 必须要求首字母大写) ---
    df = df.rename(columns={
        'chrom': 'Chromosome', 
        'start': 'Start', 
        'end': 'End',
        'chr': 'Chromosome',
        'seqname': 'Chromosome'
    })
    
    # 保留原始 ID 用于追踪
    df['PeakID'] = adata.var_names 
    
    # 最后的完整性检查
    required_cols = ['Chromosome', 'Start', 'End']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ 错误: 数据缺少必要的坐标列。当前列: {df.columns.tolist()}")
        return None

    return pr.PyRanges(df)

# ==========================================
# 工具函数 2: 提取 Peak 中心点 (用于距离计算)
# ==========================================
def get_atac_peak_centers(adata):
    """将 Peak 缩减为 1bp 的中心点"""
    pr_full = get_atac_bed(adata)
    if pr_full is None: return None
    
    df = pr_full.df
    # 计算中心点: (Start + End) / 2
    df['Center'] = (df['Start'] + df['End']) // 2
    
    # 重置 Start 和 End 为中心点
    df['Start'] = df['Center']
    df['End'] = df['Center'] + 1
    
    return pr.PyRanges(df)

# ==========================================
# 工具函数 3: 提取 TSS (用于基因锚点)
# ==========================================
def get_tss(df_gene):
    """提取基因 TSS 位点"""
    genes = df_gene[df_gene['feature'] == 'gene'].copy()
    # 正链 TSS = start, 负链 TSS = end
    genes['TSS_Pos'] = np.where(genes['strand'] == '+', genes['start'], genes['end'])
    
    df_tss = pd.DataFrame({
        'Chromosome': genes['seqname'],
        'Start': genes['TSS_Pos'],
        'End': genes['TSS_Pos'] + 1, # 左闭右开
        'gene_name': genes['gene_name'],
        'gene_id': genes['gene_id']
    })
    return pr.PyRanges(df_tss)


def add_tier_code(df):
    """
    将 Filter_Class 的文本描述转换为整数编码 (Tier_Code)
    1 = Tier 1 (Gold)
    2 = Tier 2 (Silver)
    3 = Tier 3 (Bronze)
    0 = Discard
    """
    def get_code(text):
        if str(text).startswith("Tier 1"):
            return 1
        elif str(text).startswith("Tier 2"):
            return 2
        elif str(text).startswith("Tier 3"):
            return 3
        else:
            return 0 # 代表 Discard

    # 应用转换
    df['Tier_Code'] = df['Filter_Class'].apply(get_code)
    return df

# ==========================================
# 核心过滤函数 (Integrated)
# ==========================================
def filter_atac_peaks_integrated(adata_atac, df_chip, df_hic, df_gene, 
                                 distance_threshold=250000): # 250kb
    
    print("--- 步骤 1: 构建基础区间对象 ---")
    # 构建两个版本的 ATAC 对象
    pr_atac_full = get_atac_bed(adata_atac)      # 完整版 -> 用来撞 ChIP 和 Hi-C
    pr_atac_center = get_atac_peak_centers(adata_atac) # 中心版 -> 用来算距离
    
    if pr_atac_full is None: return None

    # ChIP-seq
    df_chip_clean = df_chip.rename(columns={'chrom': 'Chromosome', 'start': 'Start', 'end': 'End'})
    pr_chip = pr.PyRanges(df_chip_clean)
    
    # TSS
    pr_tss = get_tss(df_gene)

    # Hi-C Loops
    df_hic['LoopID'] = df_hic.index
    pr_hic_1 = pr.PyRanges(df_hic.rename(columns={'chr1': 'Chromosome', 'start1': 'Start', 'end1': 'End'})[['Chromosome', 'Start', 'End', 'LoopID']])
    pr_hic_2 = pr.PyRanges(df_hic.rename(columns={'chr2': 'Chromosome', 'start2': 'Start', 'end2': 'End'})[['Chromosome', 'Start', 'End', 'LoopID']])

    print("--- 步骤 2: 评估 TF 结合证据 (使用完整 Peak) ---")
    # 只要 Peak 的任何部分和 TF ChIP-seq 重叠，就算有结合
    overlap_chip = pr_atac_full.overlap(pr_chip)
    chip_supported_ids = set(overlap_chip.df['PeakID']) 

    print("--- 步骤 3: 评估 靶基因连接证据 ---")
    
    # === 3.1 Hi-C 物理互作 (使用完整 Peak) ===
    # Peak 落在 Loop 一端，TSS 在另一端
    # A路: Peak(H1) - Loop - TSS(H2)
    atac_on_h1 = pr_atac_full.join(pr_hic_1).df 
    tss_on_h2 = pr_tss.join(pr_hic_2).df 
    path_a = pd.merge(atac_on_h1, tss_on_h2, on='LoopID')
    
    # B路: Peak(H2) - Loop - TSS(H1)
    atac_on_h2 = pr_atac_full.join(pr_hic_2).df
    tss_on_h1 = pr_tss.join(pr_hic_1).df
    path_b = pd.merge(atac_on_h2, tss_on_h1, on='LoopID')
    
    hic_supported_ids = set(path_a['PeakID']).union(set(path_b['PeakID']))

    # === 3.2 启动子近端连接 (使用完整 Peak) ===
    # 扩展 TSS 为 ±2kb 启动子区
    pr_promoter_region = pr_tss.extend(2000) 
    overlap_promoter = pr_atac_full.overlap(pr_promoter_region)
    promoter_supported_ids = set(overlap_promoter.df['PeakID'])

    # === 3.3 [核心修改] 250kb 距离连接 (使用 Peak 中心) ===
    # 逻辑: TSS 向两侧扩展 250kb，看 Peak 的【中心点】是否落进去
    pr_tss_window = pr_tss.extend(distance_threshold)
    
    cis_links = pr_atac_center.join(pr_tss_window).df
    distance_supported_ids = set(cis_links['PeakID'])
    
    print(f"统计: {len(hic_supported_ids)} 个 Hi-C 连接, {len(distance_supported_ids)} 个 250kb 内连接")

    print("--- 步骤 4: 整合分级 (Tiering) ---")
    # 使用完整版 DataFrame 返回结果
    df_res = pr_atac_full.df.copy()
    
    # 标记证据
    df_res['has_chip'] = df_res['PeakID'].isin(chip_supported_ids)
    df_res['is_promoter'] = df_res['PeakID'].isin(promoter_supported_ids)
    df_res['has_hic'] = df_res['PeakID'].isin(hic_supported_ids)
    df_res['in_250kb'] = df_res['PeakID'].isin(distance_supported_ids)
    
    # 定义分级
    def define_tier(row):
        tf_valid = row['has_chip']
        
        gene_valid = False
        link_type = "None"
        
        if row['is_promoter']:
            gene_valid = True
            link_type = "Promoter"
        elif row['has_hic']:
            gene_valid = True
            link_type = "Hi-C Loop"
        elif row['in_250kb']:
            gene_valid = True
            link_type = "Distal <250kb"
            
        if tf_valid and gene_valid:
            return f"Tier 1: Confirmed TF + {link_type}"
        elif tf_valid and not gene_valid:
            return "Tier 2: Valid TF (Orphan)"
        elif not tf_valid and gene_valid:
            return f"Tier 3: Open Chromatin + {link_type}"
        else:
            return "Discard"

    df_res['Filter_Class'] = df_res.apply(define_tier, axis=1)

    df_res = add_tier_code(df_res)

    return df_res



def atac_other_preprocess(adata_atac):

    max_peaks = min(4500, adata_atac.n_vars)

    adata_atac.layers['counts'] = adata_atac.X.copy()

    print("--- 步骤 5: 运行下游分析 (TF-IDF, SVD, UMAP) ---")

    # 1. TF-IDF 归一化 (等同于 Seurat::RunTFIDF)
    ma.pp.tfidf(adata_atac, scale_factor=1e4)

    sc.pp.highly_variable_genes(
        adata_atac,
        n_top_genes=max_peaks, # 选择所有 peak
        flavor='seurat_v3' # 使用 'seurat_v3' 算法
    )

    sc.tl.pca(adata_atac, n_comps=50, use_highly_variable=True, svd_solver='arpack')

    n_dims_to_use = 29
    n_comps_to_use = 30

    if adata_atac.obsm['X_pca'].shape[1] < n_comps_to_use:
        sc.tl.pca(adata_atac, n_comps=n_comps_to_use, use_highly_variable=True, svd_solver='arpack')

        adata_atac.obsm['X_lsi'] = adata_atac.obsm['X_pca'][:, 1:n_comps_to_use]

        print(f"已创建 'X_lsi'，使用 SVD/PCA 成分 2 到 {n_comps_to_use} (共 {n_dims_to_use} 个)")

        sc.pp.neighbors(adata_atac, n_neighbors=30, n_pcs=n_dims_to_use, use_rep='X_lsi')

        sc.tl.umap(adata_atac)

        sc.tl.leiden(adata_atac, resolution=1.0, key_added='leiden') 

        adata_atac.layers['norm'] = adata_atac.X.copy()

        adata_atac.X = adata_atac.layers['counts'].copy()

    return adata_atac
    

def atac_preprocess(adata_atac, tf_chip_seq_scenic, 
                    hic_data_df, gene_info, peak_filter_file, adata_atac_file):

    adata_atac = filter_atac_base(adata_atac)

    df_res = filter_atac_peaks_integrated(adata_atac, tf_chip_seq_scenic, hic_data_df, gene_info, 
                                    distance_threshold=250000)

    df_res.to_pickle(peak_filter_file)

    adata_atac = filter_atac_base(adata_atac)

    gold_peaks = df_res[df_res['Tier_Code'] == 1]

    filter_peak = sorted( list( set(gold_peaks["PeakID"]) & set(adata_atac.var_names) ) )

    filter_atac = adata_atac[:, filter_peak].copy()

    # 只有glue需要处理，这一块可以考虑不用

    # filter_atac = atac_other_preprocess(filter_atac)

    filter_atac.write_h5ad(adata_atac_file)

    return filter_atac

####################################################################################################
#***************************----------------------------------------*******************************#
#***************************   Gene info 数据预处理函数   *******************************#
#***************************----------------------------------------*******************************#
####################################################################################################


def process_gene_info(gtf_df):

    # 过滤非基因信息
    filter_gtf = gtf_df[gtf_df["feature"] == "gene"].copy()

    # 过滤非标准染色体
    standard_chroms = tuple([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'])

    filter_gtf = filter_gtf[filter_gtf["seqname"].isin(standard_chroms)].copy()

    # 过滤掉基因类型为 TEC 和 misc_RNA 的基因
    filter_gtf = filter_gtf[~filter_gtf["gene_type"].isin(["TEC", "misc_RNA"])].copy()

    # 去重
    filter_gtf = filter_gtf.drop_duplicates(subset=['gene_name', "seqname"], keep='first')

    return filter_gtf


####################################################################################################
#***************************----------------------------------------*******************************#
#***************************   RNA数据预处理函数   *******************************#
#***************************----------------------------------------*******************************#
####################################################################################################



def filter_adata_rna(adata_rna, data_type = 'sim'):

    adata_rna = adata_rna[:, ~adata_rna.var_names.str.startswith('ENSG00')].copy()

    adata_rna.var_names_make_unique()

    adata_rna.var['mt'] = adata_rna.var_names.str.startswith('MT-')

    sc.pp.calculate_qc_metrics(adata_rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


    MIN_GENES = 100
    MAX_GENES = 6000   # 过滤掉基因数过高的细胞（可能是双细胞 Doublets）
    MIN_COUNTS = 200


    adata_rna = adata_rna[adata_rna.obs.n_genes_by_counts > MIN_GENES, :]
    adata_rna = adata_rna[adata_rna.obs.n_genes_by_counts < MAX_GENES, :]
    adata_rna = adata_rna[adata_rna.obs.total_counts > MIN_COUNTS, :]
    
    if data_type == 'real':
        MAX_PCT_MT = 15.0  # 设定线粒体比例最高为 15%
        adata_rna = adata_rna[adata_rna.obs.pct_counts_mt < MAX_PCT_MT, :]

    sc.pp.filter_genes(adata_rna, min_cells=10)

    sc.pp.filter_genes(adata_rna, min_counts=100)

    adata_rna = adata_rna[:, ~adata_rna.var['mt']].copy()

    artifact_pattern = "^AC\d+|^AL\d+|^AP\d+|^AF\d+"
    
    is_artifact = adata_rna.var_names.str.contains(artifact_pattern, regex=True)

    adata_rna = adata_rna[:, ~is_artifact].copy()
    
    return adata_rna


def rna_preprocess(adata_rna, gtf_df, adata_rna_file, gene_info_file):
    
    gtf_df = process_gene_info(gtf_df)

    filter_gtf = gtf_df[gtf_df["gene_name"].isin(adata_rna.var_names)].copy()

    adata_rna = filter_adata_rna(adata_rna)

    common_gene =sorted(list( set(filter_gtf["gene_name"]) & set(adata_rna.var_names) ))

    adata_rna = adata_rna[:, common_gene].copy()

    adata_rna = preprocess_adata_rna(adata_rna, num_high_variable_gene=2000)

    adata_rna = adata_rna[:, adata_rna.var['highly_variable']].copy()

    filter_gtf = gtf_df[gtf_df["gene_name"].isin(adata_rna.var_names)].copy()

    adata_rna.write_h5ad(adata_rna_file)

    filter_gtf.to_pickle(gene_info_file)

    return adata_rna, filter_gtf


######################################################################################################
def preprocess_adata_rna(adata_rna, num_high_variable_gene=1500):

    adata_rna.layers["counts"] = adata_rna.X.copy()

    sc.pp.highly_variable_genes(adata_rna, n_top_genes=num_high_variable_gene, flavor="seurat_v3")

    sc.pp.normalize_total(adata_rna)

    sc.pp.log1p(adata_rna)

    sc.pp.scale(adata_rna)

    sc.tl.pca(adata_rna, n_comps=100, svd_solver="auto")

    sc.pp.neighbors(adata_rna, metric="cosine")

    sc.tl.umap(adata_rna)

    adata_rna.layers["norm"] = adata_rna.X.copy()

    adata_rna.X = adata_rna.layers["counts"].copy()

    return adata_rna




####################################################################################################
#***************************----------------------------------------*******************************#
#***************************   构建 TF-peak网络   *******************************#
#***************************----------------------------------------*******************************#
####################################################################################################


def build_tf_peak_network(atac_adata, tf_chip_df, df_res_filtered=None):
    """
    通过计算 TF ChIP-seq 与 ATAC-seq 的重叠构建 TF-Peak 调控矩阵 (AnnData)。
    
    Args:
        atac_adata (AnnData): 
            原始的 ATAC AnnData 对象，.var_names 为 'chr:start-end' 格式。
        tf_chip_df (DataFrame): 
            包含 TF ChIP-seq 峰的数据框，必须包含 'chrom', 'start', 'end', 'tf_name'。
        df_res_filtered (DataFrame, Optional): 
            [可选] 上一步过滤后的结果表 (包含 Tier_Code)。
            如果提供，将只针对这些筛选后的 Peak 构建矩阵 (例如只看 Tier 1)。
            如果不提供，则对 atac_adata 中所有的 Peak 进行计算。
            
    Returns:
        anndata.AnnData: 
            X 为稀疏的 0/1 矩阵 (Rows=Peaks, Cols=TFs)。
    """
    
    print("--- 1. 准备 ATAC Peak 数据 ---")
    
    # 策略判断：是利用过滤后的子集，还是用全量数据
    if df_res_filtered is not None:
        print(f"检测到过滤列表，将仅使用 {len(df_res_filtered)} 个筛选后的 Peaks。")
        # 构造 PyRanges 对象 (假设 df_res_filtered 已经有标准列名)
        # 如果列名不对，这里做一个保护性重命名
        target_df = df_res_filtered.reset_index().rename(columns={
            'chrom': 'Chromosome', 'start': 'Start', 'end': 'End', 
            'PeakID': 'PeakID'
        })
        # 确保包含必要的列
        if 'PeakID' not in target_df.columns: target_df['PeakID'] = target_df.index
        pr_atac = pr.PyRanges(target_df)
        
        # 记录我们要保留的 Peak 索引列表，用于最后的对齐
        valid_peak_indices = target_df['PeakID'].unique()
        
    else:
        print("未提供过滤列表，将使用 atac_adata 中所有的 Peaks。")
        # 使用之前的工具函数解析坐标
        pr_atac = get_atac_bed(atac_adata)
        if pr_atac is None: return None
        valid_peak_indices = atac_adata.var_names

    print(f"ATAC Peaks 数量: {len(pr_atac)}")


    print("--- 2. 准备 TF ChIP 数据 ---")
    # 确保列名符合 PyRanges 标准
    # 假设你的 TF 列名是 'tf_name' (或者 'tf')，这里统一处理
    tf_col = 'tf_name' if 'tf_name' in tf_chip_df.columns else 'tf'
    
    chip_clean = tf_chip_df.rename(columns={
        'chrom': 'Chromosome', 'start': 'Start', 'end': 'End',
        tf_col: 'TF_Name' # 统一重命名为 TF_Name
    })[['Chromosome', 'Start', 'End', 'TF_Name']]
    
    pr_chip = pr.PyRanges(chip_clean)
    print(f"TF ChIP Peaks 数量: {len(pr_chip)}")


    print("--- 3. 计算重叠 (构建 Network) ---")
    # 这一步等同于 pybedtools.intersect(wa=True, wb=True)
    # join 会保留两个区间的所有信息
    # 结果包含: PeakID (来自 ATAC) 和 TF_Name (来自 ChIP)
    network_pr = pr_atac.join(pr_chip)
    
    if len(network_pr) == 0:
        print("❌ 警告: 未发现任何重叠！请检查染色体名称是否一致 (如 chr1 vs 1)。")
        return None
        
    network_df = network_pr.df
    
    # 去重：同一个 TF 在同一个 Peak 上可能有多个结合位点，我们只记一次连接
    network_unique = network_df[['PeakID', 'TF_Name']].drop_duplicates()
    
    print(f"构建了包含 {len(network_unique)} 条边的 TF-Peak 网络。")


    print("--- 4. 转换为矩阵 (Matrix Construction) ---")
    # 使用 crosstab 构建 0/1 矩阵
    # Index = PeakID, Columns = TF_Name
    matrix_df = pd.crosstab(index=network_unique['PeakID'], columns=network_unique['TF_Name'])
    
    # 转换为二值 (Binary)
    matrix_df = (matrix_df > 0).astype(int)
    
    # [关键步骤] 对齐补全 (Reindexing)
    # 必须确保输出的 AnnData 包含了我们关注的所有 Peak (即使它没有结合任何 TF，也应该是一行全0)
    final_matrix = matrix_df.reindex(index=valid_peak_indices, fill_value=0)
    
    # 对 TF 列名排序，美观
    final_matrix = final_matrix.sort_index(axis=1)


    print("--- 5. 封装为 AnnData ---")
    # 构建 AnnData
    adata_tf = ad.AnnData(
        X=sparse.csr_matrix(final_matrix.values),
        obs=pd.DataFrame(index=final_matrix.index), # obs索引是 PeakID
        var=pd.DataFrame(index=final_matrix.columns) # var索引是 TF Name
    )
    
    # 如果提供了详细信息表，可以把它合并进 obs
    if df_res_filtered is not None:
        # 确保索引对齐
        adata_tf.obs = adata_tf.obs.join(df_res_filtered.set_index('PeakID'), how='left')
    
    adata_tf.uns['description'] = 'TF-Peak Binary Regulation Matrix'
    
    print(f"✅ 完成！生成的 AnnData 维度: {adata_tf.shape} (Peaks x TFs)")

    return adata_tf



####################################################################################################
#***************************----------------------------------------*******************************#
#***************************   peak-gene network *******************************#
#***************************----------------------------------------*******************************#
####################################################################################################


# ==========================================
# 依赖的工具函数 (复用之前的逻辑)
# ==========================================
# 假设 get_atac_bed, get_atac_peak_centers, get_tss 已经定义
# 如果未定义，请复制上文中的定义

def get_tss_distance(df_gene, distance):
    """提取基因 TSS 位点"""
    genes = df_gene[df_gene['feature'] == 'gene'].copy()
    # 正链 TSS = start, 负链 TSS = end
    genes['TSS_Pos'] = np.where(genes['strand'] == '+', genes['start'], genes['end'])
    
    df_tss = pd.DataFrame({
        'Chromosome': genes['seqname'],
        'Start': genes['TSS_Pos']-distance,
        'End': genes['TSS_Pos'] + distance, # 左闭右开
        'gene_name': genes['gene_name'],
        'gene_id': genes['gene_id']
    })
    return pr.PyRanges(df_tss)






import numpy as np
import pandas as pd
import scipy.sparse as sparse
import anndata as ad
import pyranges as pr


def build_hic_peak_gene_network_schic(
    atac_adata,
    hic_df,
    score_col=None,
    use_peak_center=False,
    binary=False,
    agg_method="max"
):
    """
    根据 Hi-C region-gene 表（chr, start, end, gene）构建 Peak-Gene 网络。

    规则：
    只要 ATAC peak 与 Hi-C 区间 [chr, start, end] 有重叠，
    就认为该 peak 与对应 gene 存在一条边。

    Parameters
    ----------
    atac_adata : AnnData
        ATAC-seq 数据对象。要求 var_names 能唯一标识 peak，
        且 get_atac_bed()/get_atac_peak_centers() 返回结果中包含 'PeakID' 列。

    hic_df : pd.DataFrame
        Hi-C region-gene 数据表，至少包含以下列：
        - chr
        - start
        - end
        - gene
        可选包含一个分数字段 score_col。

    gene_info_df : pd.DataFrame or None, optional
        为了兼容旧接口而保留，但当前版本不使用。

    score_col : str or None, optional
        Hi-C 边权重列名。
        - 若为 None，则每条 overlap 边权重记为 1
        - 若提供，则使用该列作为边权重
        当同一 Peak-Gene 对应多个区间时，会按 agg_method 聚合。

    use_peak_center : bool, default=False
        - False: 用完整 peak 区间与 Hi-C 区间 overlap
        - True : 用 peak 中心点与 Hi-C 区间 overlap（更严格）

    binary : bool, default=True
        - True : 输出二值矩阵（有边=1）
        - False: 输出加权矩阵（权重来自 score_col 或默认 1）

    agg_method : {"max", "sum", "mean"}, default="max"
        当同一 Peak-Gene 由于多个 Hi-C 区间重复出现时，如何聚合边权重。

    Returns
    -------
    peak_gene_df : pd.DataFrame
        包含三列：
        - PeakID
        - gene_name
        - hic_score

    peak_gene_grn : AnnData
        Peaks x Genes 的连接矩阵：
        - obs.index = all peaks
        - var.index = connected genes

    Notes
    -----
    1. 本函数适用于 Hi-C 已整理成 region-gene 配对表的情况。
    2. 这构建的是 peak-gene 候选连接网络 / prior network，
       不是原始 loop 双端严格重建。
    """

    print("--- 1. 数据准备 (Converting to PyRanges) ---")

    # ------------------------------------------------------------------
    # 1) 获取 peak 区间
    # ------------------------------------------------------------------
    if use_peak_center:
        pr_peaks = get_atac_peak_centers(atac_adata)
        print("  -> 使用 peak center 与 Hi-C 区间 overlap")
    else:
        pr_peaks = get_atac_bed(atac_adata)
        print("  -> 使用完整 peak 区间与 Hi-C 区间 overlap")

    if pr_peaks is None:
        print("❌ 错误: 无法从 atac_adata 提取 peak 区间信息。")
        return None

    peak_df = pr_peaks.df.copy()
    if 'PeakID' not in peak_df.columns:
        raise ValueError(
            "get_atac_bed() / get_atac_peak_centers() 返回的 PyRanges 中必须包含 'PeakID' 列。"
        )

    # ------------------------------------------------------------------
    # 2) 检查并标准化 Hi-C region-gene 表
    # ------------------------------------------------------------------
    if hic_df is None or len(hic_df) == 0:
        print("❌ 错误: hic_df 为空。")
        return None

    hic_df = hic_df.copy()

    required_cols = ['chr', 'start', 'end', 'gene']
    missing_cols = [c for c in required_cols if c not in hic_df.columns]
    if missing_cols:
        raise ValueError(f"hic_df 缺少必要列: {missing_cols}")

    # 基础清洗
    hic_df = hic_df.dropna(subset=['chr', 'start', 'end', 'gene']).copy()
    hic_df['start'] = hic_df['start'].astype(int)
    hic_df['end'] = hic_df['end'].astype(int)
    hic_df['gene'] = hic_df['gene'].astype(str)

    # 去除非法区间
    hic_df = hic_df[hic_df['end'] > hic_df['start']].copy()
    if hic_df.empty:
        print("❌ 错误: hic_df 清洗后没有合法区间。")
        return None

    # 处理 score
    if score_col is not None:
        if score_col not in hic_df.columns:
            raise ValueError(f"score_col='{score_col}' 不在 hic_df 中。")
        hic_df[score_col] = pd.to_numeric(hic_df[score_col], errors='coerce')
        hic_df[score_col] = hic_df[score_col].fillna(0.0)
        used_score_col = score_col
    else:
        hic_df['hic_score'] = 1.0
        used_score_col = 'hic_score'

    # 转为 PyRanges 格式
    hic_pr_df = hic_df.rename(columns={
        'chr': 'Chromosome',
        'start': 'Start',
        'end': 'End',
        'gene': 'gene_name'
    })[['Chromosome', 'Start', 'End', 'gene_name', used_score_col]].copy()

    pr_hic = pr.PyRanges(hic_pr_df)

    print(f"  -> Hi-C region-gene 区间数: {len(hic_pr_df)}")

    # ------------------------------------------------------------------
    # 3) overlap：peak × Hi-C region-gene
    # ------------------------------------------------------------------
    print("--- 2. 寻找 Peak-Gene 连接 (Overlap peak with Hi-C region-gene) ---")

    overlap_df = pr_peaks.join(pr_hic).df

    if overlap_df.empty:
        print("❌ 警告: 未找到任何 Peak-Gene 连接！")
        return None

    required_overlap_cols = ['PeakID', 'gene_name', used_score_col]
    missing_overlap_cols = [c for c in required_overlap_cols if c not in overlap_df.columns]
    if missing_overlap_cols:
        raise ValueError(
            f"overlap 结果中缺少必要列: {missing_overlap_cols}\n"
            f"请检查 get_atac_bed()/get_atac_peak_centers() 是否保留了 PeakID。"
        )

    peak_gene_df = overlap_df[['PeakID', 'gene_name', used_score_col]].copy()
    peak_gene_df = peak_gene_df.rename(columns={used_score_col: 'hic_score'})

    # ------------------------------------------------------------------
    # 4) 聚合同一 Peak-Gene 的重复边
    # ------------------------------------------------------------------
    if agg_method == "max":
        peak_gene_df = peak_gene_df.groupby(
            ['PeakID', 'gene_name'], as_index=False
        )['hic_score'].max()
    elif agg_method == "sum":
        peak_gene_df = peak_gene_df.groupby(
            ['PeakID', 'gene_name'], as_index=False
        )['hic_score'].sum()
    elif agg_method == "mean":
        peak_gene_df = peak_gene_df.groupby(
            ['PeakID', 'gene_name'], as_index=False
        )['hic_score'].mean()
    else:
        raise ValueError("agg_method 只能是 'max', 'sum', 'mean'")

    if peak_gene_df.empty:
        print("❌ 警告: 聚合后没有任何 Peak-Gene 连接！")
        return None

    print(f"共找到 {len(peak_gene_df)} 条唯一的 Peak-Gene 连接。")

    # ------------------------------------------------------------------
    # 5) 构建稀疏矩阵
    # ------------------------------------------------------------------
    print("--- 3. 构建稀疏矩阵 ---")

    all_peaks = atac_adata.var_names.tolist()
    peak_gene_df = peak_gene_df[peak_gene_df['PeakID'].isin(all_peaks)].copy()

    if peak_gene_df.empty:
        print("❌ 警告: overlap 得到的 PeakID 与 atac_adata.var_names 没有交集。")
        return None

    all_genes = sorted(peak_gene_df['gene_name'].unique())

    peak_to_idx = {peak: i for i, peak in enumerate(all_peaks)}
    gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}

    rows = peak_gene_df['PeakID'].map(peak_to_idx).values
    cols = peak_gene_df['gene_name'].map(gene_to_idx).values
    data = peak_gene_df['hic_score'].astype(float).values

    sparse_matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(all_peaks), len(all_genes))
    )

    if binary:
        sparse_matrix.data = np.where(sparse_matrix.data > 0, 1, 0)

    peak_gene_grn = ad.AnnData(
        X=sparse_matrix,
        obs=pd.DataFrame(index=all_peaks),
        var=pd.DataFrame(index=all_genes)
    )

    peak_gene_grn.uns['description'] = 'Peak-Gene connectivity matrix built from Hi-C region-gene overlap'
    peak_gene_grn.uns['source'] = 'Hi-C region-gene table'
    peak_gene_grn.uns['use_peak_center'] = use_peak_center
    peak_gene_grn.uns['binary'] = binary
    peak_gene_grn.uns['agg_method'] = agg_method
    peak_gene_grn.uns['score_col'] = None if score_col is None else score_col

    print(f"✅ 构建完成! 矩阵维度: {peak_gene_grn.shape} (Peaks x Genes)")

    return peak_gene_df, peak_gene_grn







def build_hic_peak_gene_network(atac_adata, hic_df, gene_info_df, score_col):
    """
    依据Hi-C Loops (物理互作)，构建 Peak-Gene 调控网络并存储为 AnnData 格式。

    Args:
        atac_adata (AnnData): ATAC-seq 数据对象。
        hic_df (DataFrame): 包含 'chr1', 'start1', 'end1', 'chr2', ... 列。
        gene_info_df (DataFrame): 基因注释包含 'gene_name', 'seqname', 'start', 'end', 'strand'。
        distance_threshold (int): 顺式调控距离阈值 (默认 250kb)。

    Returns:
        pd.DataFrame: 一个表示Peak-Gene连接的DataFrame。
    """
    
    print("--- 1. 数据准备 (Converting to PyRanges) ---")
    

    pr_atac_full = get_atac_bed(atac_adata)
   
    pr_atac_center = get_atac_peak_centers(atac_adata) 
    
    if pr_atac_full is None or pr_atac_center is None: return None
    
    pr_tss = get_tss_distance(gene_info_df, 1000)
    
    
    # 1.3 Hi-C Data

    hic_df = hic_df.copy()
    hic_df['LoopID'] = range(len(hic_df))
    
    cols_1 = ['Chromosome', 'Start', 'End', 'LoopID', score_col]
    cols_2 = ['Chromosome', 'Start', 'End', 'LoopID', score_col]
    
    # 重命名列以适配 PyRanges，同时保留 score_col
    df_h1 = hic_df.rename(columns={'chr1': 'Chromosome', 'start1': 'Start', 'end1': 'End'})[cols_1]
    df_h2 = hic_df.rename(columns={'chr2': 'Chromosome', 'start2': 'Start', 'end2': 'End'})[cols_2]
    
    pr_hic_1 = pr.PyRanges(df_h1)
    pr_hic_2 = pr.PyRanges(df_h2)
 
    print("--- 2. 寻找连接 (Finding Connections) ---")
    
    # 容器：存储所有的 (PeakID, GeneName) 对
    all_connections = []

    # === A. Hi-C 连接 ===
    print("  -> 计算 Hi-C Loops 连接...")
    # 路径 1: Peak在端点1, TSS在端点2
    p_h1 = pr_atac_full.join(pr_hic_1).df
    t_h2 = pr_tss.join(pr_hic_2).df
    
    links_a = pd.merge(p_h1, t_h2, on='LoopID')
    links_a['hic_score'] = hic_df.loc[links_a['LoopID'], score_col].values
    links_a = links_a[['PeakID', 'gene_name', 'hic_score']]
    
    # 路径 2: Peak在端点2, TSS在端点1
    p_h2 = pr_atac_full.join(pr_hic_2).df
    t_h1 = pr_tss.join(pr_hic_1).df
    links_b = pd.merge(p_h2, t_h1, on='LoopID')
    links_b['hic_score'] = hic_df.loc[links_b['LoopID'], score_col].values
    links_b = links_b[['PeakID', 'gene_name', 'hic_score']]
    
    all_connections.append(links_a)
    all_connections.append(links_b)

    peak_gene_df = pd.concat(all_connections).drop_duplicates()
    
    if peak_gene_df.empty:
        print("❌ 警告: 未找到任何 Peak-Gene 连接！")
        return None
    
    print(f"共找到 {len(peak_gene_df)} 条唯一的 Peak-Gene 连接。")
    
    ###################################################
    
    all_peaks = atac_adata.var_names
    peak_to_idx = {peak: i for i, peak in enumerate(all_peaks)}

    all_genes = sorted(peak_gene_df['gene_name'].unique())
    gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}

    peak_gene_df = peak_gene_df[peak_gene_df['PeakID'].isin(all_peaks)].copy()
    rows = peak_gene_df['PeakID'].map(peak_to_idx).values
    cols = peak_gene_df['gene_name'].map(gene_to_idx).values
    data = peak_gene_df["hic_score"]
    sparse_matrix = sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(len(all_peaks), len(all_genes))
        )
    sparse_matrix.data = np.where(sparse_matrix.data > 0, 1, 0)
    
    peak_gene_grn = ad.AnnData(
        X=sparse_matrix,
        obs=pd.DataFrame(index=all_peaks), # 行索引严格对齐 ATAC Peak
        var=pd.DataFrame(index=all_genes)  # 列索引是 Gene
    )
    
    peak_gene_grn.uns['description'] = 'Peak-Gene Binary Connectivity Matrix'
    
    print(f"✅ 构建完成! 矩阵维度: {peak_gene_grn.shape} (Peaks x Genes)")
    
    return peak_gene_df, peak_gene_grn

    



####################################################################################################
#***************************----------------------------------------*******************************#
#***************************   RNA数据预处理函数   *******************************#
#***************************----------------------------------------*******************************#
####################################################################################################


def build_tf_gene_network_from_anndata(adata_tf_peak, adata_peak_gene):
    """
    输入两个 AnnData 对象，利用矩阵乘法快速构建 TF-Gene 调控网络。
    
    Args:
        adata_tf_peak (AnnData): 
            行(Obs) = Peaks, 列(Var) = TFs
            值 = 0/1 (TF是否结合在该Peak)
        adata_peak_gene (AnnData): 
            行(Obs) = Peaks, 列(Var) = Genes
            值 = 0/1 (该Peak是否调控该Gene)
            
    Returns:
        adata_tf_gene (AnnData):
            行(Obs) = TFs, 列(Var) = Genes
            值(X) = Peak Count (支持该调控关系的 Peak 数量，即调控强度)
    """
    print("\n--- 构建 TF-Gene 网络 (AnnData 矩阵乘法版) ---")
    
    # 1. 确保 Peak 对齐 (Intersection)
    # 两个矩阵必须基于相同的 Peak 集合才能相乘
    # intersection 自动找到共有的 PeakID
    common_peaks = adata_tf_peak.obs_names.intersection(adata_peak_gene.obs_names)
    
    print(f"TF矩阵 Peaks: {adata_tf_peak.n_obs}")
    print(f"Gene矩阵 Peaks: {adata_peak_gene.n_obs}")
    print(f"共有 Peaks: {len(common_peaks)}")
    
    if len(common_peaks) == 0:
        print("❌ 错误: 两个矩阵没有共有的 PeakID，无法进行关联！")
        return None
        
    # 2. 对矩阵进行切片和对齐
    # 只保留共有的 Peak，并保证顺序一致
    # A矩阵: Peak x TF
    subset_tf_peak = adata_tf_peak[common_peaks, :]
    
    # B矩阵: Peak x Gene
    subset_peak_gene = adata_peak_gene[common_peaks, :]
    
    print("正在进行稀疏矩阵乘法...")
    
    # 3. 核心计算: 矩阵乘法
    # 我们需要 (TF x Gene)
    # 公式: (TF x Peak) * (Peak x Gene)
    # 因为输入是 (Peak x TF)，所以需要转置 (.T)
    
    # .X 通常是稀疏矩阵 (csr_matrix)，.T 也是极速操作
    # @ 符号在 Python 中表示矩阵乘法
    X_tf_gene = subset_tf_peak.X.T @ subset_peak_gene.X
    
    # 4. 封装结果为 AnnData
    print("正在封装结果...")
    
    adata_tf_gene = ad.AnnData(
        X=X_tf_gene,
        # 新的行索引是 TF (来自第一个矩阵的 Var)
        obs=pd.DataFrame(index=subset_tf_peak.var_names), 
        # 新的列索引是 Gene (来自第二个矩阵的 Var)
        var=pd.DataFrame(index=subset_peak_gene.var_names) 
    )
    
    # 5. 添加元数据
    adata_tf_gene.obs_names.name = "TF_Name"
    adata_tf_gene.var_names.name = "Gene_Name"
    adata_tf_gene.uns['description'] = "TF-Gene Regulatory Network (Value = Number of supporting peaks)"
    
    # 统计一下非零连接数
    num_links = adata_tf_gene.X.nnz
    print(f"✅ 构建完成! 矩阵维度: {adata_tf_gene.shape} (TFs x Genes)")
    print(f"共发现 {num_links} 条潜在的 TF-Gene 调控关系。")
    
    return adata_tf_gene






####################################################################################################
#***************************----------------------------------------*******************************#
#***************************   计算调控潜力   *******************************#
#***************************----------------------------------------*******************************#
####################################################################################################


# ==========================================
# 1. 数据清洗函数
# ==========================================
def process_gtf_info(gene_info_df, valid_genes_list):
    """提取 TSS 和 Gene Length"""
    print("--- 1. 清洗 Gene Info ---")
    
    # 尝试筛选 gene 类型
    if 'feature' in gene_info_df.columns:
        df = gene_info_df[gene_info_df['feature'] == 'gene'].copy()
    else:
        df = gene_info_df.copy() # 如果没有feature列，假设每一行都是基因
    
    # 如果筛选后为空（可能feature列里没有'gene'），尝试聚合
    if df.empty:
        print("提示: Feature列未找到'gene'，尝试聚合...")
        df = gene_info_df.groupby('gene_name').agg({
            'seqname': 'first', 'start': 'min', 'end': 'max', 'strand': 'first'
        }).reset_index()

    # 筛选有效基因
    df = df[df['gene_name'].isin(valid_genes_list)].copy()
    df = df.drop_duplicates(subset=['gene_name'])
    
    # 【核心】计算 TSS
    # 正链: start; 负链: end
    df['tss'] = np.where(df['strand'] == '+', df['start'], df['end'])
    
    # 计算基因长度 (kb)
    df['gene_length'] = (df['end'] - df['start']) / 1000.0
    df['gene_length'] = df['gene_length'].clip(lower=0.1) # 防止除零
    
    df = df.set_index('gene_name')
    return df

def get_peak_coords(adata_atac):
    """提取 Peak 中心点"""
    peaks_df = pd.DataFrame(index=adata_atac.var_names)
    split_df = peaks_df.index.to_series().str.split(r'[:\-]', expand=True)
    
    peaks_df['chrom'] = split_df[0]
    peaks_df['start'] = split_df[1].astype(int)
    peaks_df['end'] = split_df[2].astype(int)
    peaks_df['center'] = ((peaks_df['start'] + peaks_df['end']) / 2).astype(int)
    
    return peaks_df

# ==========================================
# 2. 核心计算函数 (含 250kb 限制)
# ==========================================
def calculate_rp_250kb(adata_atac, adata_rna, gene_info_df, 
                       decay_dist=50000,   # 半衰距离 (推荐 50kb)
                       max_range=250000):  # 硬性截止距离 (您要求的 250kb)
    
    print(f"\n--- 开始计算 RP Score (Decay={decay_dist}, Range={max_range}) ---")
    
    # 1. 准备数据
    valid_genes = set(adata_rna.var_names)
    gene_df = process_gtf_info(gene_info_df, valid_genes)
    
    # 对齐基因
    common_genes = sorted(list(set(gene_df.index) & set(adata_rna.var_names)))
    gene_df = gene_df.loc[common_genes]
    
    # 准备 Peaks
    peaks_df = get_peak_coords(adata_atac)
    
    # 映射索引
    peak_to_idx = {peak: i for i, peak in enumerate(peaks_df.index)}
    gene_to_idx = {gene: i for i, gene in enumerate(common_genes)}
    
    # 结果容器
    final_rows = []
    final_cols = []
    final_data = []
    
    chromosomes = gene_df['seqname'].unique()
    
    # 按染色体循环
    for chrom in tqdm(chromosomes, desc="Chromosomes"):
        # 获取当前染色体的数据
        sub_genes = gene_df[gene_df['seqname'] == chrom]
        sub_peaks = peaks_df[peaks_df['chrom'] == chrom]
        
        if sub_genes.empty or sub_peaks.empty:
            continue
            
        # === 向量化计算 ===
        # Genes: (N, 1)
        g_tss = sub_genes['tss'].values.reshape(-1, 1)
        g_start = sub_genes['start'].values.reshape(-1, 1)
        g_end = sub_genes['end'].values.reshape(-1, 1)
        g_len = sub_genes['gene_length'].values.reshape(-1, 1)
        
        # Peaks: (1, M)
        p_centers = sub_peaks['center'].values.reshape(1, -1)
        
        # 1. 计算绝对距离
        dists = np.abs(p_centers - g_tss)
        
        # 2. 【关键修改】应用 250kb 硬性过滤
        # 任何距离超过 max_range 的连接，稍后都会被 mask 过滤掉
        mask_valid = dists <= max_range
        
        # 3. 计算衰减分数: 2^(-d / decay)
        # 只计算 valid 的部分其实更快，但为了代码可读性，先全算再过滤
        scores = np.power(2.0, -(dists / decay_dist))
        
        # 4. 处理基因内部 (Gene Body)
        # 逻辑: 如果 Peak 在基因体内，分数设为 1.0 / (基因长度 + 1)
        # 加1是为了防止基因极短导致分数爆炸
        in_gene_body = (p_centers >= g_start) & (p_centers <= g_end)
        
        # 构造 Body Score 矩阵
        body_scores = np.repeat(1.0 / g_len, p_centers.shape[1], axis=1)
        
        # 覆盖基因内部的分数
        scores[in_gene_body] = body_scores[in_gene_body]
        
        # 5. 应用 250kb 过滤器
        # 这一步通过布尔索引，只保留 <= 250kb 且分数 > 0 的点
        # 另外加一个 1e-5 的极小值过滤，保证稀疏性
        final_mask = mask_valid & (scores > 1e-5)
        
        # 提取索引
        r_idx, c_idx = np.where(final_mask)
        
        # 映射回全局索引并存储
        current_g_names = sub_genes.index.values
        current_p_names = sub_peaks.index.values
        
        # 这里为了速度，不使用 append，改用 extend
        # 我们需要先获取 sub_genes 在全局 common_genes 中的索引
        g_global_base_indices = [gene_to_idx[g] for g in current_g_names]
        p_global_base_indices = [peak_to_idx[p] for p in current_p_names]
        
        # 利用 numpy 的高级索引快速映射
        g_global_mapped = np.array(g_global_base_indices)[r_idx]
        p_global_mapped = np.array(p_global_base_indices)[c_idx]
        score_values = scores[r_idx, c_idx]
        
        final_rows.extend(g_global_mapped)
        final_cols.extend(p_global_mapped)
        final_data.extend(score_values)

    print("--- 3. 构建 AnnData 对象 ---")
    
    rp_matrix = sparse.csr_matrix(
        (final_data, (final_rows, final_cols)), 
        shape=(len(common_genes), len(peaks_df))
    )
    
    adata_rp = ad.AnnData(
        X=rp_matrix,
        obs=pd.DataFrame(index=common_genes),
        var=pd.DataFrame(index=peaks_df.index)
    )
    
    # 记录参数到 uns
    adata_rp.uns['decay_distance'] = decay_dist
    adata_rp.uns['max_range'] = max_range
    adata_rp.uns['description'] = f"RP Score (Decay={decay_dist/1000}kb, Max={max_range/1000}kb)"
    
    print(f"Gene-Region regulation 完成! 矩阵维度: {adata_rp.shape}")

    print(f"非零连接数: {rp_matrix.nnz}, Sparsity: {100 * rp_matrix.nnz / (rp_matrix.shape[0] * rp_matrix.shape[1]):.4f}%")
    return adata_rp



# ==========================================
# 2. 核心计算函数 (含 250kb 限制)
# ==========================================
def calculate_rp_distance(adata_atac, adata_rna, gene_info_df, 
                       decay_dist=50000,   # 半衰距离 (推荐 50kb)
                       max_range=250000):  # 硬性截止距离 (您要求的 250kb)
    
    print(f"\n--- 开始计算 RP Score (Decay={decay_dist}, Range={max_range}) ---")
    
    # 1. 准备数据
    valid_genes = set(adata_rna.var_names)
    gene_df = process_gtf_info(gene_info_df, valid_genes)
    
    # 对齐基因
    common_genes = sorted(list(set(gene_df.index) & set(adata_rna.var_names)))
    gene_df = gene_df.loc[common_genes]
    
    # 准备 Peaks
    peaks_df = get_peak_coords(adata_atac)
    
    # 映射索引
    peak_to_idx = {peak: i for i, peak in enumerate(peaks_df.index)}
    gene_to_idx = {gene: i for i, gene in enumerate(common_genes)}
    
    # 结果容器
    final_rows = []
    final_cols = []
    final_data = []
    
    final_dist = []
    
    chromosomes = gene_df['seqname'].unique()
    
    # 按染色体循环
    for chrom in tqdm(chromosomes, desc="Chromosomes"):
        # 获取当前染色体的数据
        sub_genes = gene_df[gene_df['seqname'] == chrom]
        sub_peaks = peaks_df[peaks_df['chrom'] == chrom]
        
        if sub_genes.empty or sub_peaks.empty:
            continue
            
        # === 向量化计算 ===
        # Genes: (N, 1)
        g_tss = sub_genes['tss'].values.reshape(-1, 1)
        g_start = sub_genes['start'].values.reshape(-1, 1)
        g_end = sub_genes['end'].values.reshape(-1, 1)
        g_len = sub_genes['gene_length'].values.reshape(-1, 1)
        
        # Peaks: (1, M)
        p_centers = sub_peaks['center'].values.reshape(1, -1)
        
        # 1. 计算绝对距离
        dists = np.abs(p_centers - g_tss)
        
        # 2. 【关键修改】应用 250kb 硬性过滤
        # 任何距离超过 max_range 的连接，稍后都会被 mask 过滤掉
        mask_valid = dists <= max_range
        
        # 3. 计算衰减分数: 2^(-d / decay)
        # 只计算 valid 的部分其实更快，但为了代码可读性，先全算再过滤
        scores = np.power(2.0, -(dists / decay_dist))
        
        # 4. 处理基因内部 (Gene Body)
        # 逻辑: 如果 Peak 在基因体内，分数设为 1.0 / (基因长度 + 1)
        # 加1是为了防止基因极短导致分数爆炸
        in_gene_body = (p_centers >= g_start) & (p_centers <= g_end)
        
        # 构造 Body Score 矩阵
        body_scores = np.repeat(1.0 / g_len, p_centers.shape[1], axis=1)
        
        # 覆盖基因内部的分数
        scores[in_gene_body] = body_scores[in_gene_body]
        
        # 5. 应用 250kb 过滤器
        # 这一步通过布尔索引，只保留 <= 250kb 且分数 > 0 的点
        # 另外加一个 1e-5 的极小值过滤，保证稀疏性
        final_mask = mask_valid & (scores > 1e-5)
        
        # 提取索引
        r_idx, c_idx = np.where(final_mask)
        
        # 映射回全局索引并存储
        current_g_names = sub_genes.index.values
        current_p_names = sub_peaks.index.values
        
        # 这里为了速度，不使用 append，改用 extend
        # 我们需要先获取 sub_genes 在全局 common_genes 中的索引
        g_global_base_indices = [gene_to_idx[g] for g in current_g_names]
        p_global_base_indices = [peak_to_idx[p] for p in current_p_names]
        
        # 利用 numpy 的高级索引快速映射
        g_global_mapped = np.array(g_global_base_indices)[r_idx]
        p_global_mapped = np.array(p_global_base_indices)[c_idx]
        score_values = scores[r_idx, c_idx]
        score_dist = dists[r_idx, c_idx]
        
        final_rows.extend(g_global_mapped)
        final_cols.extend(p_global_mapped)
        final_data.extend(score_values)
        final_dist.extend(score_dist)

    print("--- 3. 构建 AnnData 对象 ---")
    
    rp_matrix = sparse.csr_matrix(
        (final_data, (final_rows, final_cols)), 
        shape=(len(common_genes), len(peaks_df))
    )
    
    rp_dist_matrix = sparse.csr_matrix(
        (final_dist, (final_rows, final_cols)), 
        shape=(len(common_genes), len(peaks_df))
    )
    
    adata_rp = ad.AnnData(
        X=rp_matrix,
        obs=pd.DataFrame(index=common_genes),
        var=pd.DataFrame(index=peaks_df.index)
    )
    
    adata_rp_dist = ad.AnnData(
        X=rp_dist_matrix,
        obs=pd.DataFrame(index=common_genes),
        var=pd.DataFrame(index=peaks_df.index)
    )
    
    # 记录参数到 uns
    adata_rp.uns['decay_distance'] = decay_dist
    adata_rp.uns['max_range'] = max_range
    adata_rp.uns['description'] = f"RP Score (Decay={decay_dist/1000}kb, Max={max_range/1000}kb)"
    
    
    adata_rp_dist.uns['decay_distance'] = decay_dist
    adata_rp_dist.uns['max_range'] = max_range
    adata_rp_dist.uns['description'] = f"RP Score (Decay={decay_dist/1000}kb, Max={max_range/1000}kb)"
    
    print(f"Gene-Region regulation 完成! 矩阵维度: {adata_rp.shape}")

    print(f"非零连接数: {rp_matrix.nnz}, Sparsity: {100 * rp_matrix.nnz / (rp_matrix.shape[0] * rp_matrix.shape[1]):.4f}%")
    return adata_rp, adata_rp_dist



####################################################################################################
#***************************----------------------------------------*******************************#
#***************************   计算调控潜力   *******************************#
#***************************----------------------------------------*******************************#
####################################################################################################


# ==========================================
# 2. Peak-Peak 调控潜力计算 (250kb 限制版)
# ==========================================
def calculate_peak_peak_rp(adata_atac, 
                           decay_distance=50000, # 推荐 50kb
                           max_range=250000):    # 硬性限制 250kb
    """
    计算 Peak-Peak 之间的调控潜力 (RP Score)。
    逻辑: 仅计算距离 <= 250kb 的 Peak 对，并应用 2^(-d/decay) 衰减。
    """
    print(f"\n--- 计算 Peak-Peak RP (Range={max_range/1000}kb, Decay={decay_distance/1000}kb) ---")
    
    # 1. 准备数据
    peaks_df = get_peak_coords(adata_atac)
    
    # 记录全局索引 (0, 1, 2...) 用于最后映射回大矩阵
    peaks_df['global_idx'] = range(len(peaks_df))
    
    final_rows = []
    final_cols = []
    final_data = []
    
    # 【核心修改 1】直接将搜索半径设为您的硬性限制 (250kb)
    radius = max_range 
    
    chromosomes = peaks_df['chrom'].unique()
    
    for chrom in tqdm(chromosomes, desc="Chromosomes"):
        sub_peaks = peaks_df[peaks_df['chrom'] == chrom]
        if sub_peaks.empty: continue
            
        # 提取坐标 (sklearn 需要二维数组)
        coords = sub_peaks['center'].values.reshape(-1, 1)
        global_indices = sub_peaks['global_idx'].values
        
        # === 核心优化: RadiusNeighborsTransformer ===
        # radius=250000: 只有距离 <= 250kb 的点才会被记录
        # mode='distance': 返回距离值
        transformer = RadiusNeighborsTransformer(radius=radius, mode='distance', metric='manhattan')
        
        # dist_matrix 是一个 csr_matrix
        # 包含 (i, j) 及其距离 d，其中 d <= 250000
        dist_matrix = transformer.fit_transform(coords)
        
        # 转为 COO 格式
        dist_coo = dist_matrix.tocoo()
        
        # === 计算 RP Score ===
        # 公式: 2^(-d / decay)
        # 此时 dist_coo.data 中的所有 d 已经保证 <= 250000
        scores = np.power(2.0, -(dist_coo.data / decay_distance))
        
        # 过滤掉极小值 (可选，例如 < 0.01) 以保持矩阵稀疏
        # 250kb / 50kb = 5个半衰期 -> 2^-5 = 0.03，所以通常不需要额外过滤
        
        # === 映射回全局索引 ===
        g_rows = global_indices[dist_coo.row]
        g_cols = global_indices[dist_coo.col]
        
        final_rows.append(g_rows)
        final_cols.append(g_cols)
        final_data.append(scores)

    print("--- 构建全局稀疏矩阵 ---")
    
    n_peaks = len(peaks_df)
    
    if len(final_data) > 0:
        all_rows = np.concatenate(final_rows)
        all_cols = np.concatenate(final_cols)
        all_data = np.concatenate(final_data)
        
        pp_matrix = sparse.csr_matrix(
            (all_data, (all_rows, all_cols)), 
            shape=(n_peaks, n_peaks)
        )
    else:
        print("警告: 未找到任何 250kb 内的 Peak 对。")
        pp_matrix = sparse.csr_matrix((n_peaks, n_peaks))

    print("--- 封装为 AnnData ---")
    
    adata_pp = ad.AnnData(
        X=pp_matrix,
        obs=pd.DataFrame(index=peaks_df.index), 
        var=pd.DataFrame(index=peaks_df.index)
    )
    
    adata_pp.uns['decay_distance'] = decay_distance
    adata_pp.uns['max_range'] = max_range
    adata_pp.uns['description'] = f"Peak-Peak RP Matrix (Range <= {max_range})"
    
    print(f"Region-Region regulation 完成! 矩阵维度: {adata_pp.shape}")
    print(f"非零连接数: {pp_matrix.nnz}, Sparsity: {100 * pp_matrix.nnz / (pp_matrix.shape[0] * pp_matrix.shape[1]):.4f}%")
    
    return adata_pp






def calculate_peak_peak_rp_distance(adata_atac, 
                           decay_distance=50000, # 推荐 50kb
                           max_range=250000):    # 硬性限制 250kb
    """
    计算 Peak-Peak 之间的调控潜力 (RP Score)。
    逻辑: 仅计算距离 <= 250kb 的 Peak 对，并应用 2^(-d/decay) 衰减。
    """
    print(f"\n--- 计算 Peak-Peak RP (Range={max_range/1000}kb, Decay={decay_distance/1000}kb) ---")
    
    # 1. 准备数据
    peaks_df = get_peak_coords(adata_atac)
    
    # 记录全局索引 (0, 1, 2...) 用于最后映射回大矩阵
    peaks_df['global_idx'] = range(len(peaks_df))
    
    final_rows = []
    final_cols = []
    final_data = []
    final_dist = []
    
    # 【核心修改 1】直接将搜索半径设为您的硬性限制 (250kb)
    radius = max_range 
    
    chromosomes = peaks_df['chrom'].unique()
    
    for chrom in tqdm(chromosomes, desc="Chromosomes"):
        sub_peaks = peaks_df[peaks_df['chrom'] == chrom]
        if sub_peaks.empty: continue
            
        # 提取坐标 (sklearn 需要二维数组)
        coords = sub_peaks['center'].values.reshape(-1, 1)
        global_indices = sub_peaks['global_idx'].values
        
        # === 核心优化: RadiusNeighborsTransformer ===
        # radius=250000: 只有距离 <= 250kb 的点才会被记录
        # mode='distance': 返回距离值
        transformer = RadiusNeighborsTransformer(radius=radius, mode='distance', metric='manhattan')
        
        # dist_matrix 是一个 csr_matrix
        # 包含 (i, j) 及其距离 d，其中 d <= 250000
        dist_matrix = transformer.fit_transform(coords)
        
        # 转为 COO 格式
        dist_coo = dist_matrix.tocoo()
        
        # === 计算 RP Score ===
        # 公式: 2^(-d / decay)
        # 此时 dist_coo.data 中的所有 d 已经保证 <= 250000
        scores = np.power(2.0, -(dist_coo.data / decay_distance))
        
        # 过滤掉极小值 (可选，例如 < 0.01) 以保持矩阵稀疏
        # 250kb / 50kb = 5个半衰期 -> 2^-5 = 0.03，所以通常不需要额外过滤
        
        # === 映射回全局索引 ===
        g_rows = global_indices[dist_coo.row]
        g_cols = global_indices[dist_coo.col]
        
        final_rows.append(g_rows)
        final_cols.append(g_cols)
        final_data.append(scores)
        final_dist.append(dist_coo.data)

    print("--- 构建全局稀疏矩阵 ---")
    
    n_peaks = len(peaks_df)
    
    if len(final_data) > 0:
        all_rows = np.concatenate(final_rows)
        all_cols = np.concatenate(final_cols)
        all_data = np.concatenate(final_data)
        all_dist = np.concatenate(final_dist)
        
        pp_matrix = sparse.csr_matrix(
            (all_data, (all_rows, all_cols)), 
            shape=(n_peaks, n_peaks)
        )
        
        pp_dist_matrix = sparse.csr_matrix(
            (all_dist, (all_rows, all_cols)), 
            shape=(n_peaks, n_peaks)
        )
    else:
        print("警告: 未找到任何 250kb 内的 Peak 对。")
        pp_matrix = sparse.csr_matrix((n_peaks, n_peaks))

    print("--- 封装为 AnnData ---")
    
    adata_pp = ad.AnnData(
        X=pp_matrix,
        obs=pd.DataFrame(index=peaks_df.index), 
        var=pd.DataFrame(index=peaks_df.index)
    )
    
    adata_pp_dist = ad.AnnData(
        X=pp_dist_matrix,
        obs=pd.DataFrame(index=peaks_df.index), 
        var=pd.DataFrame(index=peaks_df.index)
    )
    
    adata_pp.uns['decay_distance'] = decay_distance
    adata_pp.uns['max_range'] = max_range
    adata_pp.uns['description'] = f"Peak-Peak RP Matrix (Range <= {max_range})"
    
    adata_pp_dist.uns['decay_distance'] = decay_distance
    adata_pp_dist.uns['max_range'] = max_range
    adata_pp_dist.uns['description'] = f"Peak-Peak RP Matrix (Range <= {max_range})"
    
    print(f"Region-Region regulation 完成! 矩阵维度: {adata_pp.shape}")
    print(f"非零连接数: {pp_matrix.nnz}, Sparsity: {100 * pp_matrix.nnz / (pp_matrix.shape[0] * pp_matrix.shape[1]):.4f}%")
    
    return adata_pp, adata_pp_dist



####################################################################################################
#***************************----------------------------------------*******************************#
#***************************   计算anndata数据分布   *******************************#
#***************************----------------------------------------*******************************#
####################################################################################################



def analyze_score_distribution(adata, name="Data", quantiles=None, fig_flag = False):
    """
    步骤 1: 分析非零分数的分布，帮助确定阈值。
    
    参数:
        adata: AnnData 对象
        name: 数据名称 (用于打印和标题)
        quantiles: (可选) float 列表，例如 [0.1, 0.5, 0.9]。
                   范围应在 0 到 1 之间。
                   如果不传 (None)，则使用默认的一组分位数。
    """
    # 1. 提取非零值
    if sp.issparse(adata.X):
        scores = adata.X.data
    else:
        scores = adata.X[adata.X > 0].flatten()
        
    if len(scores) == 0:
        print(f"⚠️ {name} 中没有非零分数，无法分析。")
        return None

    # 2. 处理分位数设置
    if quantiles is None:
        # 默认分位数
        target_quantiles = [0.25, 0.5, 0.75, 0.90, 0.95]
    else:
        # 用户自定义，确保排序且去重
        target_quantiles = sorted(list(set(quantiles)))

    # 3. 计算基础统计量
    stats = {
        'min': np.min(scores),
        'mean': np.mean(scores),
        'std': np.std(scores),
        'max': np.max(scores)
    }
    
    # 4. 动态计算分位数
    # np.percentile 需要 0-100 的数值，所以 q * 100
    for q in target_quantiles:
        key = f"q{q}"  # 比如 q0.25
        stats[key] = np.percentile(scores, q * 100)

    # 5. 打印报告
    print(f"\n=== {name} Score Distribution Statistics ===")
    print(f"  > Count (Non-zero links): {len(scores)}")
    print(f"  > Mean:   {stats['mean']:.4f}")
    print(f"  > Std:    {stats['std']:.4f}")
    print("-" * 30)
    print(f"  > Min:    {stats['min']:.4f}")
    
    # 动态打印分位数
    for q in target_quantiles:
        key = f"q{q}"
        # 特殊标记一下中位数 (0.5)
        label = "Median" if q == 0.5 else f"{int(q*100)}%"
        print(f"  > {label:<7}: {stats[key]:.4f}")
        
    print(f"  > Max:    {stats['max']:.4f}")
    
    # 6. 画图
    if fig_flag:
        plt.figure(figsize=(10, 5))
        sns.histplot(scores, bins=50, kde=True, color='skyblue', alpha=0.6)
        
        # 标出均值
        plt.axvline(stats['mean'], color='red', linestyle='-', linewidth=2, 
                    label=f"Mean ({stats['mean']:.2f})")
        
        # 标出所有计算的分位数 (使用虚线)
        # 为了防止图例太乱，只给中位数或者特定的分位数加 label，其他的只画线
        colors = sns.color_palette("husl", len(target_quantiles))
        
        for i, q in enumerate(target_quantiles):
            key = f"q{q}"
            val = stats[key]
            
            # 仅对重要节点添加图例标签，防止图例爆炸
            if q == 0.5:
                label_str = f"Median ({val:.2f})"
                line_style = '--'
                color = 'green'
                lw = 2
            elif q in [0.05, 0.95, 0.99] or len(target_quantiles) <= 5:
                # 如果分位数不多，或者是非常极端的分位数，就显示标签
                label_str = f"q{q} ({val:.2f})"
                line_style = ':'
                color = colors[i]
                lw = 1.5
            else:
                label_str = None # 不显示图例
                line_style = ':'
                color = colors[i]
                lw = 1
                
            plt.axvline(val, color=color, linestyle=line_style, linewidth=lw, label=label_str)

        plt.title(f"Distribution of Non-zero Scores ({name})")
        plt.xlabel("Score Value")
        plt.legend()
        plt.show()
    
    return stats


