import os
import sys
import glob
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.io as sio
import scipy.sparse as sp


def check_consistency_cell(adata_rna, adata_atac, pseudotime):
    
    # 建立 小写 → 原始名称 的映射
    rna_lower2orig  = {str(c).lower().strip(): c for c in adata_rna.obs_names}
    atac_lower2orig = {str(c).lower().strip(): c for c in adata_atac.obs_names}
    pt_lower2orig   = {str(c).lower().strip(): c for c in pseudotime.index}

    # 用小写做交集
    common_lower = (set(rna_lower2orig.keys())
                    & set(atac_lower2orig.keys())
                    & set(pt_lower2orig.keys()))
    
    print(f"RNA  cells : {len(rna_lower2orig)}")
    print(f"ATAC cells : {len(atac_lower2orig)}")
    print(f"Pseudotime cells: {len(pt_lower2orig)}")
    print(f"共同 cells : {len(common_lower)}")

    # 用各自原始名称索引
    rna_cells  = [rna_lower2orig[c]  for c in common_lower]
    atac_cells = [atac_lower2orig[c] for c in common_lower]
    pt_cells   = [pt_lower2orig[c]   for c in common_lower]

    adata_rna  = adata_rna[rna_cells].copy()
    adata_atac = adata_atac[atac_cells].copy()
    pseudotime = pseudotime.loc[pt_cells].copy()

    return adata_rna, adata_atac, pseudotime


def calculate_regulation_prior(adata_rna, adata_atac):
    # ## get gene and peak info

    outprefix = "Regulation"
    model = "Enhanced"
    genedistance = 100000

    species = "GRCh38" 
    split_str="-"
    gene_bed = os.path.join(annotation_path, species + "_refgenes.txt")

    genes_list = []
    genes_info = ExtractGeneInfo(gene_bed)
    genes_info_tss = list()
    genes_info_full = list() ### [chrom, tss, start, end, 1, unique_id]
    all_genes = set(adata_rna.var_names)
    seen = set()
    gene_num = 0
    for igene in range(len(genes_info)):
        tmp_gene = genes_info[igene]
        temp_gene = tmp_gene[-1].split("@")[0]
        if temp_gene in all_genes and temp_gene not in seen :
            seen.add(temp_gene)
            genes_list.append(temp_gene)
            genes_info_full.append(tmp_gene + [gene_num])
            genes_info_tss.append([tmp_gene[0], tmp_gene[3], tmp_gene[1], tmp_gene[2]] + tmp_gene[4:] + [gene_num])
            gene_num += 1

    genes_info_Data = pd.DataFrame(genes_info_full)
    genes_info_Data.columns = ['chrom', 'start', 'end', 'tss', 'promoter',
                                'exons', 'length', 'flag 1', 'uid','gene_id']
    genes_info_Data["gene symbol"] = genes_list

    #############################################
    # get peak info
    peaks_list = [f for f in adata_atac.var_names]
    peaks_info = []
    for ipeak, peak in enumerate(peaks_list):
        peaks_tmp = peak.rsplit(split_str, maxsplit=2)
        peaks_info.append([peaks_tmp[0], \
                            (int(peaks_tmp[1])+int(peaks_tmp[2]))/2.0, \
                            int(peaks_tmp[1]), int(peaks_tmp[2]), \
                            0, peak, ipeak])

    peaks_info_Data = pd.DataFrame(peaks_info)
    peaks_info_Data.columns = ['chrom', 'center','start', 'end', 'floag 0', 'peak', 'peak_id']


    # ## 获取 gene-peak 调控潜力 


    id_peak_df = peaks_info_Data[['peak', 'peak_id']]
    id_peak_df.columns = ["peak", "id"]
    id_peak_df.set_index("peak", inplace=True)

    id_gene_df = genes_info_Data[['gene symbol','gene_id']]
    id_gene_df.columns = ['gene', 'id']
    id_gene_df.set_index("gene", inplace=True)

    id_gene_df = id_gene_df[~id_gene_df.index.duplicated(keep="first")]
    id_peak_df = id_peak_df[~id_peak_df.index.duplicated(keep="first")]

    genes_peaks_score_dok = RP_AddExonRemovePromoter(peaks_info, genes_info_full, genes_info_tss, float(genedistance))

    adata_rp_gene_peak = ad.AnnData(X= genes_peaks_score_dok,
                                        obs = pd.DataFrame(index=genes_info_Data["gene symbol"]),
                                        var = pd.DataFrame(index=peaks_info_Data["peak"]))

    adata_rp_gene_peak.X = adata_rp_gene_peak.X.tocsr().sign().astype(int)

    peak_score_dok = RP_peaks(peaks_info, float(genedistance))

    adata_rp_peak = ad.AnnData(X= peak_score_dok,
                                        obs = pd.DataFrame(index=peaks_info_Data["peak"]),
                                        var = pd.DataFrame(index=peaks_info_Data["peak"]))

    adata_rp_peak.X = adata_rp_peak.X.tocsr().sign().astype(int)

    return adata_rp_gene_peak, adata_rp_peak







def sample_cells_by_pseudotime(adata_rna, adata_atac, pseudotime, N=1000):
    """在伪时间轴上等间隔采样 N 个细胞，通过小写 barcode 匹配三者"""
    
    # 建立 小写barcode → 各自原始名称 的映射
    rna_lower2orig  = {str(c).lower().strip(): c for c in adata_rna.obs_names}
    atac_lower2orig = {str(c).lower().strip(): c for c in adata_atac.obs_names}
    pt_lower2orig   = {str(c).lower().strip(): c for c in pseudotime.index}
    
    # 三者交集（小写）
    common_lower = list(
        set(rna_lower2orig) & set(atac_lower2orig) & set(pt_lower2orig)
    )
    
    if len(common_lower) <= N:
        print(f"共同细胞数 ({len(common_lower)}) ≤ N ({N})，跳过采样，返回全部共同细胞")
        rna_cells  = [rna_lower2orig[c]  for c in common_lower]
        atac_cells = [atac_lower2orig[c] for c in common_lower]
        pt_cells   = [pt_lower2orig[c]   for c in common_lower]
        
        adata_rna  = adata_rna[rna_cells].copy()
        adata_atac = adata_atac[atac_cells].copy()
        pseudotime = pseudotime.loc[pt_cells].copy()
        
        unified = [c for c in common_lower]
        adata_rna.obs_names  = unified
        adata_atac.obs_names = unified
        pseudotime.index     = unified
        return adata_rna, adata_atac, pseudotime
    
    # 以 pseudotime 的值为基准做等间隔采样
    pt_vals = {c: pseudotime.loc[pt_lower2orig[c], 'pseudotime'] for c in common_lower}
    pt_min, pt_max = min(pt_vals.values()), max(pt_vals.values())
    targets = np.linspace(pt_min, pt_max, N)
    
    remaining = set(common_lower)
    selected = []
    
    for t in targets:
        if not remaining:
            break
        best = min(remaining, key=lambda c: abs(pt_vals[c] - t))
        selected.append(best)
        remaining.remove(best)
    
    # 按伪时间排序
    selected = sorted(selected, key=lambda c: pt_vals[c])
    
    # 用各自原始名称索引
    rna_cells  = [rna_lower2orig[c]  for c in selected]
    atac_cells = [atac_lower2orig[c] for c in selected]
    pt_cells   = [pt_lower2orig[c]   for c in selected]
    
    adata_rna  = adata_rna[rna_cells].copy()
    adata_atac = adata_atac[atac_cells].copy()
    pseudotime = pseudotime.loc[pt_cells].copy()
    
    # 统一 barcode 为小写
    adata_rna.obs_names  = selected
    adata_atac.obs_names = selected
    pseudotime.index     = selected
    
    print(f"共同细胞: {len(common_lower)}")
    print(f"采样后: {len(selected)} 细胞")
    
    return adata_rna, adata_atac, pseudotime










def recompute_umap(adata_rna, n_pcs=20):
    """在子集上重新计算 PCA + UMAP"""
    print("  重新计算 PCA + UMAP...")
    
    if 'highly_variable' not in adata_rna.var.columns:
        try:
            sc.pp.highly_variable_genes(
                adata_rna, n_top_genes=3000, flavor='seurat_v3',
                layer='counts' if 'counts' in adata_rna.layers else None)
        except Exception:
            sc.pp.highly_variable_genes(adata_rna, n_top_genes=3000)
    
    n_comps = min(30, adata_rna.shape[0] - 1, adata_rna.shape[1] - 1)
    sc.pp.pca(adata_rna, n_comps=n_comps)
    
    n_neighbors = min(15, adata_rna.shape[0] - 1)
    sc.pp.neighbors(adata_rna, n_pcs=min(n_pcs, n_comps), n_neighbors=n_neighbors)
    sc.tl.umap(adata_rna)
    
    print(f"    UMAP: {adata_rna.obsm['X_umap'].shape}")
    return adata_rna


def export_to_R(adata_rna, adata_atac, output_path):
    """导出 R 格式"""
    rna_dir = os.path.join(output_path, "R", "rna")
    os.makedirs(rna_dir, exist_ok=True)
    adata_rna.obs.to_csv(os.path.join(rna_dir, "cellinfo.csv"))
    adata_rna.var.to_csv(os.path.join(rna_dir, "rnainfo.csv"))
    mtx = adata_rna.layers['counts'] if 'counts' in adata_rna.layers else adata_rna.X
    if not sp.issparse(mtx):
        mtx = sp.csr_matrix(mtx)
    sio.mmwrite(os.path.join(rna_dir, "sparse.mtx"), mtx)
    
    atac_dir = os.path.join(output_path, "R", "atac")
    os.makedirs(atac_dir, exist_ok=True)
    adata_atac.obs.to_csv(os.path.join(atac_dir, "cellinfo.csv"))
    adata_atac.var.to_csv(os.path.join(atac_dir, "atacinfo.csv"))
    atac_mtx = adata_atac.X if sp.issparse(adata_atac.X) else sp.csr_matrix(adata_atac.X)
    sio.mmwrite(os.path.join(atac_dir, "sparse.mtx"), atac_mtx)
    
    print(f"  R 格式已导出: {os.path.join(output_path, 'R')}")



if __name__ == "__main__":
    
    ######################################################################
    # ==================== 在这里设置参数 ====================
    ######################################################################
    data_root = "/home/wuyan/dygmamba_project/NewRealPlan/case2/data/AD/process/"
    traj = 'model1_CRND8_Microglia'
    data_path = data_root + traj + "/process/"
    
    max_cells = 1000      # 采样后的最大细胞数
    n_bins    = 20        # 伪时序等间隔分 bin 数（每个 bin 采 max_cells/n_bins 个细胞）
    seed      = 42
    
    ######################################################################
    
    print("=" * 60)
    print("按伪时序等间隔采样")
    print(f"  数据路径:   {data_path}")
    print(f"  目标细胞数: {max_cells}")
    print(f"  分 bin 数:  {n_bins} ({max_cells // n_bins} cells/bin)")
    print("=" * 60)
    
    # ---- 1. 加载数据 ----

    adata_rna  = ad.read_h5ad(data_path + "rna_processed.h5ad")
    adata_atac = ad.read_h5ad(data_path + "atac_processed.h5ad")
    pseudotime = pd.read_csv(data_path + "avg_lineage_pseudotime.csv")

    pseudotime['cell_barcode'] = pseudotime['cell_barcode'].str.lower()

    pseudotime.set_index("cell_barcode", inplace=True)
    
    
    print(f"  RNA:  {adata_rna.shape}")
    print(f"  ATAC: {adata_atac.shape}")
    
        # ---- 3. 按伪时序等间隔采样 ----
    print(f"\n[3/4] 按伪时序等间隔采样...")
    
    # 备份采样前数据（只备份一次）
    rna_backup = data_path + "rna_before_downsample.h5ad"
    if not os.path.exists(rna_backup):
        adata_rna.write_h5ad(rna_backup)
        adata_atac.write_h5ad(data_path + "atac_before_downsample.h5ad")
        print(f"  备份已保存: *_before_downsample.h5ad")
        
    
    adata_rna, adata_atac, pseudotime = check_consistency_cell(adata_rna, adata_atac, pseudotime)
    
    # 等间隔采样 1000 个细胞
    sub_adata_rna, sub_adata_atac, sub_pseudotime = sample_cells_by_pseudotime(
        adata_rna, adata_atac, pseudotime, N=1000
    )
    
    # 建立小写→伪时序的映射
    pt_lower = sub_pseudotime.copy()
    pt_lower.index = pt_lower.index.str.lower().str.strip()

    # 按小写对齐
    rna_lower = sub_adata_rna.obs_names.str.lower().str.strip()
    atac_lower = sub_adata_atac.obs_names.str.lower().str.strip()

    sub_adata_rna.obs["pseudotime"]  = pt_lower['pseudotime'].reindex(rna_lower).values
    sub_adata_atac.obs["pseudotime"] = pt_lower['pseudotime'].reindex(atac_lower).values

    # 检查有没有没对上的
    n_nan = sub_adata_rna.obs["pseudotime"].isna().sum()
    if n_nan > 0:
        print(f"[警告] {n_nan} 个细胞没有匹配到伪时序")

    sub_adata_rna.obs["pseudotime"]  = sub_pseudotime['pseudotime']
    sub_adata_atac.obs["pseudotime"] = sub_pseudotime['pseudotime']
    
    
    # ---- 4. 重新 UMAP + 保存 + 导出 R ----
    print(f"\n[4/4] 重新 UMAP + 保存...")
    sub_adata_rna = recompute_umap(sub_adata_rna)
    
    # 导出子集 UMAP 给 R
    umap_df = pd.DataFrame(
        sub_adata_rna.obsm['X_umap'],
        columns=['UMAP_1', 'UMAP_2'],
        index=sub_adata_rna.obs_names
    )
    umap_df.to_csv(data_path + "umap_coords.csv")
    
    # 覆盖保存
    sub_adata_rna.write_h5ad(data_path + "rna_processed.h5ad")
    sub_adata_atac.write_h5ad(data_path + "atac_processed.h5ad")
    print(f"  已保存 rna_processed.h5ad  ({sub_adata_rna.shape})")
    print(f"  已保存 atac_processed.h5ad ({sub_adata_atac.shape})")
    
    # 导出 R 格式
    export_to_R(sub_adata_rna, sub_adata_atac, data_path)
    
    os._exit(0)