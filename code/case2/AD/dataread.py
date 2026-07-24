"""
io_10x_multiome.py
读取 10x Cell Ranger ARC Multiome 输出 → 配对 (adata_rna, adata_atac)

支持的文件:
  *_filtered_feature_bc_matrix.h5   (RNA + ATAC 合并矩阵)
  *_atac_fragments.tsv.gz           (ATAC fragments)
  *_raw_feature_bc_matrix.h5        (同样适用)
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc


# ---------------------------------------------------------------------------
# 主函数 1: 从单个 h5 拆出 RNA + ATAC
# ---------------------------------------------------------------------------
def read_10x_multiome_h5(
    h5_path: str,
    fragments_path: Optional[str] = None,
    make_var_names_unique: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    读取 10x Multiome 的 filtered/raw feature_bc_matrix.h5，拆成 RNA 和 ATAC。

    Parameters
    ----------
    h5_path : str
        路径，例如 'Multiome_RNA_ATAC_..._filtered_feature_bc_matrix.h5'
    fragments_path : str, optional
        ATAC fragments.tsv.gz 路径，会登记到 adata_atac.uns['files']['fragments']
        (snapatac2 / muon / archr 的下游分析都依赖这个约定)
    make_var_names_unique : bool
        基因名是否去重（RNA 端的常见预处理）

    Returns
    -------
    adata_rna  : AnnData (cells × genes),  X = raw counts
    adata_atac : AnnData (cells × peaks),  X = raw counts, var 含 chr/start/end
    """
    h5_path = str(h5_path)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    # gex_only=False → 一次性读出 RNA + ATAC,然后按 feature_types 拆
    adata = sc.read_10x_h5(h5_path, gex_only=False)
    adata.var_names_make_unique() if make_var_names_unique else None

    if "feature_types" not in adata.var.columns:
        raise ValueError(
            f"{h5_path} 缺少 feature_types 列,这看起来不是 multiome h5"
        )

    # ---- 拆 RNA ----
    rna_mask = adata.var["feature_types"] == "Gene Expression"
    adata_rna = adata[:, rna_mask].copy()
    adata_rna.var = adata_rna.var.drop(columns=["feature_types"])

    # ---- 拆 ATAC ----
    atac_mask = adata.var["feature_types"] == "Peaks"
    adata_atac = adata[:, atac_mask].copy()
    adata_atac.var = adata_atac.var.drop(columns=["feature_types"])

    # 解析 peak 区间 → chr / start / end (peak 名通常是 "chr1:100-200" 或 "chr1-100-200")
    chrom, start, end = _parse_peak_intervals(adata_atac.var_names)
    adata_atac.var["chr"] = chrom
    adata_atac.var["start"] = start
    adata_atac.var["end"] = end

    # ---- 配对一致性: 同一份 obs ----
    assert (adata_rna.obs_names == adata_atac.obs_names).all(), \
        "RNA 和 ATAC 的 cell barcode 不一致 (异常)"

    # ---- 注册 fragments 文件 (下游分析用) ----
    if fragments_path is not None:
        if not os.path.exists(fragments_path):
            raise FileNotFoundError(fragments_path)
        adata_atac.uns["files"] = {"fragments": os.path.abspath(fragments_path)}
        # 顺便检查 tabix index 是否存在
        tbi = fragments_path + ".tbi"
        if not os.path.exists(tbi):
            print(f"  [warn] tabix 索引不存在: {tbi}")
            print(f"         建议: tabix -p bed {fragments_path}")

    # ---- 简要打印 ----
    print(f"  RNA  : {adata_rna.shape[0]} cells × {adata_rna.shape[1]} genes")
    print(f"  ATAC : {adata_atac.shape[0]} cells × {adata_atac.shape[1]} peaks")
    if fragments_path:
        print(f"  Frag : {fragments_path}")

    return adata_rna, adata_atac


# ---------------------------------------------------------------------------
# 主函数 2: 自动从一个目录里找配对文件
# ---------------------------------------------------------------------------
def read_10x_multiome_dir(
    data_dir: str,
    prefix: Optional[str] = None,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    在目录里自动找 *_filtered_feature_bc_matrix.h5 和 *_atac_fragments.tsv.gz

    Parameters
    ----------
    data_dir : str
        包含 multiome 文件的目录
    prefix : str, optional
        如果目录里有多个样本,用 prefix 限定 (例如 "Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote")

    Returns
    -------
    adata_rna, adata_atac
    """
    data_dir = Path(data_dir)

    # 找 h5: 优先 filtered, 退而 raw
    if prefix:
        h5_candidates = (
            list(data_dir.glob(f"{prefix}*filtered_feature_bc_matrix.h5"))
            or list(data_dir.glob(f"{prefix}*raw_feature_bc_matrix.h5"))
        )
        frag_candidates = list(data_dir.glob(f"{prefix}*atac_fragments.tsv.gz"))
    else:
        h5_candidates = (
            list(data_dir.glob("*filtered_feature_bc_matrix.h5"))
            or list(data_dir.glob("*raw_feature_bc_matrix.h5"))
        )
        frag_candidates = list(data_dir.glob("*atac_fragments.tsv.gz"))

    if not h5_candidates:
        raise FileNotFoundError(f"在 {data_dir} 找不到 *feature_bc_matrix.h5")
    if len(h5_candidates) > 1:
        raise ValueError(
            f"找到多个 h5,请用 prefix 限定: {[p.name for p in h5_candidates]}"
        )

    h5_path = str(h5_candidates[0])
    frag_path = str(frag_candidates[0]) if frag_candidates else None

    print(f"[读取 Multiome]")
    print(f"  H5:        {h5_path}")
    print(f"  Fragments: {frag_path}")

    return read_10x_multiome_h5(h5_path, fragments_path=frag_path)


# ---------------------------------------------------------------------------
# 工具: 解析 peak 名 → chr / start / end
# ---------------------------------------------------------------------------
def _parse_peak_intervals(peak_names):
    """
    支持两种命名格式:
      'chr1:100-200'   (Cell Ranger ARC 默认)
      'chr1-100-200'   (有些下游工具会重命名成这种)
    """
    chrom, start, end = [], [], []
    # 同时匹配 ':' 和 '-' 作为染色体和位置的分隔符
    pat = re.compile(r"^(chr[^:_\-]+|[^:_\-]+)[:_\-](\d+)[-_](\d+)$")
    for name in peak_names:
        m = pat.match(str(name))
        if m:
            chrom.append(m.group(1))
            start.append(int(m.group(2)))
            end.append(int(m.group(3)))
        else:
            chrom.append(np.nan)
            start.append(-1)
            end.append(-1)
    return np.array(chrom), np.array(start), np.array(end)


if __name__ == "__main__":

    data_path = "/home/wuyan/dygmamba_project/NewRealPlan/case2/data/AD/"
    
    rna_path = data_path + "Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote_filtered_feature_bc_matrix.h5"
    atac_path = data_path + "Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote_atac_fragments.tsv.gz"
    
    output_path = "/home/wuyan/dygmamba_project/NewRealPlan/case2/data/AD/process/"

    adata_rna, adata_atac = read_10x_multiome_h5(
        h5_path= rna_path,
        fragments_path= atac_path,
    )

    adata_rna.write_h5ad(output_path + "rna_origin.h5ad")
    adata_atac.write_h5ad(output_path + "atac_origin.h5ad")
    
    # 之后就和你 BMMC pipeline 一样
    print(adata_rna.var.head())     # gene_ids, feature_types
    print(adata_atac.var.head())    # chr, start, end
    print(adata_atac.uns["files"])  # {'fragments': '...'}