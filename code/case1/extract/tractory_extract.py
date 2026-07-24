# ============================================================
# extract_trajectories.py
# 从 GSE194122 h5ad 文件提取3条轨迹
# 输出格式对接你现有的 main_data_process.py 流程
# ============================================================

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
import os


# ============================================================
# extract_trajectories_sampled.py
# 按比例抽样，每条轨迹最多1000个细胞
# ============================================================

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
import os


# ----------------------------------------------------------------
# 核心函数：按比例抽样
# ----------------------------------------------------------------
def proportional_sample(obs_df, cell_type_col, cell_types, 
                         max_cells=1000, random_seed=42):
    """
    按各细胞类型的原始比例抽样，总数不超过 max_cells
    
    参数:
        obs_df:        adata.obs DataFrame
        cell_type_col: 细胞类型列名
        cell_types:    需要包含的细胞类型列表
        max_cells:     最大细胞总数
        random_seed:   随机种子
    
    返回:
        selected_cells: 抽取的细胞barcode列表
        sample_info:    各类型抽取数量的DataFrame
    """
    np.random.seed(random_seed)
    
    # 筛选目标细胞类型
    mask = obs_df[cell_type_col].isin(cell_types)
    subset = obs_df[mask].copy()
    total_available = len(subset)
    
    print(f"\n  可用细胞总数: {total_available}")
    print(f"  目标抽样总数: {max_cells}")
    
    # 计算各类型的原始数量和比例
    type_counts = subset[cell_type_col].value_counts()
    type_props  = type_counts / total_available
    
    # 按比例分配抽样数量
    if total_available <= max_cells:
        # 细胞数不足max_cells，全部取用
        print(f"  细胞总数({total_available}) ≤ max_cells({max_cells})，全部使用")
        selected_cells = subset.index.tolist()
        sample_info = pd.DataFrame({
            'cell_type':   type_counts.index,
            'original_n':  type_counts.values,
            'sampled_n':   type_counts.values,
            'proportion':  type_props.values
        })
    else:
        # 按比例分配，确保每类至少1个细胞
        allocated = (type_props * max_cells).apply(np.floor).astype(int)
        
        # 处理因取整导致总数不足的情况：将剩余名额分给比例最大的类型
        remainder = max_cells - allocated.sum()
        if remainder > 0:
            # 按小数部分从大到小补充
            fractional = (type_props * max_cells) - allocated
            top_types  = fractional.nlargest(remainder).index
            allocated[top_types] += 1
        
        # 确保不超过实际可用数量
        allocated = allocated.clip(upper=type_counts)
        
        print(f"\n  各细胞类型抽样分配:")
        selected_cells = []
        sample_records = []
        
        for ct in type_counts.index:
            n_sample = allocated[ct]
            ct_cells = subset[subset[cell_type_col] == ct].index.tolist()
            
            if n_sample >= len(ct_cells):
                chosen = ct_cells
            else:
                chosen = np.random.choice(ct_cells, size=n_sample, 
                                           replace=False).tolist()
            
            selected_cells.extend(chosen)
            sample_records.append({
                'cell_type':  ct,
                'original_n': len(ct_cells),
                'sampled_n':  len(chosen),
                'proportion': f"{type_props[ct]:.3f}"
            })
            print(f"    {ct:30s}: {len(ct_cells):5d} → {len(chosen):4d} "
                  f"({type_props[ct]*100:.1f}%)")
        
        sample_info = pd.DataFrame(sample_records)
    
    print(f"\n  最终抽样细胞数: {len(selected_cells)}")
    return selected_cells, sample_info


# ----------------------------------------------------------------
# Step 4: 提取轨迹、抽样并保存
# ----------------------------------------------------------------
def save_trajectory(adata_rna, adata_atac, selected_cells, 
                     traj_name, output_root, sample_info):
    """
    保存轨迹数据为pipeline所需格式
    """
    out_dir = os.path.join(output_root, traj_name, "process/")
    os.makedirs(out_dir + "R/rna/",  exist_ok=True)
    os.makedirs(out_dir + "R/atac/", exist_ok=True)
    
    # 提取子集
    rna_sub  = adata_rna[selected_cells].copy()
    atac_sub = adata_atac[selected_cells].copy()
    
    # 处理RNA counts层
    if 'counts' in rna_sub.layers:
        counts_mtx = rna_sub.layers['counts']
    else:
        counts_mtx = rna_sub.X.copy()
    rna_sub.layers['counts'] = counts_mtx
    
    # 标准化ATAC峰名为 chr:start-end 格式
    peak_names = atac_sub.var_names.tolist()
    if len(peak_names) > 0 and ':' not in peak_names[0]:
        # chr1-1000-2000 → chr1:1000-2000
        new_names = []
        for name in peak_names:
            parts = name.split('-')
            if len(parts) >= 3:
                new_names.append(f"{parts[0]}:{parts[1]}-{parts[2]}")
            else:
                new_names.append(name)
        atac_sub.var_names = new_names
        print(f"  峰名转换: {peak_names[0]} → {new_names[0]}")
    
    # 保存 h5ad（对接 main_data_process.py）
    rna_sub.write_h5ad(out_dir + "rna_origin.h5ad")
    atac_sub.write_h5ad(out_dir + "atac_origin.h5ad")
    
    # 保存 R 格式（对接 trajectory_inference.R）
    rna_dir  = out_dir + "R/rna/"
    atac_dir = out_dir + "R/atac/"
    
    sio.mmwrite(rna_dir  + "sparse.mtx", counts_mtx)
    sio.mmwrite(atac_dir + "sparse.mtx", atac_sub.X)
    
    pd.DataFrame({'V1': rna_sub.obs_names}).to_csv(
        rna_dir + "cellinfo.csv", index=False)
    pd.DataFrame({'Geneid': rna_sub.var_names}).to_csv(
        rna_dir + "rnainfo.csv",  index=False)
    rna_sub.obs.to_csv(rna_dir + "cell_meta.csv")
    
    pd.DataFrame({'V1': atac_sub.obs_names}).to_csv(
        atac_dir + "cellinfo.csv", index=False)
    pd.DataFrame({'V1': atac_sub.var_names}).to_csv(
        atac_dir + "atacinfo.csv", index=False)
    atac_sub.obs.to_csv(atac_dir + "cell_meta.csv")
    
    # 保存抽样统计信息
    sample_info.to_csv(out_dir + "sample_info.csv", index=False)
    
    print(f"\n  保存完成 → {out_dir}")
    print(f"  RNA:  {rna_sub.shape}")
    print(f"  ATAC: {atac_sub.shape}")
    
    return rna_sub, atac_sub







if __name__ == "__main__":
    # ----------------------------------------------------------------
    # 路径设置
    # ----------------------------------------------------------------
    data_path = "/home/wuyan/dygmamba_project/Real/Claude/Other/case_study/plan1/case3/raw/"
    
    h5ad_file = data_path + "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
    
    output_root = data_path + "process/"

    # ----------------------------------------------------------------
    # Step 1: 读取数据
    # ----------------------------------------------------------------
    print("读取h5ad文件...")
    adata = sc.read_h5ad(h5ad_file)
    print(f"总细胞数: {adata.n_obs}")
    print(f"\n细胞类型分布:")
    print(adata.obs['cell_type'].value_counts())

    # ----------------------------------------------------------------
    # Step 2: 分离RNA和ATAC
    # ----------------------------------------------------------------
    if 'feature_types' in adata.var.columns:
        rna_mask  = adata.var['feature_types'] == 'GEX'
        atac_mask = adata.var['feature_types'] == 'ATAC'
    elif 'modality' in adata.var.columns:
        rna_mask  = adata.var['modality'] == 'GEX'
        atac_mask = adata.var['modality'] == 'ATAC'
    else:
        rna_mask  = ~adata.var_names.str.contains('chr')
        atac_mask =  adata.var_names.str.contains('chr')

    adata_rna  = adata[:, rna_mask].copy()
    adata_atac = adata[:, atac_mask].copy()
    print(f"\nRNA:  {adata_rna.shape}")
    print(f"ATAC: {adata_atac.shape}")

    # ----------------------------------------------------------------
    # Step 3: 定义3条轨迹
    # ----------------------------------------------------------------
    trajectories = {
        "myeloid": [
            "HSC", "G/M prog", "ID2-hi myeloid prog", "CD14+ Mono", "CD16+ Mono"
        ],
        "erythroid": [
            "HSC", "MK/E prog", "Erythroblast"
        ],
        "Bcell": [
            "HSC", "Lymph prog", "B1 B", "Transitional B", "Naive CD20+ B"
        ]
    }




    # ----------------------------------------------------------------
    # Step 5: 主流程
    # ----------------------------------------------------------------
    results = {}

    for traj_name, cell_types in trajectories.items():
        
        print(f"\n{'='*60}")
        print(f"轨迹: {traj_name}")
        print(f"细胞类型: {cell_types}")
        
        # 按比例抽样
        selected_cells, sample_info = proportional_sample(
            obs_df        = adata_rna.obs,
            cell_type_col = 'cell_type',
            cell_types    = cell_types,
            max_cells     = 1000,
            random_seed   = 42
        )
        
        # 保存
        rna_sub, atac_sub = save_trajectory(
            adata_rna, adata_atac,
            selected_cells, traj_name,
            output_root, sample_info
        )
        
        results[traj_name] = {
            'cells': selected_cells,
            'rna':   rna_sub,
            'atac':  atac_sub,
            'info':  sample_info
        }

    # ----------------------------------------------------------------
    # Step 6: 打印汇总
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("抽样汇总:")
    print(f"{'轨迹':<12} {'细胞数':>8} {'基因数':>8} {'峰数':>10}")
    print('-' * 42)
    for traj_name, res in results.items():
        print(f"{traj_name:<12} "
            f"{res['rna'].n_obs:>8} "
            f"{res['rna'].n_vars:>8} "
            f"{res['atac'].n_vars:>10}")

    print("\n各轨迹详细抽样信息已保存至 sample_info.csv")
    print("下一步：对每条轨迹运行 trajectory_inference.R")