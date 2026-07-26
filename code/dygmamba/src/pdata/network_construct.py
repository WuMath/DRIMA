import pandas as pd
import pybedtools
from tqdm import tqdm

# ==============================================================================
# Part 1: 构建 TF-Peak 调控网络 (基于ChIP-seq和ATAC-seq的重叠)
# ==============================================================================
def build_tf_peak_network(tf_motif_df, atac_adata):
    """
    通过计算TF ChIP-seq峰与ATAC-seq峰的重叠来构建TF-Peak网络。

    Args:
        tf_motif_df (pd.DataFrame): 包含TF ChIP-seq峰的数据框，
                                    需要'chrom', 'start', 'end', 'tf'这几列。
        atac_adata (anndata.AnnData): ATAC-seq的AnnData对象，
                                      .var索引是peak坐标 (e.g., 'chr1:100-500')。

    Returns:
        pd.DataFrame: 一个表示TF-Peak连接的DataFrame。
    """
    print("--- 1. 构建 TF-Peak 网络 ---")

    atac_peaks_df = pd.DataFrame(index=atac_adata.var.index)
    atac_peaks_df[['chrom', 'start', 'end']] = atac_peaks_df.index.to_series().str.split('[:-]', expand=True)
    atac_peaks_df = atac_peaks_df.reset_index().rename(columns={'index': 'peak_id'})
    
    tf_bed = pybedtools.BedTool.from_dataframe(tf_motif_df[['chrom', 'start', 'end', 'tf']])
    atac_bed = pybedtools.BedTool.from_dataframe(atac_peaks_df[['chrom', 'start', 'end', 'peak_id']])
    
    intersections = tf_bed.intersect(atac_bed, wa=True, wb=True)
    
    intersection_df = intersections.to_dataframe(names=[
        'tf_chrom', 'tf_start', 'tf_end', 'tf', 
        'peak_chrom', 'peak_start', 'peak_end', 'peak_id'
    ])
    
    tf_peak_network = intersection_df[['tf', 'peak_id']].drop_duplicates().reset_index(drop=True)
    
    return tf_peak_network

# ==============================================================================
# Part 2: 构建 Peak-Gene 调控网络 (基于Hi-C数据)
# ==============================================================================

def build_peak_gene_network_mem_efficient(hic_df, atac_adata, gene_info_df):
    """
    Args:
        hic_df (pd.DataFrame): 包含Hi-C互作的数据框，需要'chr1', 'start1', 'end1', 'chr2', 'start2', 'end2'列。
        atac_adata (anndata.AnnData): ATAC-seq的AnnData对象。
        gene_info_df (pd.DataFrame): 包含基因注释的数据框，需要'gene_name', 'seqname', 'start', 'end', 'strand'列。

    Returns:
        pd.DataFrame: 一个表示Peak-Gene连接的DataFrame。
    """
    print("\n--- 2. 构建 Peak-Gene 网络 (内存优化版) ---")

    # --- 1. 准备输入数据 ---
    print("准备ATAC peaks和基因启动子区域...")
    
    atac_peaks_df = pd.DataFrame(index=atac_adata.var.index)
    split_df = atac_peaks_df.index.to_series().str.split('[:-]', expand=True)
    atac_peaks_df['chrom'] = split_df[0]
    atac_peaks_df['start'] = split_df[1]
    atac_peaks_df['end'] = split_df[2]
    atac_peaks_df = atac_peaks_df.reset_index().rename(columns={'index': 'peak_id'})
    atac_bed = pybedtools.BedTool.from_dataframe(atac_peaks_df[['chrom', 'start', 'end', 'peak_id']])

    promoters_df = gene_info_df.copy()
    promoters_df['tss'] = promoters_df.apply(lambda row: row['start'] if row['strand'] == '+' else row['end'], axis=1)
    promoters_df['start'] = (promoters_df['tss'] - 2000).clip(lower=0)
    promoters_df['end'] = promoters_df['tss'] + 2000
    promoters_bed = pybedtools.BedTool.from_dataframe(promoters_df[['seqname', 'start', 'end', 'gene_name']])

    hic_df['interaction_id'] = range(len(hic_df))
    
    hic_side1_bed = pybedtools.BedTool.from_dataframe(hic_df[['chr1', 'start1', 'end1', 'interaction_id']])
    hic_side2_bed = pybedtools.BedTool.from_dataframe(hic_df[['chr2', 'start2', 'end2', 'interaction_id']])

    # --- 3. 分别计算与Peak和Promoter的重叠 ---
    # Case 1: side1 与 peak 重叠, side2 与 promoter 重叠
    print("情况1: 寻找 side1-peak 和 side2-promoter 的连接...")
    side1_peaks = hic_side1_bed.intersect(atac_bed, wa=True, wb=True).to_dataframe(
        # --- 修正点: 给予唯一的列名 ---
        names=['hic1_chr', 'hic1_start', 'hic1_end', 'interaction_id', 'peak_chr', 'peak_start', 'peak_end', 'peak_id']
    )
    side2_promoters = hic_side2_bed.intersect(promoters_bed, wa=True, wb=True).to_dataframe(
        # --- 修正点: 给予唯一的列名 ---
        names=['hic2_chr', 'hic2_start', 'hic2_end', 'interaction_id', 'promo_chr', 'promo_start', 'promo_end', 'gene_name']
    )
    
    merged1 = pd.merge(
        side1_peaks[['interaction_id', 'peak_id']],
        side2_promoters[['interaction_id', 'gene_name']],
        on='interaction_id'
    )

    # Case 2: side2 与 peak 重叠, side1 与 promoter 重叠 (对称情况)
    print("情况2: 寻找 side2-peak 和 side1-promoter 的连接...")
    side2_peaks = hic_side2_bed.intersect(atac_bed, wa=True, wb=True).to_dataframe(
        # --- 修正点: 给予唯一的列名 ---
        names=['hic2_chr', 'hic2_start', 'hic2_end', 'interaction_id', 'peak_chr', 'peak_start', 'peak_end', 'peak_id']
    )
    side1_promoters = hic_side1_bed.intersect(promoters_bed, wa=True, wb=True).to_dataframe(
        # --- 修正点: 给予唯一的列名 ---
        names=['hic1_chr', 'hic1_start', 'hic1_end', 'interaction_id', 'promo_chr', 'promo_start', 'promo_end', 'gene_name']
    )
    
    merged2 = pd.merge(
        side2_peaks[['interaction_id', 'peak_id']],
        side1_promoters[['interaction_id', 'gene_name']],
        on='interaction_id'
    )

    # --- 4. 合并最终结果 ---
    print("合并所有找到的连接...")
    peak_gene_network = pd.concat([merged1, merged2])
    
    peak_gene_network = peak_gene_network[['peak_id', 'gene_name']].drop_duplicates().reset_index(drop=True)
    
    print(f"成功构建Peak-Gene网络，共找到 {len(peak_gene_network)} 条连接。")
    return peak_gene_network




# ==============================================================================
# Part 3: 构建 TF-Gene 调控网络 (推断)
# ==============================================================================
def build_tf_gene_network(tf_peak_network, peak_gene_network):
    """
    通过合并TF-Peak和Peak-Gene网络来推断TF-Gene调控关系。

    Args:
        tf_peak_network (pd.DataFrame): TF-Peak连接。
        peak_gene_network (pd.DataFrame): Peak-Gene连接。

    Returns:
        pd.DataFrame: 一个表示TF-Gene连接的DataFrame。
    """
    print("\n--- 3. 构建 TF-Gene 网络 ---")
    
    tf_gene_network = pd.merge(
        tf_peak_network,
        peak_gene_network,
        on='peak_id'
    )
    
    tf_gene_network = tf_gene_network[['tf', 'gene_name']].drop_duplicates().reset_index(drop=True)
    
    
    return tf_gene_network

def build_tf_gene_network_mem_efficient(tf_peak_network, peak_gene_network, chunk_size=500000):
    """
    通过合并TF-Peak和Peak-Gene网络来推断TF-Gene调控关系。
    此版本经过内存优化，使用分块处理来避免内存溢出。

    Args:
        tf_peak_network (pd.DataFrame): TF-Peak连接。
        peak_gene_network (pd.DataFrame): Peak-Gene连接。
        chunk_size (int): 每次处理的行数。

    Returns:
        pd.DataFrame: 一个表示TF-Gene连接的DataFrame。
    """
    print("\n--- 3. 构建 TF-Gene 网络 (内存优化版) ---")

    # 为了快速查找，将peak_gene_network的'peak_id'设置为索引
    print("正在为Peak-Gene网络创建索引以便快速查找...")
    peak_gene_indexed = peak_gene_network.set_index('peak_id')
    
    # 存储每个分块处理结果的列表
    all_merged_chunks = []
    
    print(f"开始分块处理，每块大小为 {chunk_size} 行...")
    
    # 将tf_peak_network按chunk_size进行切分并遍历
    for i in tqdm(range(0, len(tf_peak_network), chunk_size), desc="Processing Chunks"):
        # 获取当前的数据块
        chunk = tf_peak_network.iloc[i:i + chunk_size]
        
        # 使用join方法进行合并。这比merge更高效，因为它利用了索引。
        # 'inner' join确保只保留在两个数据集中都存在的peak_id
        merged_chunk = chunk.join(peak_gene_indexed, on='peak_id', how='inner')
        
        # 只需要'tf'和'gene_name'这两列
        if not merged_chunk.empty:
            all_merged_chunks.append(merged_chunk[['tf', 'gene_name']])
            
    if not all_merged_chunks:
        print("警告：在TF-Peak和Peak-Gene网络之间没有找到共同的peaks。")
        return pd.DataFrame(columns=['tf', 'gene_name'])

    # --- 合并所有分块的结果 ---
    print("正在合并所有分块的处理结果...")
    tf_gene_network = pd.concat(all_merged_chunks, ignore_index=True)
    
    # 对最终结果进行去重
    print("正在对最终网络进行去重...")
    tf_gene_network.drop_duplicates(inplace=True)
    tf_gene_network.reset_index(drop=True, inplace=True)
    
    print(f"成功构建TF-Gene网络，共找到 {len(tf_gene_network)} 条连接。")
    print(tf_gene_network.head())
    
    return tf_gene_network


def new_build_tf_gene_network_mem_efficient(tf_peak_network, peak_gene_network, chunk_size=500000):
    """
    通过合并TF-Peak和Peak-Gene网络来推断TF-Gene调控关系。
    此版本经过内存优化，使用分块处理来避免内存溢出。

    Args:
        tf_peak_network (pd.DataFrame): TF-Peak连接。
        peak_gene_network (pd.DataFrame): Peak-Gene连接。
        chunk_size (int): 每次处理的行数。

    Returns:
        pd.DataFrame: 一个表示TF-Gene连接的DataFrame。
    """
    print("\n--- 3. 构建 TF-Gene 网络 (内存优化版) ---")

    # 为了快速查找，将peak_gene_network的'peak_id'设置为索引
    print("正在为Peak-Gene网络创建索引以便快速查找...")
    peak_gene_indexed = peak_gene_network.set_index('peak_id')
    
    # 存储每个分块处理结果的列表
    all_merged_chunks = []
    
    print(f"开始分块处理，每块大小为 {chunk_size} 行...")
    
    # 将tf_peak_network按chunk_size进行切分并遍历
    for i in tqdm(range(0, len(tf_peak_network), chunk_size), desc="Processing Chunks"):
        # 获取当前的数据块
        chunk = tf_peak_network.iloc[i:i + chunk_size]
        
        # 使用join方法进行合并。这比merge更高效，因为它利用了索引。
        # 'inner' join确保只保留在两个数据集中都存在的peak_id
        merged_chunk = chunk.join(peak_gene_indexed, on='peak_id', how='inner')
        
        # 只需要'tf'和'gene_name'这两列
        if not merged_chunk.empty:
            all_merged_chunks.append(merged_chunk[['tf', 'gene_name', 'peak_id']])
            
    if not all_merged_chunks:
        print("警告：在TF-Peak和Peak-Gene网络之间没有找到共同的peaks。")
        return pd.DataFrame(columns=['tf', 'gene_name','peak_count'])

    # --- 合并所有分块的结果 ---
    print("正在合并所有分块的处理结果...")
    tf_gene_network = pd.concat(all_merged_chunks, ignore_index=True)

    tf_gene_counts = tf_gene_network.groupby(['tf', 'gene_name'])['peak_id'].nunique().reset_index()
    tf_gene_counts.rename(columns={'peak_id': 'peak_count'}, inplace=True)
    
    # 对最终结果进行排序（按peak_count降序）
    print("正在对最终网络进行去重...")
    tf_gene_counts.drop_duplicates(inplace=True)
    tf_gene_counts.sort_values('peak_count', ascending=False, inplace=True)
    tf_gene_counts.reset_index(drop=True, inplace=True)
    
    print(f"成功构建TF-Gene网络，共找到 {len(tf_gene_counts)} 条连接。")
    
    return tf_gene_counts

# ==============================================================================
# 主程序运行部分
# ==============================================================================
if __name__ == '__main__':
    # 假设以下变量已经从您之前的代码中加载好了：
    # rna_adata, atac_adata, final_df, tf_motif, full_hic_data_df
    
    # --- 1. 构建 TF-Peak 网络 ---
    # 确保tf_motif DataFrame有'chrom', 'start', 'end', 'tf'这几列
    # 假设tf_motif的列是 ['chrom', 'start', 'end', 'tf_name']
    # tf_motif.rename(columns={'tf_name': 'tf'}, inplace=True) # 如有需要，重命名列
    tf_peak_grn = build_tf_peak_network(tf_motif, atac_adata)
    
    # --- 2. 构建 Peak-Gene 网络 ---
    # 确保 full_hic_data_df 有 'chr1', 'x1', 'x2', 'chr2', 'y1', 'y2'
    # 确保 final_df 有 'gene_name', 'seqname', 'start', 'end', 'strand'
    peak_gene_grn = build_peak_gene_network(full_hic_data_df, atac_adata, final_df)
    
    # --- 3. 构建 TF-Gene 网络 ---
    tf_gene_grn = build_tf_gene_network(tf_peak_grn, peak_gene_grn)
    
    print("\n ground-truth 基因调控网络构建完成！")