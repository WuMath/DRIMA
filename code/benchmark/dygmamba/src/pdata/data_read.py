import ast
import os
import io
import tarfile
import gzip
import collections
import re
import pyfaidx
import time
import itertools
import glob

import pandas as pd 
import numpy as np

import anndata
import anndata as ad

from tqdm import tqdm
from scipy.sparse import csr_matrix
import scipy.sparse as sp_sparse
from scipy import sparse
import scipy.io as sio

from typing import Dict, List, NamedTuple
from gtfparse import read_gtf

import pybedtools
from pybedtools import BedTool

import pickle
import scanpy as sc
import muon.atac as ma  # 用于 TF-IDF
import networkx as nx
from matplotlib import rcParams
import sys





def load_scenhancer_gold_standard(ep_file, score_threshold=1.0):
    """
    加载 scEnhancer E-P 对作为 region-gene 金标准
    score_threshold: ABC score 阈值，1.0 对应约 top 75%
    """
    import pandas as pd
    
    records = []
    with open(ep_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            
            score = float(parts[1])
            if score < score_threshold: continue
            
            key = parts[0]
            region_str, gene_str = key.split('_', 1)
            gene_fields = gene_str.split('$')
            
            # 解析 region
            chrom, coords = region_str.split(':')
            start, end = coords.split('-')
            
            records.append({
                'chrom': chrom,
                'region_start': int(start),
                'region_end': int(end),
                'gene_name': gene_fields[1] if len(gene_fields) > 1 else '',
                'ensembl_id': gene_fields[0],
                'tss': int(gene_fields[3]) if len(gene_fields) > 3 else 0,
                'abc_score': score
            })
    
    df = pd.DataFrame(records)
    print(f"加载 {len(df)} 个 E-P 对（score ≥ {score_threshold}）")
    print(f"  唯一 region: {df['chrom'].astype(str)+'_'+df['region_start'].astype(str)}.nunique()")
    return df






def read_cell_type_chipseq(cell_type, chip_seq_dir, output_path):

    output_file = output_path + "combined_chip_seq.parquet"


    NARROW_PEAK_COLS = [
            'chrom', 'start', 'end',        # 核心坐标 (1-3)
            'name', 'score', 'strand',      # 基本信息 (4-6)
            'signalValue', 'pValue', 'qValue', 'peak'  # 统计值 (7-10)
        ]

    file_list = []
    for root, dirs, files in os.walk(chip_seq_dir):
        for file in files:
            if file.endswith(".bed.gz"):
                file_list.append(os.path.join(root, file))


    all_data = []

    for file_path in tqdm(file_list):

        file_name = os.path.basename(file_path)

        # 2. 【关键修改】提取纯 TF 名称
        # 原始文件名示例: "CTCF_ENCFF123ABC.bed.gz"
        base_name = file_name.replace('.bed.gz', '')

        # 逻辑：以 "_ENCFF" 为界进行分割，取前面部分
        # 这样即使 TF 名字里有下划线 (如 Pol_II)，也不会切错
        if "_ENCFF" in base_name:
            tf_name = base_name.split('_ENCFF')[0]
        else:
            # 兼容性处理：如果文件名里没有 ID，就直接用原名
            tf_name = base_name
            
        df = pd.read_csv(file_path, sep='\t', header=None, compression='gzip',
                            names=NARROW_PEAK_COLS,  # 指定所有列名
                            usecols=list(range(10)),
                            on_bad_lines='skip')

        # 4. 添加清洗后的列
        df['tf_name'] = tf_name
        df['cell_type'] = cell_type
        
        if "_ENCFF" in base_name:
            file_id = "ENCFF" + base_name.split('_ENCFF')[1]
            df['file_id'] = file_id

        all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)

        # 保存
        final_df.to_parquet(output_file, index=False)
        
        final_df.drop_duplicates(subset=['chrom','start','end','tf_name'], inplace= True)
        
        print(f"已保存为 {output_file}")
        
        return final_df
    else:
        return None
    






#################################################################
#****************************************************************
#*******  Read IDR ChIP-seq Data   *************
#****************************************************************
#################################################################

def read_IDR_peaks_TFs(root_dir, output_file):
    all_data = []
    file_list = []

    # 扫描文件
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".bed.gz"):
                file_list.append(os.path.join(root, file))

    print(f"找到 {len(file_list)} 个文件，使用【自适应模式】读取...")

    for file_path in tqdm(file_list):
        try:
            # 1. 准备元数据
            cell_type = os.path.basename(os.path.dirname(file_path))
            file_name = os.path.basename(file_path).replace('.bed.gz', '')

            if "_ENCFF" in file_name:
                tf_name = file_name.split('_ENCFF')[0]
                file_id = "ENCFF" + file_name.split('_ENCFF')[1]
            else:
                tf_name = file_name
                file_id = "Unknown"

            # 2. 读取文件 (核心修改部分)
            # comment='t': 跳过以 't' 开头的行 (通常是 track)
            # header=None: 不把第一行当表头
            # sep='\t': 强制以 Tab 分隔
            df = pd.read_csv(
                file_path,
                sep='\t',
                header=None,
                compression='gzip',
                comment='t',  # 跳过 track 行
                skip_blank_lines=True
            )

            # 再次清洗：有时 comment 参数不够用，手动过滤干扰行
            # 确保第一列是 'chr' 开头的数据
            if not str(df.iloc[0, 0]).startswith('chr'):
                # 如果第一行不是 chr 开头，可能还有其他注释，尝试过滤
                df = df[df[0].astype(str).str.startswith('chr')].copy()

            # 3. 动态重命名列
            # 无论文件有几列，前三列永远是坐标
            num_cols = df.shape[1]
            new_columns = {}
            if num_cols >= 1: new_columns[0] = 'chrom'
            if num_cols >= 2: new_columns[1] = 'start'
            if num_cols >= 3: new_columns[2] = 'end'

            # 如果是标准的 narrowPeak (10列)，尝试补全其他常用名
            if num_cols >= 10:
                new_columns.update({
                    3: 'name', 4: 'score', 5: 'strand',
                    6: 'signal_value', 7: 'p_value', 8: 'q_value', 9: 'peak'
                })

            df.rename(columns=new_columns, inplace=True)

            # 4. 添加标签
            df['tf_name'] = tf_name
            df['cell_type'] = cell_type
            df['source_file_id'] = file_id

            # 简单校验：确保 start 是数字
            # 这一步能剔除掉绝大多数读取错误的行
            df = df[pd.to_numeric(df['start'], errors='coerce').notnull()]

            all_data.append(df)

        except Exception as e:
            # 打印具体是哪个文件报错，方便排查
            print(f"\n[跳过] 读取失败: {file_path}")
            print(f"原因: {e}")

    # 合并
    if all_data:
        print("\n正在合并数据...")
        final_df = pd.concat(all_data, ignore_index=True)

        # 转换类型省内存
        for col in ['chrom', 'cell_type', 'tf_name', 'source_file_id']:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(str).astype('category')

        print("合并完成。前5行预览：")
        print(final_df.head())

        final_df.to_parquet(output_file, index=False)
        print(f"保存完毕: {output_file}")

        return final_df
    else:
        print("没有读取到有效数据。")





#################################################################
#****************************************************************
#*******  Read Unibind Data   *************
#****************************************************************
#################################################################
def read_unibind_file(tar_file_path):

    print("********** 开始读取数据...")        
    # UniBind BED 文件的标准列名
    bed_columns = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'signal', 'p_val', 'q_val', 'peak_center']

    all_tf_regions = []
    tf_stats_list = []    
    print(f"正在读取压缩包: {tar_file_path} ...")

    try:
        with tarfile.open(tar_file_path, "r") as tar:
            
            # 获取包内所有成员列表
            for member in tar.getmembers():

                # 1. 只处理 BED 文件
                if member.isfile() and member.name.endswith('.bed'):

                    # 2. 从文件名中提取 TF 名称
                    # UniBind 文件名通常格式: DATASET_ID.CELL_LINE.TF_NAME.MOTIF_ID.bed
                    # 例如: ENCSR000AKB.GM12878.CTCF.MA0139.1.bed
                    filename = os.path.basename(member.name)
                    parts = filename.split('.')

                    # 假设 TF 名称在第 3 个位置 (索引 2)，根据实际情况调整
                    # 这里的逻辑是寻找全大写的单词，或者依赖 UniBind 的命名规范
                    # 如果你有 metadata TSV 文件，最好配合那个用，这里演示纯文件名提取
                    if len(parts) >= 3:
                        tf_name = parts[2]
                    else:
                        tf_name = "Unknown"

                    # 3. 读取 BED 文件内容
                    f = tar.extractfile(member)
                    if f:
                        # 使用 io.TextIOWrapper 将字节流转为文本流
                        content = io.TextIOWrapper(f, encoding='utf-8')

                        # 读取数据
                        df = pd.read_csv(content, sep='\t', header=None, names=bed_columns)

                        current_count = len(df)
                        # 提取元数据 (增加容错处理)
                        dataset_id = parts[0] if len(parts) > 0 else "Unknown"
                        cell_line = parts[1] if len(parts) > 1 else "Unknown"
                        tf_name = parts[2] if len(parts) > 2 else "Unknown"
                        tf_stats_list.append({
                            'TF': tf_name,
                            'CellLine': cell_line,
                            'DatasetID': dataset_id,
                            'FileName': filename,
                            'PeakCount': current_count
                        })
                        
                        df = df[['chrom', 'start', 'end']]
                        df['TF'] = tf_name
                        df['CellLine'] = cell_line
                        df['SourceID'] = parts[0]

                        all_tf_regions.append(df)

        # 4. 合并所有数据
        if all_tf_regions:
            unibind_df = pd.concat(all_tf_regions, ignore_index=True)
            print("处理完成！")
            print(f"共提取了 {unibind_df['TF'].nunique()} 个 TF 的数据。")
            print(f"总 Region 数量: {len(unibind_df)}")

            # 可选：保存为 CSV 备用
            # unibind_df.to_csv("GM12878_UniBind_Regions.csv", index=False)
        else:
            print("警告：未在压缩包中找到 .bed 文件。")
            
        if tf_stats_list:
            summary_df = pd.DataFrame(tf_stats_list)
            
            # 逻辑：对于同一个 CellLine 下的同一个 TF，可能来自多个 Dataset
            # 我们取 Peak 数最多的那个作为该 TF 在该细胞系下的代表 (Max)
            # 或者你也可以选择 sum() 计算总和
            
            ranked_df = summary_df.groupby(['CellLine', 'TF'])['PeakCount'].max().reset_index()
            
            # 按 PeakCount 降序排列
            ranked_df = ranked_df.sort_values(['CellLine', 'PeakCount'], ascending=[True, False])
            
            # 添加组内排名
            ranked_df['Rank'] = ranked_df.groupby('CellLine')['PeakCount'].rank(method='first', ascending=False)
            
            # 保存统计表
            # ranked_df.to_csv("UniBind_TF_Rank_Summary.csv", index=False)
            # print(f"统计排名已保存至: UniBind_TF_Rank_Summary.csv")

    except FileNotFoundError:
        print("错误：找不到指定的 tar 文件，请检查路径。")
        
    return unibind_df, ranked_df




#################################################################
#****************************************************************
#*******  Read ATAC data   *************
#****************************************************************
#################################################################

def read_atac_featurecounts(counts_file, output_file):
    """
    读取 featureCounts 格式的 ATAC-seq 计数矩阵并转换为 AnnData
    """
    print("--- 步骤 1: 加载计数矩阵 ---")
    
    # 1. 读取数据
    # featureCounts 文件通常包含以 '#' 开头的注释行，需要跳过 (comment='#')
    # 使用 sep='\t' 指定制表符分隔
    counts_data = pd.read_csv(counts_file, sep="\t", comment='#')
    
    print(f"成功读取文件，原始形状: {counts_data.shape}")

    print("--- 步骤 2: 处理元数据与细胞名 ---")

    # 2. 提取 Peak 信息 (前 6 列是标准元数据)
    # Geneid, Chr, Start, End, Strand, Length
    peak_info = counts_data.iloc[:, :6].copy()
    
    # 构建标准的 chr-start-end 格式作为索引
    # 注意: featureCounts 的 Start/End 列通常是整数，转为字符串拼接
    peak_names = peak_info['Chr'] + '-' + peak_info['Start'].astype(str) + '-' + peak_info['End'].astype(str)
    peak_info.index = peak_names
    
    # 3. 提取计数矩阵 (从第 7 列开始是样品数据)
    counts_matrix = counts_data.iloc[:, 6:]
    counts_matrix.index = peak_names  # 行是 Peak
    
    # 4. 清洗细胞名称 (列名)
    # 原始列名示例: "atac_bam/atac_cell_100.sorted.bam"
    # 目标: "cell_100" (或者保留 atac_cell_100，视你习惯而定)
    
    new_columns = []
    for col in counts_matrix.columns:
        # 获取文件名: atac_bam/atac_cell_100.sorted.bam -> atac_cell_100.sorted.bam
        filename = os.path.basename(col)
        # 去掉后缀: -> atac_cell_100
        cell_id = filename.split('.')[0]
        # (可选) 如果你想要去掉 'atac_' 前缀变成 'cell_100'
        if cell_id.startswith('atac_'):
            cell_id = cell_id.replace('atac_', '')
        new_columns.append(cell_id)
        
    counts_matrix.columns = new_columns
    
    print(f"检测到 {len(new_columns)} 个细胞，{len(peak_names)} 个 Peaks")
    print(f"细胞名示例: {new_columns[:5]}")

    print("--- 步骤 3: 创建 AnnData 对象 ---")

    # 5. 创建 AnnData
    # Scanpy/AnnData 要求: 行(Obs)是细胞，列(Var)是特征(Peaks)
    # 所以需要转置 (.T)
    adata = ad.AnnData(counts_matrix.T)
    
    # 6. 填充 Peak 元数据
    # 将 Chr, Start, End 等信息放进 adata.var
    adata.var = peak_info
    # 同时也把生成的 chr-start-end 设为 var_names
    adata.var_names = peak_names
    
    # 7. 稀疏化矩阵 (对于 ATAC 数据非常重要，能极大减少内存占用)
    adata.X = sparse.csr_matrix(adata.X)
    
    # 8. 保存

    adata.write_h5ad(output_file)
    
    print(f"✅ 处理完成！AnnData 维度: {adata.shape}")
    print(f"文件已保存至: {output_file}")
    
    return adata


    
def read_bedtools_multicov_detailed(file_path):

    column_names = [
        'chrom', 'start', 'end', 'name', 'score', 'strand',
        'signalValue', 'pValue', 'qValue', 'peak', 'count'
    ]
    
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=column_names
        )
        return df
    
    except FileNotFoundError:
        print(f"Error: can't find file '{file_path}'")
        return None
    except Exception as e:
        print(f"Error: read or analysis {e}")   
        return None





def anndata_to_bed(adata, output_file="peaks.bed"):
    """
    将 adata.var_names 中的坐标转换为标准的 BED 文件
    """
    # 1. 获取峰的名称列表 (通常是 index)
    # 如果您的坐标在某一列里，请改为 peak_strings = adata.var['column_name']
    peak_strings = adata.var_names.to_series()
    
    print(f"检测到 {len(peak_strings)} 个 Peaks。")
    print(f"示例格式: {peak_strings.iloc[0]}")

    # 2. 解析坐标 (使用正则表达式拆分)
    # 这个正则非常强大，可以同时兼容 "chr1:100-200" 和 "chr1-100-200"
    # 逻辑：
    #   (chr[\w\.]+) : 捕获染色体 (兼容 chr1, chrX, chr1_gl...)
    #   [-:]         : 分隔符是 - 或 :
    #   (\d+)        : 起始坐标 (数字)
    #   [-:]         : 分隔符
    #   (\d+)        : 终止坐标 (数字)
    pattern = r'(?P<chrom>chr[\w\.]+)[-:_](?P<start>\d+)[-:_](?P<end>\d+)'
    
    bed_df = peak_strings.str.extract(pattern)
    
    # 3. 检查是否有解析失败的行
    if bed_df.isnull().any().any():
        print("警告：部分坐标解析失败，请检查 var_names 格式！")
        print(bed_df[bed_df.isnull().any(axis=1)].head())
        # 删除坏行 (可选)
        bed_df = bed_df.dropna()

    # 4. 添加第 4 列：Peak Name (非常重要！)
    # 我们把原始的字符串 (如 chr1-100-200) 作为 ID，方便后续 map 回去
    bed_df['name'] = peak_strings.values
    
    # 5. 确保是标准的 4 列格式: chrom, start, end, name
    # 也可以加上 score, strand 等 (设为 . 或 0)
    bed_df = bed_df[['chrom', 'start', 'end', 'name']]
    
    # 6. 保存为 BED 文件
    # header=False: BED 文件不能有表头
    # index=False: 不需要行号
    # sep='\t': 必须是制表符分隔
    bed_df.to_csv(output_file, sep='\t', header=False, index=False)
    
    print(f"成功保存到: {output_file}")
    return bed_df
#################################################################
#****************************************************************
#*******  Read RNA data   *************
#****************************************************************
#################################################################

def read_scrna_counts(file_path):

    columns_name = ['Geneid', 'Chr', 'Start', 'End', 'Strand', 'Length', 'count']
    try:

        counts_df = pd.read_csv(
            file_path,
            sep='\t',
            header=0,
            comment='#',
            names = columns_name
        )

        filter_name = ['Geneid', 'count']

        counts_df = counts_df[filter_name].copy()
    
        return counts_df

    except FileNotFoundError:
        print(f"Error: can't find file '{file_path}'")
        return None
    except Exception as e:
        print(f"Error: read or analysis {e}")
        return None





def create_anndata_from_folder(data_directory, type):

    all_counts_list = []
    cell_names = []

    count_files = [f for f in os.listdir(data_directory) if f.endswith(('.txt', '.tsv'))]
    
    if not count_files:
        print(f"Error: There are no count file in '{data_directory}'!")
        return None

    for filename in tqdm(count_files, desc="Processing files"):
        file_path = os.path.join(data_directory, filename)
        
        if type =='gene':
            single_cell_df = read_scrna_counts(file_path)

            if single_cell_df is None:
                print(f'error read: {file_path}')
                continue 
            
            cell_name = filename.replace('_counts.txt', '').replace('rna_', '')
            cell_names.append(cell_name)
            
            single_cell_df.rename(columns={'count': cell_name}, inplace=True)
            
            single_cell_df.set_index('Geneid', inplace=True)

        elif type == 'atac':

            single_cell_df = read_bedtools_multicov_detailed(file_path)
            
            if single_cell_df is None:
                print(f'error read: {file_path}')
                continue 

            single_cell_df['peak_id'] = (
                single_cell_df['chrom'] + ':' + 
                single_cell_df['start'].astype(str) + '-' + 
                single_cell_df['end'].astype(str)
            )

            cell_name = filename.replace('_atac_counts_bedtools.txt', '').replace('atac_','')
            cell_names.append(cell_name)
            
            single_cell_df = single_cell_df[['peak_id', 'count']].rename(
                columns={'count': cell_name}
            ).set_index('peak_id')

        else:
            print(f'Unknow type file: {type}')
            return

        
        all_counts_list.append(single_cell_df)

    combined_matrix = pd.concat(all_counts_list, axis=1)

    combined_matrix.fillna(0, inplace=True)

    combined_matrix = combined_matrix.astype(int)

    # --- create AnnData object ---
    print("Create AnnData Object...")

    adata = ad.AnnData(combined_matrix.T)
    
    adata.X = csr_matrix(adata.X)
    
    print("Successfully Create AnnData!")
    
    return adata


def read_rna_from_txt(rna_counts_directory, output_file):
    
    # get gene counts

    print("********************---read gene & atac counts---*****************************")

    adata_rna = create_anndata_from_folder(rna_counts_directory, type= 'gene')

    adata_rna.write_h5ad( output_file )

    print("RNA txt data saved to", output_file)

    return adata_rna

#################################################################
#****************************************************************
#*******  Read Gene annotation data   *************
#****************************************************************
#################################################################

def read_genenotation(gtf_file_path):

    try:

        gtf_df = read_gtf(gtf_file_path)

        gtf_df = gtf_df.to_pandas()

        return gtf_df

    except FileNotFoundError:
        print(f"Error: can't find file '{gtf_file_path}'")
        
        return None
    except Exception as e:
        print(f"Error: read or analysis {e}")

        return None



#################################################################
#****************************************************************
#*******  Read JASPAR data from folder  *************
#****************************************************************
#################################################################

def read_jaspar_bed(jaspar_bed_file):

    COLUMNS = pd.Index(
            [
                "chrom",
                "chromStart",
                "chromEnd",
                "name",
                "score",
                "strand",
                "thickStart",
                "thickEnd",
                "itemRgb",
                "blockCount",
                "blockSizes",
                "blockStarts",
            ]
        )

    jaspar_data = pd.read_csv(jaspar_bed_file, sep='\t', header=None, comment="#")

    jaspar_data.columns = COLUMNS[: jaspar_data.shape[1]]

    return jaspar_data


def get_id_to_symbol_map(meme_path):
    """
    解析 MEME 文件，提取 ID 到 Symbol 的映射关系。
    目标行格式: MOTIF MA0004.1 Arnt
    """
    id_map = {}
    print(f"正在解析 MEME 文件: {meme_path} ...")
    
    with open(meme_path, 'r') as f:
        for line in f:
            if line.startswith("MOTIF"):
                # 拆分行: ['MOTIF', 'MA0004.1', 'Arnt']
                parts = line.strip().split()
                
                if len(parts) >= 3:
                    motif_id = parts[1]   # MA0004.1
                    symbol = parts[2]     # Arnt
                    id_map[motif_id] = symbol
                elif len(parts) == 2:
                    # 极少数情况可能没有 Symbol，只有 ID
                    motif_id = parts[1]
                    id_map[motif_id] = motif_id # 用 ID 代替
                    
    print(f"解析完成，共提取 {len(id_map)} 个 TF 的映射。")
    return id_map



def read_fimo_jaspar(jaspar_fimo_file, meme_file, output_file=None):

    fimo_df = pd.read_csv(jaspar_fimo_file, sep="\t", comment="#", header=None)

    num_cols = fimo_df.shape[1]
    print(f"检测到 {num_cols} 列")

    # 3. 根据列数动态分配列名
    if num_cols == 9:
        fimo_df.columns = ["motif_id", "sequence_name", "start", 
                      "end", "strand", "score", "p-value", "q-value", "matched_sequence"]
    else:
        # 其他情况，生成通用列名
        fimo_df.columns = ["motif_id", "motif_alt_id","sequence_name", "start", 
                      "end", "strand", "score", "p-value", "q-value", "matched_sequence"]

    mapping_dict = get_id_to_symbol_map(meme_file)
    
    # FIMO 结果的第一列通常叫 'motif_id'
    if 'motif_id' in fimo_df.columns:
        fimo_df['TF_Symbol'] = fimo_df['motif_id'].map(mapping_dict)
        
        # 填充那些没匹配上的（防止变成 NaN）
        fimo_df['TF_Symbol'] = fimo_df['TF_Symbol'].fillna(fimo_df['motif_id'])
        
        # 5. 调整列顺序，把 Symbol 放在最前面方便看
        cols = ['motif_id', 'TF_Symbol'] + [c for c in fimo_df.columns if c not in ['motif_id', 'TF_Symbol']]
        fimo_df = fimo_df[cols]
        
        print("转换成功！前 5 行预览：")
        print(fimo_df.head())
        
        # 6. 保存
        if output_file:
            fimo_df.to_csv(output_file, index=False)
            print(f"保存至: {output_file}")
            
    else:
        print("错误：在 FIMO 文件中找不到 'motif_id' 列，请检查列名。")

    return fimo_df


#################################################################
#****************************************************************
#*******  Read JASPAR data from folder  *************
#****************************************************************
#################################################################

def scan_sequence_numpy(seq_indices, pwm):
    """
    使用 numpy stride_tricks 在数字序列上滑动 PWM。
    """
    motif_len = pwm.shape[1] # L
    seq_len = len(seq_indices)
    
    if seq_len < motif_len:
        return -np.inf # 序列比 motif 短

    pwm_T = pwm.T

    windows = np.lib.stride_tricks.as_strided(
        seq_indices,
        shape=(seq_len - motif_len + 1, motif_len),
        strides=(seq_indices.strides[0], seq_indices.strides[0])
    )

    all_window_scores = pwm_T[np.arange(motif_len), windows]

    scores = all_window_scores.sum(axis=1)

    return np.max(scores)



def load_jaspar_folder_to_pfm(folder_path):

    jaspar_files = glob.glob(os.path.join(folder_path, "*.jaspar"))

    if not jaspar_files:

        jaspar_files = glob.glob(os.path.join(folder_path, "*.txt"))
        
    print(f"Find {len(jaspar_files)} motif file in {folder_path}")

    if not jaspar_files:
        raise FileNotFoundError(f"There are not found .jaspar or .txt file in {folder_path}")

    motif_dict = {}
    motif_names_list = []
    
    for file_path in jaspar_files:
        with open(file_path, 'r') as f:
            header = ""
            matrix_lines = []
            try:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        # format: >MA0002.3  Runx1 -> MA0002.3_Runx1
                        parts = re.split(r'\s+', line.strip()[1:], maxsplit=1)
                        if len(parts) == 2:
                            header = f"{parts[1]}"
                        else:
                            header = parts[0]
                    elif re.match(r'^[ACGT]\s*\[', line):
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            matrix_lines.append(np.fromiter(numbers, dtype=float))
                
                if header and len(matrix_lines) == 4:
        
                    motif_dict[header] = np.stack(matrix_lines)
                    motif_names_list.append(header)
                else:
                    print(f"警告: 跳过格式不正确的文件 {os.path.basename(file_path)}")
            except Exception as e:
                print(f"错误: 解析文件 {os.path.basename(file_path)} 失败: {e}")
                
    print(f"成功加载并解析了 {len(motif_dict)} 个 motifs。")
    return motif_dict, motif_names_list

# --- 辅助函数：PFM -> PWM 转换 ---

def convert_pfms_to_pwms(pfm_dict, pseudocount=0.5):

    background = np.array([0.25, 0.25, 0.25, 0.25])
    pwm_dict = {}
    
    for name, pfm in pfm_dict.items():
        # 1. 添加伪计数
        pfm_with_pseudo = pfm + pseudocount
        # 2. 计算频率 (P_i,b)
        probabilities = pfm_with_pseudo / pfm_with_pseudo.sum(axis=0, keepdims=True)
        # 3. 计算 Log-Odds (log2(P_i,b / B_b))
        pwm_log_odds = np.log2(probabilities / background[:, np.newaxis])
        
        # 4. 创建 (5, L) 矩阵，第 5 行 (索引 4) 全为 0，用于 N
        pwm_final = np.zeros((5, pwm_log_odds.shape[1]))
        pwm_final[0:4, :] = pwm_log_odds
        
        pwm_dict[name] = pwm_final
        
    return pwm_dict

def read_jaspar_folder(jaspar_folder_path, genome_fasta_file, adata_atac):
    
    print("**********"*10)
    print("--- 步骤 1: 加载并解析 JASPAR motifs ---")

    start_time = time.time()

    pfm_dict, motif_names = load_jaspar_folder_to_pfm(jaspar_folder_path)

    print(f"Motif 加载耗时: {time.time() - start_time:.2f} 秒")

    print("\n--- 步骤 2: 将 PFM 转换为 PWM (Log-Odds) ---")

    pwm_dict = convert_pfms_to_pwms(pfm_dict)

    print("\n--- 步骤 3: 加载并索引参考基因组 (pyfaidx) ---")

    start_time = time.time()

    try:
        genome = pyfaidx.Fasta(genome_fasta_file)
        print(f"基因组索引完成。耗时: {time.time() - start_time:.2f} 秒")

    except pyfaidx.FastaIndexingError as e:
        print(f"!!! 基因组索引失败: {e}")
        print("请确保您有该文件的写入权限 (以创建 .fai 索引)，或者文件未损坏。")

    peaks_list = adata_atac.var_names.to_list()

    # --- 步骤 4: 执行扫描 ---
    print("\n--- 步骤 4: 开始扫描 Peaks ... ---")
    print(f"将扫描 {len(peaks_list)} 个 peaks (obs) x {len(motif_names)} 个 TFs (var)")

    start_time = time.time()

    n_peaks = len(peaks_list)
    n_motifs = len(motif_names)
    results_matrix = np.zeros((n_peaks, n_motifs), dtype=np.float32)

    for i, peak_str in enumerate(peaks_list):
        # 1. 解析 peak 坐标
        try:
            chrom, start, end = re.split(r'[:-]', peak_str)
            start, end = int(start), int(end)
        except Exception as e:
            print(f"警告: 跳过格式错误的 peak: {peak_str} ({e})")
            results_matrix[i, :] = -np.inf # 标记为无效
            continue

        # 2. 从基因组提取序列 (pyfaidx)
        try:
            seq_obj = genome[chrom][start:end]
            fwd_seq = str(seq_obj).upper()
            rev_seq = str(seq_obj.reverse.complement).upper()
        except KeyError:
            print(f"警告: 在 FASTA 中找不到染色体 '{chrom}'。跳过 peak {peak_str}")
            results_matrix[i, :] = -np.inf
            continue
        except Exception as e:
            print(f"警告: 提取序列 {peak_str} 失败: {e}。跳过。")
            results_matrix[i, :] = -np.inf
            continue
            
        # 3. 将序列转换为数字数组

        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        seq_to_indices = lambda seq: np.array([base_map.get(base, 4) for base in seq], dtype=np.int8)
        fwd_arr = seq_to_indices(fwd_seq)
        rev_arr = seq_to_indices(rev_seq)
        
        # 4. 遍历所有 motif
        for j, motif_name in enumerate(motif_names):
            pwm = pwm_dict[motif_name]
            
            # 5. 扫描正向和反向链
            score_fwd = scan_sequence_numpy(fwd_arr, pwm)
            score_rev = scan_sequence_numpy(rev_arr, pwm)
            
            # 6. 存储最高分
            results_matrix[i, j] = max(score_fwd, score_rev)

        if (i + 1) % 100 == 0: # 每 100 个 peak 打印一次进度
            elapsed = time.time() - start_time
            print(f"  ... 已处理 {i+1} / {n_peaks} 个 peaks ({elapsed:.2f} 秒)")

    print(f"--- 扫描完成！总耗时: {time.time() - start_time:.2f} 秒 ---")

        # --- 步骤 5: 构建并保存 AnnData 对象 ---
    print(f"\n--- 步骤 5: 正在构建 Region-TF AnnData 对象 ---")

    # 创建 AnnData 对象
    adata_region_tf = ad.AnnData(
        X=results_matrix,
        obs=pd.DataFrame(index=peaks_list),
        var=pd.DataFrame(index=motif_names)
    )

    # 添加描述信息
    adata_region_tf.uns['description'] = (
        "Region-TF matrix created from custom scan. "
        "obs_names are genomic regions (peaks). "
        "var_names are TF motifs from JASPAR. "
        "X contains the max log-odds score (PWM) for each motif in each region."
    )

    return adata_region_tf


#################################################################
#****************************************************************
#*******  Read ChIP-seq peak data from folder  *************
#****************************************************************
#################################################################

# 定义命名元组来表示完整的peak信息
class PeakInfo(NamedTuple):
    chromosome: str
    start: int
    end: int
    name: str
    score: float
    strand: str
    signal_value: float
    p_value: float
    q_value: float
    peak: int


def iter_read_single_chip(folder_path: str) -> dict:
    """
    Args:
        folder_path: the file path containing all .bed.gz file

    Returns:
        一A dict with key as TF(str), value as peaks list
        each peak is a tuple (chromosome, start, end)。
    """
    tf_peaks = collections.defaultdict(list)

    print(f"Start scan the folder: {folder_path}")

    if not os.path.isdir(folder_path):
        print(f"Error: folder '{folder_path}' don't exist")
        return {}

    for filename in os.listdir(folder_path):

        if filename.endswith(".bed.gz"):

            tf_name = filename.removesuffix('.bed.gz')

            file_path = os.path.join(folder_path, filename)
            print(f"  Now deal file: {filename}, identify TF: {tf_name}")

            try:

                with gzip.open(file_path, 'rt') as f:
                    for line in f:

                        if line.startswith(('#', 'track', 'browser')):
                            continue

                        parts = line.strip().split('\t')

                        if len(parts) >= 3:
                            chromosome = parts[0]
                            start = int(parts[1])
                            end = int(parts[2])
                            
                            # 为可选列提供默认值
                            name = parts[3] if len(parts) > 3 else f"peak_0"
                            score = float(parts[4]) if len(parts) > 4 and parts[4] != '.' else 0.0
                            strand = parts[5] if len(parts) > 5 else '.'
                            signal_value = float(parts[6]) if len(parts) > 6 and parts[6] != '.' else 0.0
                            p_value = float(parts[7]) if len(parts) > 7 and parts[7] != '.' else 1.0
                            q_value = float(parts[8]) if len(parts) > 8 and parts[8] != '.' else 1.0
                            peak = int(parts[9]) if len(parts) > 9 and parts[9] != '.' else -1
                            
                            # 创建完整的peak信息对象
                            peak_info = PeakInfo(
                                chromosome=chromosome,
                                start=start,
                                end=end,
                                name=name,
                                score=score,
                                strand=strand,
                                signal_value=signal_value,
                                p_value=p_value,
                                q_value=q_value,
                                peak=peak
                            )

                            tf_peaks[tf_name].append(peak_info)

            except (IOError, ValueError, IndexError) as e:
                print(f"********** deal file {filename} with error: {e}")
            except Exception as e:
                print(f"********** unknow error: {e}")

    print("All file had deal")
    return dict(tf_peaks)


def Chip_dict_to_df(tf_peaks_dict: dict) -> pd.DataFrame:
    """
    Args:
        tf_peaks_dict: the dict get by "parse_tf_peaks_from_folder" function

    Returns:
        A Pandas DataFrame contain TF-peak info
    """
    records = []

    for tf, peaks in tf_peaks_dict.items():
        for peak in peaks:
            records.append({
                'TF': tf,
                'Chromosome': peak.chromosome,
                'Start': peak.start,
                'End': peak.end,
                'Name': peak.name,
                'Score': peak.score,
                'Strand': peak.strand,
                'SignalValue': peak.signal_value,
                'PValue': peak.p_value,
                'QValue': peak.q_value,
                'Peak': peak.peak
            })

    df = pd.DataFrame(records)

    df.columns = ['tf','chrom','start','end','name', 'score', 'strand', 'signal', 'pvalue', 'qvalue','peak']
    
    return df


def read_chipseq(data_folder):
    
    print("--- Start analysis function ---")

    all_peaks_data = iter_read_single_chip(data_folder)

    print("--- Have done ---\n")

    print("--- Result ---")
    if all_peaks_data:
        for tf, peaks in all_peaks_data.items():
            print(f"TF: {tf}")
            print(f"  - find {len(peaks)} peaks。")
            for peak in peaks[:3]:
                print(f"    - Chrom: {peak[0]}, Start: {peak[1]}, End: {peak[2]}")
            if len(peaks) > 3:
                print("    - ...")
            print("-" * 20)

        print("--- Convert dict to DataFrame ---")

        peaks_df = Chip_dict_to_df(all_peaks_data)

        print("Successfully convert! DataFrame info as following:")
        peaks_df.info()

        return peaks_df
    else:
        print("Can't find any TF-peak data")

        return None


###############################################################################
#****************************************************************
#*******  Read Unibind data  *************
#****************************************************************
#################################################################



def get_peak_count_from_tar(tar, file_path):
    """从 tar 包中读取文件并计算行数"""
    try:
        member = tar.getmember(file_path)
        f = tar.extractfile(member)
        # 统计行数 (Peak 数)
        return sum(1 for _ in f)
    except KeyError:
        # 尝试其他路径组合，有时候 tar 包内没有文件夹前缀
        try:
            # 尝试只用文件名
            filename = os.path.basename(file_path)
            member = tar.getmember(filename)
            f = tar.extractfile(member)
            return sum(1 for _ in f)
        except KeyError:
            print(f"Warning: {file_path} not found in tar archive.")
            return 0


def get_peak_count_from_dir(root_dir, file_path):
    """从文件夹中读取文件并计算行数"""
    full_path = os.path.join(root_dir, file_path)
    if not os.path.exists(full_path):
        # 尝试不带文件夹路径
        full_path = os.path.join(root_dir, os.path.basename(file_path))

    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            return sum(1 for _ in f)
    else:
        print(f"Warning: {file_path} not found in directory.")
        return 0


def process_unibind_data(tsv_path, data_source_path):
    """
    处理 UniBind 数据
    :param tsv_path: TSV 文件路径
    :param data_source_path: .tar.gz 文件路径 或 解压后的文件夹路径
    """
    print(f"Loading metadata from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t')

    # 结果列表
    results = []

    # 判断是 tar 包还是文件夹
    is_tar = tarfile.is_tarfile(data_source_path) if os.path.isfile(data_source_path) else False

    tar = None
    if is_tar:
        print(f"Opening tar archive {data_source_path}...")
        tar = tarfile.open(data_source_path, "r:gz")

    print("Processing TF peak counts...")

    # 遍历 TSV 中的每一行
    for idx, row in df.iterrows():
        tf_name = row['tf_name']
        # 提取细胞系名称，这里简单使用 title，或者您可以根据需要清洗 title 列
        # SCENIC+ 通常关注特定细胞系，如 'MCF7', 'HepG2' 等
        cell_line = row['title']
        folder = row['folder']

        # 处理可能存在的多个 BED 文件 (用 | 分隔)
        bed_files = str(row['bed_filename']).split('|')

        for bed_file in bed_files:
            # 构建文件路径，通常 UniBind 结构是 "folder/bed_file"
            # 或者是直接在根目录，视下载包结构而定
            file_path_in_tar = f"{folder}/{bed_file}"

            if is_tar:
                count = get_peak_count_from_tar(tar, file_path_in_tar)
            else:
                count = get_peak_count_from_dir(data_source_path, file_path_in_tar)

            if count > 0:
                results.append({
                    'TF': tf_name,
                    'CellLine': cell_line,
                    'PeakCount': count,
                    'Dataset': bed_file
                })

    if tar:
        tar.close()

    # 创建 DataFrame
    res_df = pd.DataFrame(results)

    if res_df.empty:
        print("No peak counts found. Please check paths.")
        return None

    # --- 核心步骤：聚合与排名 ---
    # 对于每个细胞系中的每个 TF，取其所有数据集中 Peak 数最大的那个值
    # 这符合 SCENIC+ 的 "ranked based on the number of target regions"
    print("Aggregating and ranking...")
    ranked_df = res_df.groupby(['CellLine', 'TF'])['PeakCount'].max().reset_index()

    # 按细胞系和 PeakCount 降序排列
    ranked_df = ranked_df.sort_values(['CellLine', 'PeakCount'], ascending=[True, False])

    # 添加排名列
    ranked_df['Rank'] = ranked_df.groupby('CellLine')['PeakCount'].rank(method='first', ascending=False)

    return ranked_df


def read_unibind_data(tsv_file, data_file):
    """
    读取并处理 UniBind 数据
    :param tsv_path: TSV 文件路径
    :param data_source_path: .tar.gz 文件路径 或 解压后的文件夹路径
    """
    # ================= 使用示例 =================
    # 1. 设置路径
    # tsv_file = 'UniBind_search_02rpkg57.tsv'  # 您上传的 TSV 文件名
    # data_file = 'UniBind_search_0_998agt.tar.gz'  # 您下载的 tar.gz 文件名 (或解压后的文件夹路径)

    # 2. 运行处理
    if os.path.exists(tsv_file) and os.path.exists(data_file):
        ranked_tfs = process_unibind_data(tsv_file, data_file)

    if ranked_tfs is not None:
        # 3. 保存结果
        output_file = 'UniBind_TF_Rankings.csv'
        ranked_tfs.to_csv(output_file, index=False)
        print(f"Done! Rankings saved to {output_file}")

        # 4. 打印 HepG2 的前 10 名示例 (如果有 HepG2 数据)
        print("\nTop 10 TFs in HepG2 (example):")
        
    return ranked_tfs


def read_unibind_tf_peak():
    # ================= 配置 =================
    tar_file_path = 'UniBind_search_0_998agt.tar.gz'  # 你的文件名
    # UniBind BED 文件的标准列名
    bed_columns = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'signal', 'p_val', 'q_val', 'peak_center']

    # ================= 处理流程 =================
    all_tf_regions = []

    print(f"正在读取压缩包: {tar_file_path} ...")

    try:
        with tarfile.open(tar_file_path, "r") as tar:
            # 获取包内所有成员列表
            for member in tar.getmembers():

                # 1. 只处理 BED 文件
                if member.isfile() and member.name.endswith('.bed'):

                    # 2. 从文件名中提取 TF 名称
                    # UniBind 文件名通常格式: DATASET_ID.CELL_LINE.TF_NAME.MOTIF_ID.bed
                    # 例如: ENCSR000AKB.GM12878.CTCF.MA0139.1.bed
                    filename = os.path.basename(member.name)
                    parts = filename.split('.')

                    # 假设 TF 名称在第 3 个位置 (索引 2)，根据实际情况调整
                    # 这里的逻辑是寻找全大写的单词，或者依赖 UniBind 的命名规范
                    # 如果你有 metadata TSV 文件，最好配合那个用，这里演示纯文件名提取
                    if len(parts) >= 3:
                        tf_name = parts[2]
                    else:
                        tf_name = "Unknown"

                    # 3. 读取 BED 文件内容
                    f = tar.extractfile(member)
                    if f:
                        # 使用 io.TextIOWrapper 将字节流转为文本流
                        content = io.TextIOWrapper(f, encoding='utf-8')

                        # 读取数据
                        df = pd.read_csv(content, sep='\t', header=None, names=bed_columns)

                        # 只保留核心坐标列，节省内存
                        df = df[['chrom', 'start', 'end']]

                        # 添加 TF 标签
                        df['TF'] = tf_name

                        # 添加源文件 ID (可选，用于追溯)
                        df['SourceID'] = parts[0]

                        all_tf_regions.append(df)

        # 4. 合并所有数据
        if all_tf_regions:
            unibind_df = pd.concat(all_tf_regions, ignore_index=True)
            print("处理完成！")
            print(f"共提取了 {unibind_df['TF'].nunique()} 个 TF 的数据。")
            print(f"总 Region 数量: {len(unibind_df)}")
            print(unibind_df.head())

            # 可选：保存为 CSV 备用
            # unibind_df.to_csv("GM12878_UniBind_Regions.csv", index=False)
        else:
            print("警告：未在压缩包中找到 .bed 文件。")

    except FileNotFoundError:
        print("错误：找不到指定的 tar 文件，请检查路径。")


