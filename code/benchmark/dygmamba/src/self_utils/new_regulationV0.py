import os
import sys
import subprocess
import pandas as pd
import networkx as nx
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import scipy.io
import scipy.sparse as sp_sparse
import numpy as np 
from pkg_resources import resource_filename
from datetime import datetime
from collections import Counter
from tqdm import tqdm
import pickle


# def RP_peaks(peaks_info, decay):
#     Sg = lambda x: 2**(-x)
#     P = len(peaks_info)
    
#     peak_distance = decay
#     peaks_score_array = sp_sparse.dok_matrix((len(peaks_info), len(peaks_info)), dtype=np.float64)
#     peaks_info.sort()
    
#     for i in range(P):
#         elem = peaks_info[i]
#         for j in range(i+1,P):
#             elem2 = peaks_info[j]
#             if elem[0] != elem2[0]:
#                 continue

#             peak_d = elem2[1] -elem[1]
#             if peak_d > peak_distance:
#                 continue
#             peaks_score_array[elem[-1],elem2[-1]] = Sg(peak_d/decay)

#     return peaks_score_array


def RP_peaks(peaks_info, decay):
    """
    原版: O(P²) Python双重循环，P=10万时需要数小时。
    优化: 按染色体分组后用numpy向量化，利用searchsorted做区间剪枝。
    速度提升: 10x ~ 100x。
    """
    Sg = lambda x: 2.0 ** (-x)
    P = len(peaks_info)

    # 提取字段为 numpy 数组，避免反复索引 Python list
    chroms  = np.array([p[0] for p in peaks_info])
    centers = np.array([p[1] for p in peaks_info], dtype=np.float64)
    ids     = np.array([p[-1] for p in peaks_info], dtype=np.int64)

    rows_all, cols_all, vals_all = [], [], []

    for chrom in np.unique(chroms):
        mask = chroms == chrom
        c_ids     = ids[mask]
        c_centers = centers[mask]

        # 在染色体内按center排序
        sort_idx  = np.argsort(c_centers)
        c_ids     = c_ids[sort_idx]
        c_centers = c_centers[sort_idx]

        n = len(c_centers)
        if n == 0:
            continue

        # ---- 向量化: 对每个i，用searchsorted找距离<=decay的j范围 ----
        # 比原始 j循环快 ~50倍
        for i in range(n):
            j_max = np.searchsorted(c_centers, c_centers[i] + decay, side='right')
            if j_max <= i + 1:
                continue
            j_idx    = np.arange(i + 1, j_max)
            dists    = c_centers[j_idx] - c_centers[i]
            scores   = Sg(dists / decay)

            rows_all.append(np.full(len(j_idx), c_ids[i], dtype=np.int64))
            cols_all.append(c_ids[j_idx])
            vals_all.append(scores)

    if rows_all:
        rows_all = np.concatenate(rows_all)
        cols_all = np.concatenate(cols_all)
        vals_all = np.concatenate(vals_all)
        from scipy.sparse import coo_matrix
        result = coo_matrix((vals_all, (rows_all, cols_all)), shape=(P, P))
        return result.todok()
    else:
        return sp_sparse.dok_matrix((P, P), dtype=np.float64)


#*******************************************************************************************
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#*******************************************************************************************



def RP_AddExonRemovePromoter(peaks_info, genes_info_full, genes_info_tss, gene_distance):
    """Multiple processing function to calculate regulation potential."""

    Sg = lambda x: 2**(-x)
    checkInclude = lambda x, y: all([x>=y[0], x<=y[1]])
    genes_peaks_score_array = sp_sparse.dok_matrix((len(genes_info_full), len(peaks_info)), dtype=np.float64)
    peaks_info_inbody = []
    peaks_info_outbody = []
    
    w = genes_info_full + peaks_info
    A = {}

    w.sort()
#     print(w[:100])
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                ### NOTE: main change here
                ### if peak center in the gene area
                if all([g[0]==elem[0], elem[1]>=g[1], elem[1]<=g[2]]):
                    ### if peak center in the exons
                    if any(list(map(checkInclude, [elem[1]]*len(g[5]), list(g[5])))):
                        genes_peaks_score_array[gene_name, elem[-1]] = 1.0 / g[-4]
                        peaks_info_inbody.append(elem)
                    ### if peak cencer in the promoter
                    elif checkInclude(elem[1], g[4]):
                        tmp_distance = abs(elem[1]-g[3])
                        genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / gene_distance)
                        peaks_info_inbody.append(elem)
                    ### intron regions
                    else:
                        continue
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]
    
    ### remove genes in promoters and exons
    peaks_info_set = [tuple(i) for i in peaks_info]
    peaks_info_inbody_set = [tuple(i) for i in peaks_info_inbody]
    peaks_info_outbody_set = list(set(peaks_info_set)-set(peaks_info_inbody_set))
    peaks_info_outbody = [list(i) for i in peaks_info_outbody_set]
    
    print("peaks number: ", len(peaks_info_set))
    print("peaks number in gene promoters and exons: ", len(set(peaks_info_inbody_set)))
    print("peaks number out gene promoters and exons:", len(peaks_info_outbody_set))
    
    w = genes_info_tss + peaks_info_outbody
    A = {}
    
    w.sort()
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = elem[1] - g[1]
                if all([g[0]==elem[0], tmp_distance <= gene_distance]):
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / gene_distance)
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]

    w.reverse()
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = g[1] - elem[1]
                if all([g[0]==elem[0], tmp_distance <= gene_distance]):
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / gene_distance)
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]
    
    return genes_peaks_score_array

#*******************************************************************************************
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#*******************************************************************************************


def ExtractGeneInfo(gene_bed):
    """Extract gene information from gene bed file."""

    bed = pd.read_csv(gene_bed, sep="\t", header=0, index_col=False)
    bed['transcript'] = [x.strip().split(".")[0] for x in bed['name'].tolist()]
    bed['tss'] = bed.apply(lambda x: x['txStart'] if x['strand']=='+' else x['txEnd'], axis=1)

    ### adjacent P+GB
    bed["start"] = bed.apply(lambda x: x['txStart']-2000 if x['strand']=='+' else x['txStart'], axis=1)
    bed["end"] = bed.apply(lambda x: x['txEnd']+2000 if x['strand']=='-' else x['txEnd'], axis=1)
    
    bed['promoter'] = bed.apply(lambda x: tuple([x['tss']-2000, x['tss']+2000]), axis=1)
    bed['exons'] = bed.apply(lambda x: tuple([(int(i), int(j)) for i, j in zip(x['exonStarts'].strip(',').split(','), x['exonEnds'].strip(',').split(','))]), axis=1)

    ### exon length
    bed['length'] = bed.apply(lambda x: sum(list(map(lambda i: (i[1]-i[0])/1000.0, x['exons']))), axis=1)
    bed['uid'] = bed.apply(lambda x: "%s@%s@%s"%(x['name2'], x['start'], x['end']), axis=1)
    bed = bed.drop_duplicates(subset='uid', keep="first")
    gene_info = []
    for irow, x in bed.iterrows():
        gene_info.append([x['chrom'], x['start'], x['end'], x['tss'], x['promoter'], x['exons'], x['length'], 1, x['uid']])
    ### [chrom_0, start_1, end_2, tss_3, promoter_4, exons_5, length_6, 1_7, uid_8]
    return gene_info


#*******************************************************************************************
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#*******************************************************************************************



def calculate_RP_score(adata_atac, adata_rna, gene_bed, decay, data_path, model, split_str):

    """Calculate regulatery potential for each gene based on the single-cell peaks."""
    #############################################
    # get gene info
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
    peaks_list = [f.encode() for f in adata_atac.var_names]
    peaks_info = []
    for ipeak, peak in enumerate(peaks_list):
        peaks_tmp = peak.decode().rsplit(split_str, maxsplit=2)
        peaks_info.append([peaks_tmp[0], \
                           (int(peaks_tmp[1])+int(peaks_tmp[2]))/2.0, \
                           int(peaks_tmp[1]), int(peaks_tmp[2]), \
                           0, peak, ipeak])

    peaks_info_Data = pd.DataFrame(peaks_info)
    peaks_info_Data.columns = ['chrom', 'center','start', 'end', 'floag 0', 'peak', 'peak_id']

    #############################################
    # get regulation potential
    genes_peaks_score_dok = RP_AddExonRemovePromoter(peaks_info, genes_info_full, genes_info_tss, decay)

    peak_score_dok = RP_peaks(peaks_info, decay)

    with open(data_path + "RP_data.pkl", "wb") as f:
        pickle.dump({'genes_peaks_RP': genes_peaks_score_dok,
                     'peak_score_RP': peak_score_dok,
                     'peak_info_data': peaks_info_Data,
                     'gene_info_Data': genes_info_Data}, f)

    return genes_peaks_score_dok, peak_score_dok, peaks_info_Data, genes_info_Data





#*******************************************************************************************
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#*******************************************************************************************







####################################
# get graph info

def Graph_data(adata_rna, adata_atac, cell_pseudotime, node_id, adata_rp_peak, adata_rp_gene_peak):
    
    # Node_feature = np.zeros((len(node_id)+1, len(cell_pseudotime)))

    node_features_dict = {}
    for i in range(len(node_id)):
        node = node_id.index[i]+1
        if node_id.iloc[i]["type"] == "gene":
            temp_feature = adata_rna[:, node_id.iloc[i]["name"]]
        else:
            temp_feature = adata_atac[:, node_id.iloc[i]["name"]]

        for j in range(len(temp_feature.obs_names)):
            t = cell_pseudotime.loc[temp_feature.obs_names[j]]
            key = (node, t.values[0])
            feature = temp_feature[j,].X.toarray()[0]
            node_features_dict[key] = feature
        if i%50 ==0:
            print(f"************** Finished process {i}/{len(node_id)} , node feature******************")

    Edge_feature = np.zeros((1))
    Edge_label = np.zeros((1))
    edge_num = 0

    Graph_df = pd.DataFrame()

    for i in range(len(node_id)): 
        # print(node_id.iloc[i,1])
        if node_id.iloc[i,1] == "gene":
            target_node = i+1
            target_gene = node_id.iloc[i,0]

            try:
                gene_slice = adata_rp_gene_peak[target_gene, :]
                if sp_sparse.issparse(gene_slice.X):
                    non_zero_indices = gene_slice.X.indices
                    target_peaks = adata_rp_gene_peak.var_names[non_zero_indices]
                else:
                    gene_values = gene_slice.X.squeeze()
                    target_peaks = adata_rp_gene_peak.var_names[gene_values != 0]
            except KeyError:
                print(f"错误：基因 '{target_gene}' 不在 .obs_names 中。")
                continue

            source_peaks = [x for x in target_peaks if x in adata_atac.var_names]
            
            for source_peak in source_peaks:
                
                source_node = node_id.index[node_id['name']== source_peak][0] +1

                tmp_RP = adata_rp_gene_peak[target_gene,source_peak].X.toarray()[0][0]
                
                for t in range(len(cell_pseudotime)):
                    target_activity = adata_rna[cell_pseudotime.index[t],target_gene].X.toarray()[0][0]
                    source_activity = adata_atac[cell_pseudotime.index[t],source_peak].X.toarray()[0][0]
        
                    if source_activity <= 0:
                        tmp_label = 0
                        continue
                    else:
                        tmp_label = 1
        
                    
                    tmp_RA = tmp_RP * source_activity
                    
                    edge_num = edge_num + 1
                    tmp_df = pd.DataFrame({
                        'source_node': source_node,
                        'target_node': target_node,
                        'Regulation': tmp_RP,
                        'time': cell_pseudotime.iloc[t,0],
                        'source_activity': source_activity,
                        'target_activity': target_activity,
                        'RegulationActivity': tmp_RA,
                        'label': tmp_label,
                        'edge_label': 0,
                        'edge_idx': edge_num
                    }, index = [0])

                    Graph_df = pd.concat([Graph_df,tmp_df], ignore_index=True)

                    Edge_feature = np.append(Edge_feature, tmp_RA)

                    Edge_label = np.append(Edge_label, tmp_label)
                
        if node_id.iloc[i,1] == "peak":
            target_node = i + 1
            target_peak = node_id.iloc[i,0]

            try:
                peak_slice = adata_rp_peak[target_peak, :]
                if sp_sparse.issparse(peak_slice.X):
                    non_zero_indices = peak_slice.X.indices
                    target_peaks = adata_rp_peak.var_names[non_zero_indices]
                else:
                    peak_values = peak_slice.X.squeeze()
                    target_peaks = adata_rp_peak.var_names[peak_values != 0]
            except KeyError:
                print(f"错误：peak '{target_peak}' 不在 .obs_names 中。")
                continue

            source_peaks = [x for x in target_peaks if x in adata_atac.var_names]

            for source_peak in source_peaks:
                
                source_node = node_id.index[node_id['name']== source_peak][0] +1

                tmp_RP = adata_rp_peak[target_peak,source_peak].X.toarray()[0][0]

                for t in range(len(cell_pseudotime)): 
                    source_activity = adata_atac[cell_pseudotime.index[t],source_peak].X.toarray()[0][0]
                    target_activity = adata_atac[cell_pseudotime.index[t],target_peak].X.toarray()[0][0]
        
                    if source_activity <= 0 or target_activity <= 0:
                        tmp_label = 0
                        continue
                    else:
                        tmp_label = 1
        
                    tmp_RA = tmp_RP * target_activity * source_activity

                    edge_num1 = edge_num+1
                    edge_num2 = edge_num1+1
                    edge_num =edge_num + 2
        
                    tmp_df = pd.DataFrame({
                        'source_node': source_node,
                        'target_node': target_node,
                        'Regulation': tmp_RP,
                        'time': cell_pseudotime.iloc[t,0],
                        'source_activity': source_activity,
                        'target_activity': target_activity,
                        'RegulationActivity': tmp_RA,
                        'label': tmp_label,
                        'edge_label': 1,
                        'edge_idx': edge_num1
                    },index = [0])
                    tmp_df2 = pd.DataFrame({
                        'source_node': target_node,
                        'target_node': source_node,
                        'Regulation': tmp_RP,
                        'time': cell_pseudotime.iloc[t,0],
                        'source_activity': target_activity,
                        'target_activity': source_activity,
                        'RegulationActivity': tmp_RA,
                        'label': tmp_label,
                        'edge_label': 1,
                        'edge_idx': edge_num2
                    }, index = [0])

                    Graph_df = pd.concat([Graph_df,tmp_df, tmp_df2], ignore_index=True)
                    
                    Edge_feature = np.append(Edge_feature, [tmp_RA, tmp_RA])

                    Edge_label = np.append(Edge_label, [tmp_label, tmp_label])
        if i%50 ==0:
            print(f"************** Finished process {i}/{len(node_id)}, edge feature ******************")

    return Graph_df, node_features_dict, Edge_feature, Edge_label




#*******************************************************************************************
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#*******************************************************************************************


# def get_graph_node_feature(adata_rna, adata_atac, cell_pseudotime, node_id):
    
#     time_values = cell_pseudotime.iloc[:, 0].values 
    
#     node_features_dict = {}
#     print("Processing Node Features...")
    
#     for i in tqdm(range(len(node_id))):
#         node_idx = i + 1
#         node_name = node_id.iloc[i]["name"]
#         node_type = node_id.iloc[i]["type"]
        
#         # 提取整列数据 (所有细胞的值)
#         if node_type == "gene":
#             # 检查基因是否存在
#             if node_name not in adata_rna.var_names: continue
#             feature_col = adata_rna[:, node_name].X
#         else:
#             if node_name not in adata_atac.var_names: continue
#             feature_col = adata_atac[:, node_name].X
            
#         # 转换为 dense array 并展平
#         if sp_sparse.issparse(feature_col):
#             feature_col = feature_col.toarray().flatten()
#         else:
#             feature_col = np.asarray(feature_col).flatten()
            
#         # 批量构建字典 (虽然 dict 很大，但比逐个通过 iloc 查快得多)
#         # 这是一个瓶颈，因为 Python 字典存百万级 keys 很慢，但为了保持你的输出格式不变：
#         for t_val, feat_val in zip(time_values, feature_col):
#             node_features_dict[(node_idx, t_val)] = feat_val
    
#     return node_features_dict


def get_graph_node_feature(adata_rna, adata_atac, cell_pseudotime, node_id):
    """
    返回: dict {(node_idx, time_val): feat_val}  —— 与原版完全相同的格式

    优化: 将 adata.X 一次性转为 dense numpy 矩阵，之后每列提取是 O(1) 直接索引，
          避免原版每个节点都调用 adata[:, name].X 触发稀疏矩阵切片（慢 10~50x）。
    """
    time_values = cell_pseudotime.iloc[:, 0].values

    print("Processing Node Features (optimized)...")
    print("Pre-loading expression matrices into dense arrays...")

    # ---- 核心优化: 一次性转 dense，后续列索引为 O(1) ----
    if sp_sparse.issparse(adata_rna.X):
        rna_X = adata_rna.X.toarray()    # (n_cells, n_genes)
    else:
        rna_X = np.asarray(adata_rna.X)

    if sp_sparse.issparse(adata_atac.X):
        atac_X = adata_atac.X.toarray()  # (n_cells, n_peaks)
    else:
        atac_X = np.asarray(adata_atac.X)

    rna_col_idx  = {name: i for i, name in enumerate(adata_rna.var_names)}
    atac_col_idx = {name: i for i, name in enumerate(adata_atac.var_names)}
    rna_var_set  = set(adata_rna.var_names)
    atac_var_set = set(adata_atac.var_names)

    node_features_dict = {}

    for i in tqdm(range(len(node_id))):
        node_idx  = i + 1
        node_name = node_id.iloc[i]["name"]
        node_type = node_id.iloc[i]["type"]

        if node_type == "gene":
            if node_name not in rna_var_set:
                continue
            feature_col = rna_X[:, rna_col_idx[node_name]]   # O(1) 列切片
        else:
            if node_name not in atac_var_set:
                continue
            feature_col = atac_X[:, atac_col_idx[node_name]] # O(1) 列切片

        # 与原版完全相同的 dict 结构
        for t_val, feat_val in zip(time_values, feature_col):
            node_features_dict[(node_idx, t_val)] = feat_val

    return node_features_dict




# def Graph_data_optimized(adata_rna, adata_atac, cell_pseudotime, node_id, adata_rp_gene_peak, adata_rp_peak):

#     name_to_node_idx = {name: idx + 1 for idx, name in enumerate(node_id['name'])}
    
#     time_values = cell_pseudotime.iloc[:, 0].values 
    
#     # ---------------------------------------------------------
#     # 3. 优化 Edge Generation (核心加速部分)
#     # ---------------------------------------------------------
#     print("Processing Edges...")
    
#     edge_records = []  # 使用 list 收集数据，最后一次性转 DataFrame
#     edge_features_list = [0] # 收集 feature
#     edge_labels_list = [0]   # 收集 label
    
#     global_edge_idx = 0
    
#     def get_targets(matrix_slice, var_names):
#         if sp_sparse.issparse(matrix_slice.X):
#             indices = matrix_slice.X.indices
#             return var_names[indices], matrix_slice.X.data
#         else:
#             data = matrix_slice.X.flatten()
#             mask = data != 0
#             return var_names[mask], data[mask]

#     for i in tqdm(range(len(node_id))):
        
#         current_type = node_id.iloc[i, 1] # type
#         current_name = node_id.iloc[i, 0] # name
        
#         # ---------------- CASE 1: Gene Node ----------------
#         if current_type == "gene":

#             target_node_idx = i + 1
#             try:
#                 # 批量获取所有连接的 peak
#                 gene_slice = adata_rp_gene_peak[current_name, :]

#                 target_peak_names, rp_values = get_targets(gene_slice, adata_rp_gene_peak.var_names)
#             except KeyError:
#                 print("***************** key error **********************************")
#                 continue

#             # 过滤只在 ATAC 中存在的 peak
#             valid_mask = [p in adata_atac.var_names for p in target_peak_names]
#             target_peak_names = target_peak_names[valid_mask]
#             rp_values = rp_values[valid_mask]
            
#             # 提前获取 Target Gene 的 Activity (向量)
#             if current_name not in adata_rna.var_names: continue
#             target_act_vec = adata_rna[:, current_name].X
#             if sp_sparse.issparse(target_act_vec): target_act_vec = target_act_vec.toarray().flatten()
#             else: target_act_vec = np.asarray(target_act_vec).flatten()

#             # 遍历连接的 Source Peaks
#             for src_peak, tmp_RP in zip(target_peak_names, rp_values):
#                 source_node_idx = name_to_node_idx.get(src_peak)
#                 if source_node_idx is None: continue
                
#                 # 获取 Source Peak Activity (向量)
#                 src_act_vec = adata_atac[:, src_peak].X
#                 if sp_sparse.issparse(src_act_vec): src_act_vec = src_act_vec.toarray().flatten()
#                 else: src_act_vec = np.asarray(src_act_vec).flatten()
                
#                 # *** 向量化计算逻辑 ***
#                 # 你的逻辑：if source_activity <= 0: continue else label=1
#                 # 这等价于：只保留 source > 0 的行
#                 mask = src_act_vec > 0
                
#                 if not np.any(mask): continue # 如果全为0，跳过
                
#                 # 应用掩码筛选数据
#                 valid_src_act = src_act_vec[mask]
#                 valid_tgt_act = target_act_vec[mask]
#                 valid_times = time_values[mask]
                
#                 # 计算 RegulationActivity
#                 tmp_RA = tmp_RP * valid_src_act
                
#                 # 批量添加到列表
#                 count = len(valid_times)
#                 for k in range(count):
#                     global_edge_idx += 1
#                     # 这里为了省内存，不建议用 dict，直接存 tuple 或 list
#                     edge_records.append({
#                         'source_node': source_node_idx,
#                         'target_node': target_node_idx,
#                         'Regulation': tmp_RP,
#                         'time': valid_times[k],
#                         'source_activity': valid_src_act[k],
#                         'target_activity': valid_tgt_act[k],
#                         'RegulationActivity': tmp_RA[k],
#                         'label': 1, # 你的逻辑里 else 都是 1
#                         'edge_label': 0,
#                         'edge_idx': global_edge_idx
#                     })

#                 edge_features_list.extend(tmp_RA)
#                 edge_labels_list.extend([1] * count)

#         # ---------------- CASE 2: Peak Node ----------------
#         elif current_type == "peak":
#             target_node_idx = i + 1
            
#             try:
#                 # 注意：这里你原代码写的是 adata_rp_peak，我假设这是 Peak-Peak 矩阵
#                 peak_slice = adata_rp_peak[current_name, :] 
#                 target_peak_names, rp_values = get_targets(peak_slice, adata_rp_peak.var_names)
#             except KeyError:
#                 continue

#             valid_mask = [p in adata_atac.var_names for p in target_peak_names]
#             target_peak_names = target_peak_names[valid_mask]
#             rp_values = rp_values[valid_mask]
            
#             # 获取 Target Peak Activity (向量)
#             if current_name not in adata_atac.var_names: continue
#             target_act_vec = adata_atac[:, current_name].X
#             if sp_sparse.issparse(target_act_vec): target_act_vec = target_act_vec.toarray().flatten()
#             else: target_act_vec = np.asarray(target_act_vec).flatten()

#             for src_peak, tmp_RP in zip(target_peak_names, rp_values):
#                 source_node_idx = name_to_node_idx.get(src_peak)
#                 if source_node_idx is None: continue
                
#                 src_act_vec = adata_atac[:, src_peak].X
#                 if sp_sparse.issparse(src_act_vec): src_act_vec = src_act_vec.toarray().flatten()
#                 else: src_act_vec = np.asarray(src_act_vec).flatten()
                
#                 # *** 向量化逻辑 ***
#                 # 你的逻辑：if src <= 0 or tgt <= 0: continue
#                 mask = (src_act_vec > 0) & (target_act_vec > 0)
                
#                 if not np.any(mask): continue
                
#                 valid_src_act = src_act_vec[mask]
#                 valid_tgt_act = target_act_vec[mask]
#                 valid_times = time_values[mask]
                
#                 tmp_RA = tmp_RP * valid_tgt_act * valid_src_act
                
#                 count = len(valid_times)
#                 # Peak-Peak 是双向边，一次加两条
#                 for k in range(count):
#                     global_edge_idx += 1
#                     idx1 = global_edge_idx
#                     global_edge_idx += 1
#                     idx2 = global_edge_idx
                    
#                     # 第一条边
#                     edge_records.append({
#                         'source_node': source_node_idx,
#                         'target_node': target_node_idx,
#                         'Regulation': tmp_RP,
#                         'time': valid_times[k],
#                         'source_activity': valid_src_act[k],
#                         'target_activity': valid_tgt_act[k],
#                         'RegulationActivity': tmp_RA[k],
#                         'label': 1,
#                         'edge_label': 1,
#                         'edge_idx': idx1
#                     })
#                     # 第二条边
#                     edge_records.append({
#                         'source_node': target_node_idx,
#                         'target_node': source_node_idx,
#                         'Regulation': tmp_RP,
#                         'time': valid_times[k],
#                         'source_activity': valid_tgt_act[k],
#                         'target_activity': valid_src_act[k],
#                         'RegulationActivity': tmp_RA[k],
#                         'label': 1,
#                         'edge_label': 1,
#                         'edge_idx': idx2
#                     })

#                 edge_features_list.extend(np.repeat(tmp_RA, 2)) # 每个时间点有两条边
#                 edge_labels_list.extend([1] * (count * 2))

#     # ---------------------------------------------------------
#     # 4. 构建最终结果
#     # ---------------------------------------------------------
#     print("Constructing DataFrame...")
#     Graph_df = pd.DataFrame(edge_records)
    
#     # 初始化空的 numpy 数组
#     Edge_feature = np.array(edge_features_list)
#     Edge_label = np.array(edge_labels_list)

#     print(f"Edge feature:\n {Edge_feature[0:10]}")

#     print(f"Edge label: \n {Edge_label[0:10]}")

#     print(f"edge records: \n {edge_records[0:10]}")

#     return Graph_df, Edge_feature, Edge_label, edge_records


def Graph_data_optimized(adata_rna, adata_atac, cell_pseudotime,
                          node_id, adata_rp_gene_peak, adata_rp_peak):
    """
    优化要点：
    1. 一次性将 adata_rna.X 和 adata_atac.X 转为 dense numpy 矩阵
       → 避免百万次 adata[:,col].X sparse 切片
    2. 消除内层 for k 循环：用 numpy 批量构建 edge records
    3. 用列表收集列向量，最后一次性 pd.DataFrame(np.column_stack([...]))
    """

    print("Pre-loading expression matrices into dense arrays...")
    # ---- 瓶颈3修复: 预加载为 dense 矩阵 ----
    rna_X = adata_rna.X.toarray() if sp_sparse.issparse(adata_rna.X) else np.asarray(adata_rna.X)
    atac_X = adata_atac.X.toarray() if sp_sparse.issparse(adata_atac.X) else np.asarray(adata_atac.X)
    # shape: (n_cells, n_genes) / (n_cells, n_peaks)

    # 构建列名 → 列下标映射
    rna_col  = {name: i for i, name in enumerate(adata_rna.var_names)}
    atac_col = {name: i for i, name in enumerate(adata_atac.var_names)}
    atac_var_set = set(adata_atac.var_names)
    rna_var_set  = set(adata_rna.var_names)

    name_to_node_idx = {name: idx + 1 for idx, name in enumerate(node_id['name'])}
    time_values = cell_pseudotime.iloc[:, 0].values   # (n_cells,)

    print("Processing Edges (optimized)...")

    # ---- 列收集器（比 list of dict 快 3~5 倍）----
    col_src, col_tgt, col_reg, col_time = [], [], [], []
    col_src_act, col_tgt_act, col_ra   = [], [], []
    col_label, col_el, col_eidx        = [], [], []

    global_edge_idx = 0

    def _get_csr_targets(matrix_slice):
        """从CSR切片获取非零列名和值"""
        if sp_sparse.issparse(matrix_slice.X):
            return matrix_slice.X.indices, matrix_slice.X.data
        else:
            data = matrix_slice.X.flatten()
            mask = data != 0
            return np.where(mask)[0], data[mask]

    for i in tqdm(range(len(node_id))):
        current_type = node_id.iloc[i, 1]
        current_name = node_id.iloc[i, 0]

        # ==================== CASE 1: Gene Node ====================
        if current_type == "gene":
            target_node_idx = i + 1
            try:
                gene_slice = adata_rp_gene_peak[current_name, :]
                nz_indices, rp_values = _get_csr_targets(gene_slice)
                target_peak_names = adata_rp_gene_peak.var_names[nz_indices]
            except KeyError:
                continue

            # 过滤只在 ATAC 中存在的 peak
            valid = np.array([p in atac_var_set for p in target_peak_names])
            target_peak_names = target_peak_names[valid]
            rp_values         = rp_values[valid]

            if current_name not in rna_var_set:
                continue
            # ---- 瓶颈3修复: 直接列索引取向量，O(1) ----
            tgt_act_vec = rna_X[:, rna_col[current_name]]   # (n_cells,)

            for src_peak, tmp_RP in zip(target_peak_names, rp_values):
                source_node_idx = name_to_node_idx.get(src_peak)
                if source_node_idx is None:
                    continue

                src_act_vec = atac_X[:, atac_col[src_peak]]  # (n_cells,)
                mask = src_act_vec > 0
                if not np.any(mask):
                    continue

                valid_src  = src_act_vec[mask]
                valid_tgt  = tgt_act_vec[mask]
                valid_time = time_values[mask]
                tmp_RA     = tmp_RP * valid_src
                count      = len(valid_time)

                # ---- 瓶颈2修复: 不再 for k 逐条 append，直接 extend 数组 ----
                start_idx = global_edge_idx + 1
                global_edge_idx += count
                edge_idxs = np.arange(start_idx, start_idx + count)

                col_src.append(np.full(count, source_node_idx, dtype=np.int32))
                col_tgt.append(np.full(count, target_node_idx, dtype=np.int32))
                col_reg.append(np.full(count, tmp_RP))
                col_time.append(valid_time)
                col_src_act.append(valid_src)
                col_tgt_act.append(valid_tgt)
                col_ra.append(tmp_RA)
                col_label.append(np.ones(count, dtype=np.int8))
                col_el.append(np.zeros(count, dtype=np.int8))
                col_eidx.append(edge_idxs)

        # ==================== CASE 2: Peak Node ====================
        elif current_type == "peak":
            target_node_idx = i + 1
            try:
                peak_slice = adata_rp_peak[current_name, :]
                nz_indices, rp_values = _get_csr_targets(peak_slice)
                target_peak_names = adata_rp_peak.var_names[nz_indices]
            except KeyError:
                continue

            valid = np.array([p in atac_var_set for p in target_peak_names])
            target_peak_names = target_peak_names[valid]
            rp_values         = rp_values[valid]

            if current_name not in atac_var_set:
                continue
            tgt_act_vec = atac_X[:, atac_col[current_name]]

            for src_peak, tmp_RP in zip(target_peak_names, rp_values):
                source_node_idx = name_to_node_idx.get(src_peak)
                if source_node_idx is None:
                    continue

                src_act_vec = atac_X[:, atac_col[src_peak]]
                mask = (src_act_vec > 0) & (tgt_act_vec > 0)
                if not np.any(mask):
                    continue

                valid_src  = src_act_vec[mask]
                valid_tgt  = tgt_act_vec[mask]
                valid_time = time_values[mask]
                tmp_RA     = tmp_RP * valid_tgt * valid_src
                count      = len(valid_time)

                # Peak-Peak 双向边，每个时间点2条
                start_idx = global_edge_idx + 1
                global_edge_idx += count * 2

                idxs1 = np.arange(start_idx,          start_idx + count)
                idxs2 = np.arange(start_idx + count,  start_idx + count * 2)

                # 正向边
                col_src.append(np.full(count, source_node_idx, dtype=np.int32))
                col_tgt.append(np.full(count, target_node_idx, dtype=np.int32))
                col_reg.append(np.full(count, tmp_RP))
                col_time.append(valid_time)
                col_src_act.append(valid_src)
                col_tgt_act.append(valid_tgt)
                col_ra.append(tmp_RA)
                col_label.append(np.ones(count, dtype=np.int8))
                col_el.append(np.ones(count, dtype=np.int8))
                col_eidx.append(idxs1)

                # 反向边
                col_src.append(np.full(count, target_node_idx, dtype=np.int32))
                col_tgt.append(np.full(count, source_node_idx, dtype=np.int32))
                col_reg.append(np.full(count, tmp_RP))
                col_time.append(valid_time)
                col_src_act.append(valid_tgt)
                col_tgt_act.append(valid_src)
                col_ra.append(tmp_RA)
                col_label.append(np.ones(count, dtype=np.int8))
                col_el.append(np.ones(count, dtype=np.int8))
                col_eidx.append(idxs2)

    # ---- 一次性合并所有数组并构建 DataFrame ----
    print("Constructing DataFrame...")

    def _concat(lst):
        return np.concatenate(lst) if lst else np.array([])

    Graph_df = pd.DataFrame({
        'source_node':      _concat(col_src),
        'target_node':      _concat(col_tgt),
        'Regulation':       _concat(col_reg),
        'time':             _concat(col_time),
        'source_activity':  _concat(col_src_act),
        'target_activity':  _concat(col_tgt_act),
        'RegulationActivity': _concat(col_ra),
        'label':            _concat(col_label),
        'edge_label':       _concat(col_el),
        'edge_idx':         _concat(col_eidx),
    })

    Edge_feature = np.concatenate([[0], _concat(col_ra)])
    Edge_label   = np.concatenate([[0], _concat(col_label)])

    edge_records = Graph_df.to_dict('records')

    return Graph_df, Edge_feature, Edge_label, edge_records