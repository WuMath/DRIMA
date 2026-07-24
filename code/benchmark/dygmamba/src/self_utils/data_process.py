import os
import sys

import scipy.sparse as sp
from scipy.io import mmread
import pandas as pd
import networkx as nx
import anndata as ad
import scanpy as sc
import numpy as np
import pickle

from pkg_resources import resource_filename

from self_utils.regulation_info import genescore, new_genescore



def Anndata_read(data_path, feature_file1, feature_file2, feature1, feature2):

    X = mmread(data_path + "counts.mtx").tocsr()
    obs = pd.read_feather(data_path + "metadata.feather")
    obs.index = pd.read_csv(data_path + feature_file1, header=0)[feature1]
    var = pd.DataFrame(index=pd.read_csv(data_path + feature_file2, header=0)[feature2])
    
    adata = ad.AnnData(X=X.T, obs=obs, var=var)

    return adata



def data_reader(data_path):
    atac_path = data_path + "ATACData/"
    rna_path = data_path + "RNAData/"

    rna_counts_matrix = mmread(rna_path + "matrix.mtx").tocsr()
    rna_genes_df = pd.read_csv(rna_path + "features.tsv", sep='\t', header=None, names=[ 'gene_symbol'])
    rna_barcodes_df = pd.read_csv(rna_path + "barcodes.tsv", sep='\t', header=None, names=['barcode'])

    rna_var = rna_genes_df.set_index('gene_symbol')
    rna_obs = pd.DataFrame(index=rna_barcodes_df['barcode'])

    adata_rna = ad.AnnData(
        X=rna_counts_matrix.T,
        obs=rna_obs,
        var=rna_var
    )
    print("RNA AnnData 对象基础构建完成:")
    print(adata_rna)


    atac_counts_matrix = mmread(atac_path + "matrix.mtx").tocsr()
    atac_peaks_df = pd.read_csv(atac_path + "features.tsv", sep='\t', header=None, names=['peak_id'])
    atac_barcodes_df = pd.read_csv(atac_path + "barcodes.tsv", sep='\t', header=None, names=['barcode'])

    atac_var = atac_peaks_df.set_index('peak_id')
    atac_obs = pd.DataFrame(index=atac_barcodes_df['barcode'])

    adata_atac = ad.AnnData(
        X=atac_counts_matrix.T,
        obs=atac_obs,
        var=atac_var
    )
    print("ATAC AnnData 对象基础构建完成:")
    print(adata_atac)


    shared_metadata_df = pd.read_csv(atac_path + "cell_metadata.csv")

    shared_metadata_df = shared_metadata_df.set_index('barcode')

    adata_rna.obs = shared_metadata_df.reindex(adata_rna.obs_names)
    adata_atac.obs = shared_metadata_df.reindex(adata_atac.obs_names)

    print("细胞元数据已成功合并！")

    return adata_rna, adata_atac


####################################
# get graph info

def Graph_data(adata_rna, adata_atac, cell_pseudotime, node_id, id_gene_df, id_peak_df, rp_peak_data, rp_gene_peak_data):
    
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

    Edge_feature = np.zeros((1))
    Edge_label = np.zeros((1))
    edge_num = 0;

    Graph_df = pd.DataFrame()

    for i in range(len(node_id)): 
        # print(node_id.iloc[i,1])
        if node_id.iloc[i,1] == "gene":
            target_node = i+1
            target_gene = node_id.iloc[i,0]
            target_gene_RP_id = id_gene_df.loc[target_gene,'id']

            source_peaks_RP_id = rp_gene_peak_data.getrow(target_gene_RP_id).indices
            
            source_peaks = id_peak_df.index[id_peak_df["id"].isin(source_peaks_RP_id)].tolist()

            source_peaks = [x for x in source_peaks if x in adata_atac.var_names]
            
            for source_peak in source_peaks:
                
                source_node = node_id.index[node_id['name']== source_peak][0] +1
                
                source_peak_RP_id = id_peak_df.loc[source_peak,'id']

                tmp_RP = rp_gene_peak_data[target_gene_RP_id, source_peak_RP_id]
                
                for t in range(len(cell_pseudotime)):
                    target_activity = adata_rna[cell_pseudotime.index[t],target_gene].X.toarray()[0][0]
                    source_activity = adata_atac[cell_pseudotime.index[t],source_peak].X.toarray()[0][0]
        
                    if source_activity <= 0:
                        tmp_label = 0
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
            target_peak_RP_id = id_peak_df.loc[target_peak,'id']

            # Node_feature[i+1,:] = adata_atac[:,target_peak].X.toarray().T

            source_peaks_RP_ids = rp_peak_data.getrow(target_peak_RP_id).indices
            source_peaks = id_peak_df.index[id_peak_df["id"].isin(source_peaks_RP_ids)].tolist()

            # print(len(source_peaks))
            source_peaks = [x for x in source_peaks if x in adata_atac.var_names]
            # print(len(source_peaks))

            for source_peak in source_peaks:
                
                source_node = node_id.index[node_id['name']== source_peak][0] +1
                
                source_peak_RP_id = id_peak_df.loc[source_peak,'id']

                tmp_RP = rp_peak_data[target_peak_RP_id, source_peak_RP_id]

                for t in range(len(cell_pseudotime)): 
                    source_activity = adata_atac[cell_pseudotime.index[t],source_peak].X.toarray()[0][0]
                    target_activity = adata_atac[cell_pseudotime.index[t],target_peak].X.toarray()[0][0]
        
                    if source_activity <= 0 or target_activity <= 0:
                        tmp_label = 0
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

    return Graph_df, node_features_dict, Edge_feature, Edge_label




def Load_pseudotime_Data(data_path, data_type, lineage = "Lineage1"):
    
    file_path = data_path + "Pseudotime/" + "pseudotime.feather"
    
    cell_pseudotime = pd.read_feather(file_path)
    if data_type =="real":
        cell_pseudotime.set_index("cell_id", inplace=True)
        cell_pseudotime.columns = ["pseudotime", "umap1", "umap2"]
    else:
        cell_pseudotime.set_index("cell_id", inplace=True)
    #########
    # filter
    ########
    cell_pseudotime = cell_pseudotime.dropna(subset = ["pseudotime"])
    cell_pseudotime = cell_pseudotime.sort_values(by = "pseudotime", ascending = True)

    return cell_pseudotime




def ReadPreprocessData(adata_rna, adata_atac, cell_pseudotime, data_path, file_path):

    #############################################
    # regulation
    ####################################
    outprefix = "Regulation"
    genedistance = 10000
    species = "GRCh38" 
    model = "Enhanced"

    id_gene_df, id_peak_df, adata_rp_gene_peak, adata_rp_peak \
                = new_genescore(file_path, outprefix, adata_atac, adata_rna, \
                        genedistance, species, model)

    cell_id = cell_pseudotime.index
    ########################
    # get vaild gene and peaks in study
    gene_data = adata_rna.var_names
    gene_criteria = id_gene_df.index
    gene_names = set(gene_data).intersection(set(gene_criteria))
    adata_rna = adata_rna[cell_id,list(gene_names)].copy()

    if sp.issparse(adata_atac.X):
        total_counts = np.asarray(adata_atac.X.sum(axis=0)).flatten()
    else:
        total_counts = adata_atac.X.sum(axis=0)

    flag_peak =0        
    if flag_peak ==1:
        zero_sum_mask = (total_counts == 0)
        removed_genes = adata_atac.var_names[zero_sum_mask].tolist()
        adata_atac = adata_atac[:, ~zero_sum_mask].copy()
        all_peaks_df = pd.DataFrame({
            'peak_name': adata_atac.var_names,
            'total_count': total_counts[~zero_sum_mask]
        }) 
    else:
        all_peaks_df = pd.DataFrame({
            'peak_name': adata_atac.var_names,
            'total_count': total_counts
        }) 

    peak_data = adata_atac.var_names
    peak_criteria = id_peak_df.index
    peak_names = set(peak_data).intersection(set(peak_criteria))
    adata_atac = adata_atac[cell_id, list(peak_names)].copy()

    ##################################################
    # compute highly variable gene for filter gene
    adata_rna.layers["counts"] = adata_rna.X.copy()
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=200)
    hv_genes = adata_rna.var[adata_rna.var['highly_variable']].index.tolist()

    adata_rna.X = adata_rna.layers["counts"]

    ####################################################
    # remove unregulated genes
        
    all_peaks_df = all_peaks_df.sort_values(by='total_count', ascending=False)

    peaks_names = set()
    error_gene = []
    num = 0
    for g in hv_genes:
        g_id = id_gene_df.loc[g,'id']
        tmp_peaks_id = adata_rp_gene_peak.X.getrow(g_id).indices
        if len(tmp_peaks_id) == 0:
            error_gene.append(g)
            continue
        tmp_peaks = id_peak_df.index[id_peak_df["id"].isin(tmp_peaks_id)].tolist()

        subset_df = all_peaks_df[all_peaks_df['peak_name'].isin(tmp_peaks)].copy()

        num_to_select = max(1, int(len(subset_df) * 0.5))
        top_percent_df = subset_df.head(num_to_select)
        selected = top_percent_df['peak_name'].tolist()

        num += len(selected)
        # print(num)
        peaks_names = peaks_names.union(set(selected))

    All_gene_names =[]
    for g in hv_genes:
        if g not in error_gene:
            All_gene_names.append(g)

    peaks_names = list(peaks_names)
    adata_rna = adata_rna[cell_id, All_gene_names].copy()
    adata_atac = adata_atac[cell_id, peaks_names].copy()

    #################################################################
    # Transformer Data

    ########################
    # get gene data
    gene_names = adata_rna.var_names

    df_genes = pd.DataFrame({
        "name" : list(gene_names),
        "type" : "gene"
    })
    df_genes = df_genes.sort_values(by = "name", ascending = True)

    print(f' The sequencing gene num: {len(gene_data)}, \n \
    The criteria gene num: {len(gene_criteria)}, \n \
    The common gene num: {len(gene_names)}.')

    ########################
    # get peak data
    peak_names = adata_atac.var_names
    df_peaks = pd.DataFrame({
        "name" : list(peak_names),
        "type" : "peak"
    })

    df_peaks = df_peaks.sort_values(by = "name", ascending = True)

    print(f' The sequencing peak num: {len(peak_data)}, \n \
    The criteria peak num: {len(peak_criteria)}, \n \
    The common peak num: {len(peak_names)}.')

    node_id =  pd.concat([df_genes, df_peaks], ignore_index=True)

    ####################################################
    # get simulation
    ##########################
    Graph_df, Node_feature, Edge_feature, Edge_label = Graph_data(adata_rna, adata_atac, cell_pseudotime,
                        node_id, id_gene_df, id_peak_df, adata_rp_peak.X, adata_rp_gene_peak.X)

    breakpoint()
    Graph_df["Unnamed"] = Graph_df.index
    name_list = ["Unnamed", "source_node", "target_node", "time", "label", "edge_idx"]
    New_Graph = Graph_df[name_list].copy()
    New_Graph.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx']

    return Node_feature, Edge_feature, New_Graph , node_id, Graph_df, Edge_label


def ReadSimData(adata_rna, adata_atac, cell_pseudotime, RP_adata, Real_data):


    nonzero_elements = adata_atac.X.toarray()[adata_atac.X.toarray() != 0]

    mean_activity = np.mean(nonzero_elements)

    #############################################
    # total node
    ####################################
    df_genes = pd.DataFrame({
        "name": adata_rna.var_names,
        "type": "gene"
    })
    df_genes = df_genes.sort_values(by="name", ascending=True)

    df_peaks = pd.DataFrame({
        "name": adata_atac.var_names,
        "type": "peak"
    })

    df_peaks = df_peaks.sort_values(by="name", ascending=True)

    node_id = pd.concat([df_genes, df_peaks], ignore_index=True)

    node_id['id'] = node_id.index + 1

    node_features_dict = {}
    for i in range(len(node_id)):
        node = node_id.iloc[i]["id"]
        if node_id.iloc[i]["type"] == "gene":
            temp_feature = adata_rna[:, node_id.iloc[i]["name"]]
        else:
            temp_feature = adata_atac[:, node_id.iloc[i]["name"]]


        for j in range(len(temp_feature.obs_names)):
            t = cell_pseudotime.loc[temp_feature.obs_names[j]]
            key = (node, t.values[0])
            feature = temp_feature[j,].X.toarray()[0]
            node_features_dict[key] = feature


    #############################################
    # statistical info
    ####################################


    Edge_feature = np.zeros((1))
    Edge_label = np.zeros((1))
    edge_num = 0
    Graph_df = pd.DataFrame()

    for i in range(len(node_id)):

        if node_id.iloc[i, 1] == "gene":
            target_node = i + 1
            target_gene = node_id.iloc[i]["name"]

            col = RP_adata.X[i,:]
            nonzero_rows = col.nonzero()[1]
            peak_names = RP_adata.var_names[nonzero_rows]

            for peak in peak_names:
                source_node = node_id.index[node_id['name'] == peak][0] + 1

                tmp_RP = RP_adata[target_gene, peak].X.toarray()[0][0]

                for t in range(len(cell_pseudotime)):
                    target_activity = adata_rna[cell_pseudotime.index[t], target_gene].X.toarray()[0][0]
                    source_activity = adata_atac[cell_pseudotime.index[t], peak].X.toarray()[0][0]

                    tmp_RA = tmp_RP * source_activity

                    if tmp_RP > 0:

                        edge_num = edge_num + 1

                        edge_label = 1
  
                        if Real_data[target_gene,peak].X.toarray()[0][0] >0:
                            tmp_label = 1
                        else:
                            tmp_label = 0

                        tmp_df = pd.DataFrame({
                            'target_symbol':target_gene,
                            'source_symbol': peak,
                            'source_node': source_node,
                            'target_node': target_node,
                            'Regulation': tmp_RP,
                            'time': cell_pseudotime.iloc[t, 0],
                            'source_activity': source_activity,
                            'target_activity': target_activity,
                            'RegulationActivity': tmp_RA,
                            'label': tmp_label,
                            'edge_label': edge_label,
                            'edge_idx': edge_num
                        }, index=[0])

                        Graph_df = pd.concat([Graph_df, tmp_df], ignore_index=True)

                        Edge_feature = np.append(Edge_feature, tmp_RA)

                        Edge_label = np.append(Edge_label, tmp_label)

    
    Graph_df["Unnamed"] = Graph_df.index
    name_list = ["Unnamed", "source_node", "target_node", "time", "label", "edge_idx", 'edge_label']
    New_Graph = Graph_df[name_list].copy()
    New_Graph.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx', 'edge_label']


    return node_features_dict, Edge_feature, New_Graph , Edge_label, node_id, Graph_df



def Get_Triplet(adata_atac, adata_rna, cell_pseudotime, RP_adata, RT_adata):

    ground_truth_set = {}

    for peak in RT_adata.var_names:
        id_tf = RT_adata[:, peak].X.toarray()
        nonzero_tfs = id_tf.nonzero()[0]
        tf_names = RT_adata.obs_names[nonzero_tfs]

        id_gene = RP_adata[:, peak].X.toarray()
        nonzero_genes = id_gene.nonzero()[0]
        gene_names = RP_adata.obs_names[nonzero_genes]

        if len(tf_names) == 0 or len(gene_names) == 0:
            continue

        for tf in tf_names:
            for g in gene_names:
                for t in range(len(cell_pseudotime)):
                    cell_id = cell_pseudotime.index[t]
                    peak_count = adata_atac[cell_id, peak].X.toarray()[0][0]
                    tf_count = adata_rna[cell_id, tf].X.toarray()[0][0]
                    gene_count = adata_rna[cell_id, g].X.toarray()[0][0]
                    key = (tf, peak, g, cell_id)
                    if tf_count * peak_count * gene_count == 0:
                        continue

                    ground_truth_set[key] = tf_count * peak_count * gene_count

    return ground_truth_set


def create_time_series_graphs(graph_df, node_id_df):
    """
    create NetworkX Graph for each point

    output: 
        a dict with key as timestamp, value as graph
        
    """
    existing_edges_df = graph_df[graph_df['label'] == 1].copy()
    
    name_series = node_id_df['name']
    name_series.index = node_id_df.index + 1  
    id_to_name_map = name_series.to_dict()

    existing_edges_df['source_name'] = existing_edges_df['u'].map(id_to_name_map)
    existing_edges_df['target_name'] = existing_edges_df['i'].map(id_to_name_map)
    
    graphs_by_time = {}
    
    for timestamp, edges_in_timestamp in existing_edges_df.groupby('ts'):
        print(f"\nNow is creting Graph for ts={timestamp} ...")
        
        G = nx.from_pandas_edgelist(
            edges_in_timestamp,
            source='source_name',  
            target='target_name',  
            create_using=nx.DiGraph()
        )
        
        graphs_by_time[timestamp] = G
        
        print(f"Created Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
    return graphs_by_time
