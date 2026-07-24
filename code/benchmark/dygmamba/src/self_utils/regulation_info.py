import os
import sys
sys.path.append("/fs/ess/PCON0022/liyang/BioWuYan/MyselfFunction/")
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
import pickle

def read_peak_data(matrix_file, feature_file, barcode_file, datatype, gene_column = 2):
    """Convert 10x mtx as matrix."""

    matrix = scipy.io.mmread(matrix_file)
    matrix = sp_sparse.csc_matrix(matrix, dtype=np.float32)
    
    features = pd.read_csv(feature_file, header=0)["peaks"].tolist()
    
    barcodes =  pd.read_csv(barcode_file, header=0)["cells"].tolist()
    
    return {"matrix": matrix, "features": features, "barcodes": barcodes}

def RP_peaks(peaks_info, decay):
    Sg = lambda x: 2**(-x)

    peak_distance = 15 * decay
    peaks_score_array = sp_sparse.dok_matrix((len(peaks_info), len(peaks_info)), dtype=np.float64)
    peaks_info.sort()
    P = len(peaks_info)
    for i in range(P):
        elem = peaks_info[i]
        for j in range(i+1,P):
            elem2 = peaks_info[j]
            if elem[0] != elem2[0]:
                continue

            peak_d = elem2[1] -elem[1]
            if peak_d > peak_distance:
                continue
            peaks_score_array[elem[-1],elem2[-1]] = Sg(peak_d/decay)

    return peaks_score_array


def RP_AddExonRemovePromoter(peaks_info, genes_info_full, genes_info_tss, decay):
    """Multiple processing function to calculate regulation potential."""

    Sg = lambda x: 2**(-x)
    checkInclude = lambda x, y: all([x>=y[0], x<=y[1]])
    gene_distance = 15 * decay
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
                        genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
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
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
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
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]
    
    return(genes_peaks_score_array)


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
    return(gene_info)


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


def new_genescore(directory, outprefix, adata_atac, adata_rna, genedistance, species, model="Enhanced"):

    annotation_path = directory + "annotations"

    genebed = os.path.join(annotation_path, species + "_refgenes.txt")

    decay = float(genedistance)

    if os.path.exists(directory + "RP_data.pkl"):
        with open(directory + "RP_data.pkl", "rb") as f:
            RP_data = pickle.load(f)
            rp_gene_peak_data = RP_data["genes_peaks_RP"]
            rp_peak_data = RP_data["peak_score_RP"]
            peak_info_data = RP_data["peak_info_data"]
            gene_info_Data = RP_data["gene_info_Data"]
    else:
        rp_gene_peak_data, rp_peak_data, peak_info_data, gene_info_Data = \
            calculate_RP_score(adata_atac, adata_rna, genebed, decay,
                                 directory, model, split_str="-")

    adata_rp_gene_peak = ad.AnnData(X= rp_gene_peak_data,
                                    obs = pd.DataFrame(index=gene_info_Data["gene symbol"]),
                                    var = pd.DataFrame(index=peak_info_data["peak"]))

    adata_rp_peak = ad.AnnData(X= rp_peak_data,
                                    obs = pd.DataFrame(index=peak_info_data["peak"]),
                                    var = pd.DataFrame(index=peak_info_data["peak"]))
    
    for i in range(len(peak_info_data)):
        peak_info_data.iloc[i,5] = peak_info_data.iloc[i,5].decode('utf-8')

    id_peak_df = peak_info_data[['peak', 'peak_id']]
    id_peak_df.columns = ["peak", "id"]
    id_peak_df.set_index("peak", inplace=True)

    # gene to id

    id_gene_df = gene_info_Data[['gene symbol','gene_id']]
    id_gene_df.columns = ['gene', 'id']
    id_gene_df.set_index("gene", inplace=True)

    id_gene_df = id_gene_df[~id_gene_df.index.duplicated(keep="first")]
    id_peak_df = id_peak_df[~id_peak_df.index.duplicated(keep="first")]

    # breakpoint()

    return id_gene_df, id_peak_df, adata_rp_gene_peak, adata_rp_peak

############################################################

def genescore(fileformat, directory, outprefix, peakcount, feature_file, barcode_file, genedistance, species, model = "Enhanced"):

    annotation_path = "../annotations"

    # resource_filename('MAESTRO', 'annotations')

    genebed = os.path.join(annotation_path, species + "_refgenes.txt")
    
    decay = float(genedistance)
    score_file = os.path.join(directory, outprefix + "_gene_score.h5")
    
    matrix_dict = read_peak_data(matrix_file = peakcount, feature_file = feature_file, 
                                 barcode_file = barcode_file, datatype = "Peak")
    peakmatrix = matrix_dict["matrix"]
    features = matrix_dict["features"]
    features = [f.encode() for f in features]
    barcodes = matrix_dict["barcodes"]

    if os.path.exists("RP_data2.pkl"):
        with open("RP_data2.pkl", "rb") as f:
            RP_data = pickle.load(f)
            rp_gene_peak_data = RP_data["genes_peaks_RP"]
            rp_peak_data = RP_data["peak_score_RP"]
            peak_info = RP_data["peaks_info"]
            genes_info_full = RP_data["genes_info"]
    else:
        rp_gene_peak_data, rp_peak_data, peak_info,genes_info_full = calculate_RP_score(peakmatrix, features,
                                                                barcodes, genebed, decay, score_file, model, split_str = "-")
    # peak to id
    id_peak_data = {}
    for elem in peak_info:
        tmp_id = elem[-1]
        tmp_peak = elem[-2].decode('utf-8')
        id_peak_data[tmp_id] = tmp_peak
        
    id_peak_df = pd.DataFrame.from_dict(id_peak_data, orient='index') #, columns=['value']
    id_peak_df["id"] = id_peak_df.index
    id_peak_df.columns = ["peak","id"]
    id_peak_df.set_index("peak", inplace=True)
    
    
    # gene to id
    id_gene_data = {}
    for elem in genes_info_full:
        tmp_id = elem[-1]
        tmp_gene = elem[-2].split("@")[0]
        id_gene_data[tmp_id] = tmp_gene
    
    id_gene_df = pd.DataFrame.from_dict(id_gene_data, orient='index', columns=['value'])
    id_gene_df["id"] = id_gene_df.index
    id_gene_df.columns = ['gene', 'id']
    id_gene_df.set_index("gene", inplace=True)

    id_gene_df = id_gene_df[~id_gene_df.index.duplicated(keep = "first")]
    id_peak_df = id_peak_df[~id_peak_df.index.duplicated(keep = "first")]

    return rp_gene_peak_data, rp_peak_data, id_gene_df, id_peak_df

# chrom, start, end, tss, promoter, exons, exon_length, flag, uid, index
# chrom, center, start, end, flag=0, uid, index