import anndata as ad
import scanpy as sc
import pandas as pd
import os
from LingerGRN.pseudo_bulk import *
import LingerGRN.LINGER_tr as LINGER_tr
import LingerGRN.LL_net as LL_net
from LingerGRN.preprocess import *

if __name__ == "__main__":
    
    #########################################################################################
    # Configuration
    #########################################################################################
    
    cell_type = "HELA"
    
    data_path = "/home/wuyan/dygmamba_project/data/cell_line/" + cell_type + "/process/"
    
    output_path = "/home/wuyan/dygmamba_project/data/cell_line/" + cell_type + "/data_linger/"
    os.makedirs(output_path, exist_ok=True)
    
    GRNdir='/home/wuyan/dygmamba_project/data/Data_LINGER/data_bulk/'
    
    outdir = output_path + "train/"
    os.makedirs(outdir, exist_ok=True)
    
    Datadir=output_path
    method='LINGER'
    genome='hg38'
    activef='ReLU' # active function chose from 'ReLU','sigmoid','tanh'

    #########################################################################################
    # Data read
    #########################################################################################
    adata_RNA = ad.read_h5ad(data_path + "rna_processed.h5ad")
    adata_ATAC = ad.read_h5ad(data_path + "atac_processed.h5ad")

    def fix_peak_format(name):
        if ':' not in name and '-' in name:
            return name.replace('-', ':', 1)
        return name

    adata_ATAC.var_names = [fix_peak_format(name) for name in adata_ATAC.var_names]
    
    #########################################################################################
    # Data process
    #########################################################################################
    sc.pp.filter_cells(adata_RNA, min_genes=200)
    sc.pp.filter_genes(adata_RNA, min_cells=3)
    sc.pp.filter_cells(adata_ATAC, min_genes=200)
    sc.pp.filter_genes(adata_ATAC, min_cells=3)

    selected_cell=list(set(adata_RNA.obs.index)&set(adata_ATAC.obs.index))

    adata_RNA = adata_RNA[selected_cell,]
    adata_ATAC = adata_ATAC[selected_cell,]


    adata_RNA.obs['barcode'] = adata_RNA.obs_names
    adata_ATAC.obs['barcode'] = adata_ATAC.obs_names

    adata_RNA.obs['sample'] = 'sample_1'
    adata_ATAC.obs['sample'] = 'sample_1'
    adata_RNA.obs['label'] = '0' 
    adata_ATAC.obs['label'] = '0'
    adata_ATAC.var['gene_ids'] = adata_ATAC.var_names
    adata_RNA.var['gene_ids'] = adata_RNA.var_names
    
    
    #########################################################################################
    # Generate pseudo-bulk data
    #########################################################################################
    samplelist=list(set(adata_ATAC.obs['sample'].values)) # sample is generated from cell barcode 
    tempsample=samplelist[0]

    TG_pseudobulk=pd.DataFrame([])
    RE_pseudobulk=pd.DataFrame([])
    singlepseudobulk = (adata_RNA.obs['sample'].unique().shape[0]*adata_RNA.obs['sample'].unique().shape[0]>100)
    for tempsample in samplelist:
        adata_RNAtemp=adata_RNA[adata_RNA.obs['sample']==tempsample]
        adata_ATACtemp=adata_ATAC[adata_ATAC.obs['sample']==tempsample]
        TG_pseudobulk_temp,RE_pseudobulk_temp=pseudo_bulk(adata_RNAtemp,adata_ATACtemp,singlepseudobulk)                
        TG_pseudobulk=pd.concat([TG_pseudobulk, TG_pseudobulk_temp], axis=1)
        RE_pseudobulk=pd.concat([RE_pseudobulk, RE_pseudobulk_temp], axis=1)
        RE_pseudobulk[RE_pseudobulk > 100] = 100

    adata_ATAC.write(output_path + 'adata_ATAC.h5ad')
    adata_RNA.write(output_path + 'adata_RNA.h5ad')
    TG_pseudobulk=TG_pseudobulk.fillna(0)
    RE_pseudobulk=RE_pseudobulk.fillna(0)
    pd.DataFrame(adata_ATAC.var.index).to_csv(output_path + 'Peaks.txt',header=None,index=None)
    TG_pseudobulk.to_csv(output_path + 'TG_pseudobulk.tsv')
    RE_pseudobulk.to_csv(output_path + 'RE_pseudobulk.tsv')
    
    #########################################################################################
    # Data process
    #########################################################################################

    preprocess(TG_pseudobulk,RE_pseudobulk,GRNdir,genome,method,output_path, outdir)

    LINGER_tr.training(GRNdir,method,outdir,activef,'Human')

    LL_net.TF_RE_binding(GRNdir, adata_RNA, adata_ATAC, genome, method, Datadir, outdir)
    
    LL_net.cis_reg(GRNdir, adata_RNA, adata_ATAC, genome, method, output_path, outdir)
    
    LL_net.trans_reg(GRNdir,method,output_path, outdir,genome)
    
    