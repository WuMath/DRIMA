import pandas as pd


def create_cre_bed(peaks, data_path):

    split_peaks = peaks.str.split('-', expand=True)
    split_peaks.columns = ['chr', 'start', 'end']

    # BED need int type
    split_peaks['start'] = split_peaks['start'].astype(int)
    split_peaks['end'] = split_peaks['end'].astype(int)

    # remove prefix
    split_peaks['chr'] = split_peaks['chr'].str.replace('chr', '', regex=False)

    # create BED file（No header）
    split_peaks.to_csv(data_path + 'cre.bed', sep='\t', index=False, header=False)

    return split_peaks


def Extract_tf_cre_gene_info(Node_id, Graph_df, data_path):

    ##################################
    # get CRE-gene and CRE-CRE
    id_to_symbol = dict(zip(Node_id.index+1, Node_id['name']))

    graph_edge = Graph_df[Graph_df["label"]==1].copy()

    graph_edge = graph_edge[["source_node", "target_node", "label", "time"]]
    
    # replace source and target as symbol
    
    graph_edge['source_symbol'] = graph_edge['source_node'].map(id_to_symbol)
    
    graph_edge['target_symbol'] = graph_edge['target_node'].map(id_to_symbol)

    cre_gene = graph_edge[['source_symbol', 'target_symbol']]
    
    cre_gene.columns = ["cre", "gene"] 

    ##################################
    # FIMO output（TF-CRE binding site）
    fimo_df = pd.read_csv(data_path + 'fimo_output/fimo.tsv', sep='\t')
    
    fimo_df = fimo_df[fimo_df['p-value'] < 1e-5]  # get high confidence degree

    # only save necessary column
    fimo_df = fimo_df[['motif_alt_id', 'sequence_name']]
    
    fimo_df.columns = ['TF', 'CRE_id']

    # drop duplicated
    tf_cre_df = fimo_df.drop_duplicates()

    tf_cre_df["cre"] = tf_cre_df["CRE_id"].str.extract(r'([^:]+):(\d+)-(\d+)') \
                                .apply(lambda x: f"chr{x[0]}-{x[1]}-{x[2]}", axis=1)

    tf_cre = tf_cre_df[["TF", "cre"]]
    
    tf_cre.columns = ["tf", "cre"]

    return cre_gene, tf_cre