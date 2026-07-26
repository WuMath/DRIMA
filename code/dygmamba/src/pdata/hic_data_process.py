import os
import gzip
import collections
import re
import pyfaidx
import time

import itertools
import glob
import pandas as pd 
import numpy as np
import anndata as ad
from tqdm import tqdm
from scipy.sparse import csr_matrix

from typing import Dict, List, NamedTuple

from gtfparse import read_gtf
import ast

import hicstraw

#################################################################
#****************************************************************
#*******  Read Hi-C data from folder  *************
#****************************************************************
#################################################################

def get_interaction_matrix(
        hic_file_path: str,
        chr1_name: str,
        chr2_name: str,
        resolution: int,
        normalization: str = 'KR',
        min_score: float = 10.0
) -> pd.DataFrame:
    
    if chr1_name != chr2_name:
        return pd.DataFrame()
    
    print(f"-> Querying: {chr1_name} vs {chr2_name}...")
    
    try:
        records = hicstraw.straw(
            'observed', normalization, hic_file_path,
            chr1_name, chr2_name, 'BP', resolution
        )
        
        interaction_list = []
        
        for r in records:
            if r.counts > min_score:  # <--- 核心过滤
                interaction_list.append({
                    'chr1': chr1_name, 'start1': r.binX, 'end1': r.binX + resolution,
                    'chr2': chr2_name, 'start2': r.binY, 'end2': r.binY + resolution,
                    'contact_count': r.counts
                })

        if not interaction_list:
            print(f"...There are no interaction between {chr1_name} vs {chr2_name} !")
            return pd.DataFrame()

        df = pd.DataFrame(interaction_list)
        
        print(f"...successfully find {len(df)} interaction records!")
        
        return df
    
    except RuntimeError as e:
        print(f"     ...error: {e}, skip the combination")
        return pd.DataFrame()
    except Exception as e:
        print(f"     ...file read error: {e}")
        return pd.DataFrame()




def read_hic_data(
        hic_file_path: str,
        resolution: int,
        normalization: str = 'SCALE',
        min_score: float = 10.0
) -> pd.DataFrame:
    """
    Args:
        hic_file_path (str): .the file path of the hic
        resolution (int): the resolution Hi-C
        normalization (str, optional): normalization method, Defaults to 'VC'.

    Returns:
        pd.DataFrame: contain all the interaction
    """
    
    print(f"--- start read all the data of '{hic_file_path}' ---")
    print(f"--- warning: this will need large memory and time! ---")

    try:
        hic = hicstraw.HiCFile(hic_file_path)
        chrom_names = [c.name for c in hic.getChromosomes() if 'all' not in c.name.lower()]
        
        print(f"The file contain {len(chrom_names)} chromosomes")
    
    except Exception as e:
        
        print(f"Error: can't open .hic file or get the chr list: {e}")
        return pd.DataFrame()
    
    all_dfs = []
    for chrom in chrom_names:
        pair_df = get_interaction_matrix(
            hic_file_path, chrom, chrom, resolution, normalization, min_score
        )
        if not pair_df.empty:
            all_dfs.append(pair_df)

    if not all_dfs:
        
        print("\ncan't read any data!")
        return pd.DataFrame()

    print("\n--- Read all data and Create the final Dataframe... ---")

    full_df = pd.concat(all_dfs, ignore_index=True)
    print("Successfully create DataFrame!")

    return full_df
