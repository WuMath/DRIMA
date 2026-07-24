import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json


from utils.metrics import get_link_prediction_metrics
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data


def model_link_prediction(model_name: str, 
                            model: nn.Module,
                            neighbor_sampler: NeighborSampler,
                            evaluate_idx_data_loader: DataLoader,
                            evaluate_data: Data,
                            node_features,
                            edge_features):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # evaluation phase use all the graph information
    model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        result_list = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)

        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):

            evaluate_data_indices = evaluate_data_indices.numpy()

            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]
                

            # get temporal embedding of source , destination nodes and time difference
            # three Tensors, with shape (batch_size, node_feat_dim)
            batch_src_node_embeddings, batch_dst_node_embeddings, batch_time_diff_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                    dst_node_ids=batch_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                  node_features=node_features,
                                                                  edge_features=edge_features)
            
            positive_probabilities = model[1](input_1=batch_src_node_embeddings,
                                              input_2=batch_dst_node_embeddings,
                                              input_3=batch_time_diff_embeddings).squeeze(dim=-1).sigmoid()
            # predicts = positive_probabilities
            # labels = edge_labels[batch_edge_ids].float()

            # loss = loss_func(input=predicts, target=labels)

            result_list.append(positive_probabilities)
            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch')
        
        result = torch.cat(result_list, dim=0)

    return result

