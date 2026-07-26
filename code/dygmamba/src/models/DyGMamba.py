import numpy as np
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

from models.modules import TimeEncoder
from utils.DataLoader import get_node_feature
from utils.utils import NeighborSampler
from mamba_ssm import Mamba

from typing import Dict, Tuple, Union

import math

def check_nan_by_type(data):
    # 1. 判断是否为 PyTorch 张量
    if torch.is_tensor(data):
        # 使用 torch.isnan()，返回一个布尔张量，.any() 检查是否有任何一个 True
        has_nan = torch.isnan(data).any().item()
        if has_nan:
            print("Tensor 中存在 NaN！建议处理方案：使用 .nan_to_num(0) 填充")
        else:
            print("Tensor 中没有 NaN。")
        return has_nan

    # 2. 判断是否为 List
    elif isinstance(data, list):
        # 列表需要遍历检查，注意 NaN 不等于自身的特性
        # 或者使用 math.isnan()，但需确保元素是数值类型
        has_nan = any(
            isinstance(x, float) and math.isnan(x) for x in data
        )
        if has_nan:
            print("List 中存在 NaN！")
        else:
            print("List 中没有 NaN。")
        return has_nan

    else:
        print(f"跳过处理：变量类型为 {type(data)}，暂未定义检查逻辑")
        return False



class DyGMamba(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                        time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, 
                        num_layers: int = 2, num_heads: int = 2,
                        dropout: float = 0.1,gamma: float = 0.5,
                        max_input_sequence_length: int = 512, 
                        max_interaction_times: int = 10, device: str = 'gpu'):
        """
        DyGMamba model.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param gamma: float, gamma
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param max_interaction_times: int, maximal interactions for src and dst to consider
        :param device: str, device
        """
        super(DyGMamba, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.gamma = gamma
        self.max_input_sequence_length = max_input_sequence_length
        self.max_interaction_times = max_interaction_times
        self.device = device


        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim

        self.neighbor_co_occurrence_encoder = NIFEncoder(nif_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, 
                                out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, 
                                out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, 
                                out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, 
                                                out_features=self.channel_embedding_dim, bias=True)
        })


        self.num_channels = 4

        feature_expansion_size = 2

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim // feature_expansion_size, 
                                        out_features=self.node_feat_dim, bias=True)

        self.output_layer_t_diff = nn.Linear(in_features=int(self.gamma*self.channel_embedding_dim),
                                            out_features=self.node_feat_dim, bias=True)

        self.mamba = nn.ModuleList([
            Mamba(d_model=self.num_channels * self.channel_embedding_dim // feature_expansion_size,  # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,  # Local convolution width
                expand=1,  # Block expansion factor
                )
            for _ in range(self.num_layers)
        ])

        self.mamba_t_diff = nn.ModuleList([
            Mamba(d_model=int(self.gamma*self.channel_embedding_dim),  # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,  # Local convolution width
                expand=1,  # Block expansion factor
                )
            for _ in range(self.num_layers)
        ])

        self.projection_layer_t_diff = nn.Linear(in_features=self.time_feat_dim, 
                                out_features=int(self.gamma*self.channel_embedding_dim), bias=True)

        self.projection_layer_t_diff_up = nn.Linear(in_features=int(self.gamma*self.channel_embedding_dim),
                                out_features=self.num_channels * self.channel_embedding_dim // feature_expansion_size, bias=True)


        self.weightagg = nn.Linear(self.num_channels * self.channel_embedding_dim // feature_expansion_size, 1)

        self.reduce_layer = nn.Linear(self.num_channels * self.channel_embedding_dim, 
                                    self.num_channels * self.channel_embedding_dim // feature_expansion_size)

        self.channel_norm = nn.LayerNorm(self.num_channels * self.channel_embedding_dim // feature_expansion_size)

        self.channel_feedforward = FeedForwardNet(input_dim=self.num_channels * self.channel_embedding_dim // feature_expansion_size,
                                                dim_expansion_factor=4,
                                                dropout=self.dropout)

        self.neighbor_selection_layer = nn.Linear(self.num_channels * self.channel_embedding_dim // feature_expansion_size, 
                                        self.num_channels * self.channel_embedding_dim // feature_expansion_size)


    def compute_src_dst_node_temporal_embeddings(self, 
                                                src_node_ids: np.ndarray, 
                                                dst_node_ids: np.ndarray, 
                                                node_interact_times: np.ndarray,
                                                node_features: torch.Tensor, 
                                                edge_features: torch.Tensor,
                                                edge_ids: np.ndarray = None):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param node_features: Tensor, dynamic node features with shape (num_nodes, node_feat_dim) or dict for (node_id, timestamp) indexing
        :param edge_features: Tensor, dynamic edge features with shape (num_edges, edge_feat_dim)
        :param edge_ids: ndarray, shape (batch_size, ), 当前批次的边id，用于空邻居时填充正确的边特征
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, 
                                                            node_interact_times=node_interact_times,
                                                            interact_dst_ids=dst_node_ids,
                                                            interact_edge_ids=edge_ids)

        # print(src_nodes_neighbor_times_list)
        # print(src_nodes_neighbor_times_list)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, 
                                                            node_interact_times=node_interact_times,
                                                            interact_dst_ids=src_node_ids,
                                                            interact_edge_ids=edge_ids)

        # print(dst_nodes_neighbor_times_list)
        # print(dst_nodes_neighbor_times_list)
        # sys.exit("Ending the program here.")

        padded_time_diff_emb = self.time_modeling(src_node_ids, dst_node_ids, node_interact_times,
                                                    src_nodes_neighbor_ids_list, src_nodes_neighbor_times_list, 
                                                    self.time_encoder)

        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_padded_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, 
                            node_interact_times=node_interact_times, 
                            nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                            nodes_edge_ids_list=src_nodes_edge_ids_list, 
                            nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                            patch_size=self.patch_size, 
                            max_input_sequence_length=self.max_input_sequence_length,
                            current_interact_node_ids=dst_node_ids,
                            current_interact_edge_ids=edge_ids)

        # dst_padded_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, 
                            node_interact_times=node_interact_times, 
                            nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                            nodes_edge_ids_list=dst_nodes_edge_ids_list, 
                            nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                            patch_size=self.patch_size, 
                            max_input_sequence_length=self.max_input_sequence_length,
                            current_interact_node_ids=src_node_ids,
                            current_interact_edge_ids=edge_ids)

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_node_ids=src_node_ids, 
                                                dst_node_ids=dst_node_ids, 
                                                src_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # get the features of the sequence of source and destination nodes
        # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_padded_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)

        # breakpoint()

        # print(f"Min edge ID: {np.min(src_padded_nodes_edge_ids)}")
        # print(f"Max edge ID: {np.max(src_padded_nodes_edge_ids)}")
        # print(f"Shape of edge_features: {edge_features.shape}")

        # breakpoint()
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, 
                            padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                            padded_nodes_edge_ids=src_padded_nodes_edge_ids, 
                            padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, 
                            time_encoder=self.time_encoder,
                            node_features=node_features, edge_features=edge_features)

        # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_padded_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, 
                                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                            padded_nodes_edge_ids=dst_padded_nodes_edge_ids, 
                            padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, 
                            time_encoder=self.time_encoder,
                            node_features=node_features, edge_features=edge_features)



        # get the patches for source and destination nodes
        # src_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
        # src_patches_nodes_edge_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * edge_feat_dim)
        # src_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                            padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                            padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                            padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                            patch_size=self.patch_size)

        # dst_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
        # dst_patches_nodes_edge_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * edge_feat_dim)
        # dst_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                            padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                            padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                            padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                            patch_size=self.patch_size)


        # align the patch encoding dimension
        # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)


        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        src_patches_data = [src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features,
                            src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features]
        dst_patches_data = [dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features,
                            dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features]
        
        # #********************************************************************************
        # print("*"*50)
        # print("check nan for dst_patches_data before stacking:")
        # check_nan_by_type(src_patches_data)
        # print(f"neighbor feature {len(src_patches_nodes_neighbor_node_raw_features)}, \
        #     edge feature {len(src_patches_nodes_edge_raw_features)}, \
        #     time feature {len(src_patches_nodes_neighbor_time_features)}, \
        #     co-occurrence feature {len(src_patches_nodes_neighbor_co_occurrence_features)}")
        
        
        # print("*"*50)
        # print("check nan for padded time before stacking:")
        # check_nan_by_type(padded_time_diff_emb)

        # print(f"src node {len(src_node_ids)}, \
        #     dst node {len(dst_node_ids)}, \
        #     node interact times {len(node_interact_times)}, \
        #     src neighbor ids list {len(src_nodes_neighbor_ids_list)}, \
        #     src neighbor times list {len(src_nodes_neighbor_times_list)}")
        
        # print("*"*50)
        # print(src_node_ids)
        # print("*"*50)
        # print(dst_node_ids)
        # print("*"*50)
        # print(node_interact_times)
        # print("*"*50)
        # print(src_nodes_neighbor_ids_list)
        # print("*"*50)
        # print(src_nodes_neighbor_times_list)
        
        
        # print("*"*50)
        # print("check nan for src_patches_data before stacking:")
        # check_nan_by_type(dst_patches_data)
        
        # print(f"neighbor feature {len(dst_patches_nodes_neighbor_node_raw_features)}, \
        #     edge feature {len(dst_patches_nodes_edge_raw_features)}, \
        #     time feature {len(dst_patches_nodes_neighbor_time_features)}, \
        #     co-occurrence feature {len(dst_patches_nodes_neighbor_co_occurrence_features)}")
        
        # sys.exit("Ending the program here.")
        
        # #********************************************************************************
        
        src_patches_data = torch.stack(src_patches_data, dim=2)
        dst_patches_data = torch.stack(dst_patches_data, dim=2)
        src_patches_data = src_patches_data.reshape(batch_size, src_num_patches, 
                                                self.num_channels * self.channel_embedding_dim)
        dst_patches_data = dst_patches_data.reshape(batch_size, dst_num_patches,
                                                self.num_channels * self.channel_embedding_dim)

        # reduce to channel embsize
        src_patches_data = self.reduce_layer(src_patches_data)
        dst_patches_data = self.reduce_layer(dst_patches_data)


        for mamba in self.mamba:
            src_patches_data = mamba(src_patches_data) + src_patches_data
            dst_patches_data = mamba(dst_patches_data) + dst_patches_data
            src_patches_data = self.channel_norm(src_patches_data)
            dst_patches_data = self.channel_norm(dst_patches_data)
            src_patches_data = self.channel_feedforward(src_patches_data) + src_patches_data
            dst_patches_data = self.channel_feedforward(dst_patches_data) + dst_patches_data

        padded_time_diff_emb = self.projection_layer_t_diff(padded_time_diff_emb)
        for mamba_t in self.mamba_t_diff:
            padded_time_diff_emb = mamba_t(padded_time_diff_emb) + padded_time_diff_emb

        src_weight = self.weightagg(src_patches_data).transpose(1, 2)
        dst_weight = self.weightagg(dst_patches_data).transpose(1, 2)

        src_patches_data_ = src_weight.matmul(src_patches_data).squeeze(dim=1)
        dst_patches_data_ = dst_weight.matmul(dst_patches_data).squeeze(dim=1)

        time_diff_emb = torch.mean(padded_time_diff_emb, dim=1)
        time_diff_emb_ = self.projection_layer_t_diff_up(time_diff_emb)

        # Tensor, shape (batch_size, 1, channel_embedding_dim)
        src_selection_param = (self.neighbor_selection_layer(dst_patches_data_) * time_diff_emb_).unsqueeze(1)
        dst_selection_param = (self.neighbor_selection_layer(src_patches_data_) * time_diff_emb_).unsqueeze(1)


        src_patches_data = torch.sum(src_patches_data * torch.nn.functional.softmax(torch.sum(src_selection_param * src_patches_data, dim=2), dim=1).unsqueeze(2), dim=1)
        dst_patches_data = torch.sum(dst_patches_data * torch.nn.functional.softmax(torch.sum(dst_selection_param * dst_patches_data, dim=2), dim=1).unsqueeze(2), dim=1)


        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_patches_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        time_diff_emb = self.output_layer_t_diff(time_diff_emb)
        return src_node_embeddings, dst_node_embeddings, time_diff_emb
    
    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                    nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256,
                    current_interact_node_ids: np.ndarray = None,
                    current_interact_edge_ids: np.ndarray = None):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :param current_interact_node_ids: ndarray, shape (batch_size, ), 当前时刻与 node_ids[i] 交互的对方节点id
                                        当某节点历史邻居为空时，用此节点作为当前时刻邻居填充，避免全零导致NaN
        :param current_interact_edge_ids: ndarray, shape (batch_size, ), 当前时刻交互对应的边id
                                        与 current_interact_node_ids 配套，填充对应的真实边特征
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'

        max_seq_length = max_input_sequence_length
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]


        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float64)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, -1] = node_ids[idx]
            padded_nodes_edge_ids[idx, -1] = 0
            padded_nodes_neighbor_times[idx, -1] = node_interact_times[idx]


            if len(nodes_neighbor_ids_list[idx]) > 0:
                # left padding
                padded_nodes_neighbor_ids[idx, -len(nodes_neighbor_ids_list[idx])-1:-1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, -len(nodes_edge_ids_list[idx])-1:-1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, -len(nodes_neighbor_times_list[idx])-1:-1] = nodes_neighbor_times_list[idx]
                
            else:
                # 【修复】邻居为空（首次出现的节点）
                # 优先用当前时刻的交互对方节点（更有信息量），否则退回到自身节点
                if current_interact_node_ids is not None:
                    # 兼容 torch.Tensor（GPU）和 np.ndarray
                    cid = current_interact_node_ids[idx]
                    fill_node_id = int(cid.cpu().item() if hasattr(cid, 'cpu') else cid)
                else:
                    fill_node_id = int(node_ids[idx])

                # 用当前交互的真实边id填充；若未提供则退回0（padding edge）
                if current_interact_edge_ids is not None:
                    eid = current_interact_edge_ids[idx]
                    fill_edge_id = int(eid.cpu().item() if hasattr(eid, 'cpu') else eid)
                else:
                    fill_edge_id = 0

                padded_nodes_neighbor_ids[idx, :] = fill_node_id
                padded_nodes_edge_ids[idx, :] = fill_edge_id
                padded_nodes_neighbor_times[idx, :] = node_interact_times[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def find_previous_interaction(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_node_interact_times: np.ndarray, dst_node_interact_times: np.ndarray,
                                src_nodes_neighbor_ids_list: list, dst_nodes_neighbor_ids_list: list, src_nodes_edge_ids_list: list, dst_nodes_edge_ids_list: list,
                                src_nodes_neighbor_times_list: list, dst_nodes_neighbor_times_list: list):
        src_latest_time_interaction, dst_latest_time_interaction = [], []
        gamma = 100
        shrink_ratio = 1e8
        shrink_coeff = torch.tensor(1/(gamma * shrink_ratio), device=self.device)

        for idx in range(len(src_node_ids)):

            find_interact = np.where(src_nodes_neighbor_ids_list[idx] == dst_node_ids[idx], src_nodes_neighbor_ids_list[idx], 0)

            find_interact_index = np.nonzero(find_interact)

            if find_interact_index[0].shape[0] == 0: # previous interaction not found
                src_latest_time_interaction.append(0.0)
            else:
                time_gap = float(src_node_interact_times[idx]) - float(src_nodes_neighbor_times_list[idx][find_interact_index[0][-1]])
                src_latest_time_interaction.append(max(0.0, time_gap))

        pair_latest_time_interaction = torch.exp(-torch.from_numpy(np.array(src_latest_time_interaction)).to(self.device) * shrink_coeff)

        return pair_latest_time_interaction

    def time_modeling(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_node_interact_times: np.ndarray,
                    src_nodes_neighbor_ids_list: list, src_nodes_neighbor_times_list: list, time_encoder):

        max_interaction_times = self.max_interaction_times

        # padded_time = np.ones((len(src_node_ids), max_interaction_times)).astype(np.longlong) * 1e10
        padded_time = np.zeros((len(src_node_ids), max_interaction_times)).astype(np.float32)
        
        for idx in range(len(src_node_ids)):
            find_interact = np.where(src_nodes_neighbor_ids_list[idx] == dst_node_ids[idx], src_nodes_neighbor_ids_list[idx], 0)
            find_interact_index = np.nonzero(find_interact)
            if find_interact_index[0].shape[0] == 0: # previous interaction not found
                continue
            else:
                unique_ts = np.unique(src_nodes_neighbor_times_list[idx][find_interact_index[0]])
                find_idx_back = np.concatenate((unique_ts, [src_node_interact_times[idx].item()]))
                find_idx_front = np.concatenate(([0.0], unique_ts))
                time_diff = find_idx_back - find_idx_front
                if time_diff.shape[0] - 1 < max_interaction_times:
                    padded_time[idx][-time_diff.shape[0]+1:] = time_diff[1:]
                else:
                    padded_time[idx][:] = time_diff[-max_interaction_times:]


        padded_time_diff_emb = time_encoder(timestamps=torch.from_numpy(padded_time).float().to(self.device))

        return padded_time_diff_emb

    def get_batch_node_features(self, node_features: Union[torch.Tensor, Dict],
                        node_ids: np.ndarray, timestamps: np.ndarray = None, node_dim = 172) -> torch.Tensor:
        """
        获取节点特征，支持二维元组索引 (node_id, timestamp)

        Args:
            node_features: 节点特征数据
                - 如果是Dict: {(node_id, timestamp): feature_vector}
                - 如果是Tensor: shape (num_nodes, node_feat_dim) 传统方式
            node_ids: 节点ID数组
            timestamps: 时间戳数组（用于二维索引）

        Returns:
            torch.Tensor: shape (len(node_ids), max_node_feat_dim)
        """
        batch_size = node_ids.shape[0]
        max_sequence_length = node_ids.shape[1]

        if isinstance(node_features, dict):
            
            node_ids_np = node_ids.cpu().numpy() if isinstance(node_ids, torch.Tensor) else np.asarray(node_ids)
            ts_np = timestamps.cpu().numpy() if isinstance(timestamps, torch.Tensor) else np.asarray(timestamps)

            # 展平为一维
            flat_node_ids = node_ids_np.reshape(-1)   # (batch_size * seq_len,)
            flat_timestamps = ts_np.reshape(-1)        # (batch_size * seq_len,)

            zero_feat = np.zeros(node_dim, dtype=np.float32)

            flat_features = np.stack([
                np.array(node_features.get((int(n), float(t)), zero_feat), dtype=np.float32).reshape(-1)
                for n, t in zip(flat_node_ids, flat_timestamps)
            ], axis=0)  # shape: (batch_size * seq_len, node_dim)

            # 一次性转为tensor并reshape
            features_tensor = torch.from_numpy(flat_features).to(
                dtype=torch.float32, device=self.device
            ).reshape(batch_size, max_sequence_length, node_dim)
            
            if torch.isnan(features_tensor).any():
                print("Warning: NaN values found in features_tensor.")
                sys.exit(1)

            # for i in range(batch_size):
            #     for j in range(max_sequence_length):
            #         t = timestamps[i, j]
            #         n = node_ids[i, j]
            #         key = (t, n)
            #         temp_feature = node_features.get(key, np.zeros(node_dim))
            #         features[i,j]= torch.tensor(temp_feature, dtype=features.dtype, device=features.device)
            
            # features_tensor = features.clone().detach().to(dtype=torch.float32, device=self.device)
            # # features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

            return features_tensor
        else:
            print("node type wrong")
            return


    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                    padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder,
                    node_features: torch.Tensor, edge_features: torch.Tensor):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = edge_features[torch.from_numpy(padded_nodes_edge_ids)].to(self.device)

        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.get_batch_node_features(node_features,
                                                                padded_nodes_neighbor_ids,
                                                                padded_nodes_neighbor_times)


        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        # 【修复】clamp 确保时差非负，防止极端值在 float16 下溢出
        time_diffs = np.clip(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times, a_min=0.0, a_max=None)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(time_diffs).float().to(self.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):
        """
        get the sequence of patches for nodes
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, \
        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features = [], [], [], []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_edge_raw_features.append(padded_nodes_edge_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_co_occurrence_features.append(padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)

        patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class NIFEncoder(nn.Module):

    def __init__(self, nif_feat_dim: int, device: str = 'cpu'):

        super(NIFEncoder, self).__init__()

        self.nif_feat_dim = nif_feat_dim
        self.device = device

        self.nif_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.nif_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.nif_feat_dim, out_features=self.nif_feat_dim))

    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):

        # two lists to store the appearances of source and destination nodes
        src_nodes_appearances, dst_nodes_appearances = [], []
        # src_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for i in range(len(src_node_ids)):
            src_node_id = src_node_ids[i]
            dst_node_id = dst_node_ids[i]
            src_node_neighbor_ids = src_nodes_neighbor_ids[i]
            dst_node_neighbor_ids = dst_nodes_neighbor_ids[i]

            # Calculate unique keys and counts for source and destination
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_node_neighbor_ids, return_inverse=True,
                                                                         return_counts=True)
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_node_neighbor_ids, return_inverse=True,
                                                                         return_counts=True)

            # Create mappings from node IDs to their counts
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # Adjust counts specifically for the cases where src_node_id appears in dst's neighbors and vice versa
            if src_node_id in dst_mapping_dict:
                src_count_in_dst = dst_mapping_dict[src_node_id]
                src_mapping_dict[src_node_id] = src_count_in_dst
                dst_mapping_dict[src_node_id] = src_count_in_dst
            if dst_node_id in src_mapping_dict:
                dst_count_in_src = src_mapping_dict[dst_node_id]
                src_mapping_dict[dst_node_id] = dst_count_in_src
                dst_mapping_dict[dst_node_id] = dst_count_in_src

            # Calculate appearances in each other's lists
            src_node_neighbor_counts_in_dst = torch.tensor(
                [dst_mapping_dict.get(neighbor_id, 0) for neighbor_id in src_node_neighbor_ids]).float().to(self.device)
            dst_node_neighbor_counts_in_src = torch.tensor(
                [src_mapping_dict.get(neighbor_id, 0) for neighbor_id in dst_node_neighbor_ids]).float().to(self.device)

            # Stack counts to get a two-column tensor for each node list
            src_nodes_appearances.append(torch.stack(
                [torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device),
                 src_node_neighbor_counts_in_dst], dim=1))
            dst_nodes_appearances.append(torch.stack([dst_node_neighbor_counts_in_src,
                                                      torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(
                                                          self.device)], dim=1))

        # Stack to form batch tensors
        src_nodes_appearances = torch.stack(src_nodes_appearances, dim=0)
        dst_nodes_appearances = torch.stack(dst_nodes_appearances, dim=0)

        return src_nodes_appearances, dst_nodes_appearances

    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray,
                dst_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_nodes_neighbor_ids and dst_nodes_neighbor_ids
        :param src_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_nodes_appearances, dst_nodes_appearances = self.count_nodes_appearances(src_node_ids=src_node_ids,
                                                                                    dst_node_ids=dst_node_ids,
                                                                                    src_nodes_neighbor_ids=src_nodes_neighbor_ids,
                                                                                    dst_nodes_neighbor_ids=dst_nodes_neighbor_ids)

        # Tensor, shape (batch_size, src_max_seq_length, nif_feat_dim)
        # Tensor, shape (batch_size, dst_max_seq_length, nif_feat_dim)

        src_nodes_nif_features = (src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features = (dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        src_nodes_nif_features = self.nif_encode_layer(src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features = self.nif_encode_layer(dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        return src_nodes_nif_features, dst_nodes_nif_features


class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)
