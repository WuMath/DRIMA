from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
from collections import Counter

import numpy as np
import pandas as pd


def _format_feature(feat, feat_dim):
    """
    将读取到的特征统一整理成 float32 的 1D 向量，长度固定为 feat_dim
    """
    if feat is None:
        return np.zeros(feat_dim, dtype=np.float32)

    feat = np.asarray(feat, dtype=np.float32).reshape(-1)

    if feat.shape[0] == feat_dim:
        return feat

    out = np.zeros(feat_dim, dtype=np.float32)
    n = min(feat.shape[0], feat_dim)
    out[:n] = feat[:n]
    return out


def _lookup_node_feature(node_raw_features, node_id, ts, feat_dim):
    """
    兼容多种 NodeFeatureLookup 接口写法：
    1) lookup.get((node, ts))
    2) lookup.get(node, ts)
    3) lookup[(node, ts)]
    4) lookup[node, ts]
    5) lookup.query(node, ts)

    若都失败，返回全零向量
    """
    feat = None

    # 方案1: get((node, ts))
    if feat is None and hasattr(node_raw_features, "get"):
        try:
            feat = node_raw_features.get((node_id, ts))
        except Exception:
            pass

    # 方案2: get(node, ts)
    if feat is None and hasattr(node_raw_features, "get"):
        try:
            feat = node_raw_features.get(node_id, ts)
        except Exception:
            pass

    # 方案3: [(node, ts)]
    if feat is None and hasattr(node_raw_features, "__getitem__"):
        try:
            feat = node_raw_features[(node_id, ts)]
        except Exception:
            pass

    # 方案4: [node, ts]
    if feat is None and hasattr(node_raw_features, "__getitem__"):
        try:
            feat = node_raw_features[node_id, ts]
        except Exception:
            pass

    # 方案5: query(node, ts)
    if feat is None and hasattr(node_raw_features, "query"):
        try:
            feat = node_raw_features.query(node_id, ts)
        except Exception:
            pass

    return _format_feature(feat, feat_dim)


def get_node_feature(graph_df, node_raw_features, NODE_FEAT_DIM):
    """
    输出仍保持旧格式：
        node_raw_feature_dict[(node_id, ts)] = np.ndarray(shape=(NODE_FEAT_DIM,))
    """

    node_raw_feature_dict = {}

    # -------------------------
    # 情况1：老版本，node_raw_features 本身就是 dict
    # -------------------------
    if isinstance(node_raw_features, dict):
        for k, v in node_raw_features.items():
            node_raw_feature_dict[k] = _format_feature(v, NODE_FEAT_DIM)
        return node_raw_feature_dict

    # -------------------------
    # 情况2：新版本，node_raw_features 是 NodeFeatureLookup
    # 从图里抽出所有真正需要的 (node, ts)
    # -------------------------
    src_pairs = graph_df[['u', 'ts']].rename(columns={'u': 'node'})
    dst_pairs = graph_df[['i', 'ts']].rename(columns={'i': 'node'})
    needed_pairs = pd.concat([src_pairs, dst_pairs], axis=0, ignore_index=True).drop_duplicates()

    for row in needed_pairs.itertuples(index=False):
        node_id = row.node
        ts = row.ts
        node_raw_feature_dict[(node_id, ts)] = _lookup_node_feature(
            node_raw_features=node_raw_features,
            node_id=node_id,
            ts=ts,
            feat_dim=NODE_FEAT_DIM
        )

    return node_raw_feature_dict



# def get_node_feature(graph_df, node_raw_features, NODE_FEAT_DIM):
#     # Convert node features to tuple-indexed format: {(node_id, timestamp): feature_vector}
#     # Create a mapping from node_id to feature for all unique nodes
#     # node_features_dict = {}
#     # for k,v in node_raw_features.items():
#     #     node_zero_padding = np.zeros(NODE_FEAT_DIM - 1)
#     #     v_array = np.array([v], dtype=float)
#     #     temp = np.concatenate((v_array , node_zero_padding))
#     #     # temp = np.concatenate((v , node_zero_padding))
#     #     node_features_dict[k] = temp.reshape(1, -1)

#     node_features_dict = {}
#     node_zero_padding = np.zeros(NODE_FEAT_DIM - 1)

#     for k, v in node_raw_features.items():
#         # k 是 (node_id, timestamp) 元组，直接保留
#         v_array = np.array([v], dtype=np.float32)
#         temp = np.concatenate((v_array, node_zero_padding)).astype(np.float32)
#         node_features_dict[k] = temp  # shape: (NODE_FEAT_DIM,)
#     return node_features_dict


def get_model_data(graph_df, edge_raw_features, node_raw_features, feature_dim: int):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    # graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    # edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    # node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = feature_dim

    # assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension is bigger than {NODE_FEAT_DIM}!'
    # assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension is bigger than {EDGE_FEAT_DIM}!'

    node_raw_feature_dict = get_node_feature(graph_df, node_raw_features, NODE_FEAT_DIM)

    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, 
                    node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))

    return node_raw_feature_dict, edge_raw_features, full_data



class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=False,
                            num_workers=1,          # 新增：多进程加载
                            pin_memory=True,        # 新增：锁页内存
                            prefetch_factor=2,      # 新增：预取因子
                            persistent_workers=True # 新增：保持worker进程
                            )
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        self.uni_time_ids = set(node_interact_times)
        self.num_unique_times = len(self.uni_time_ids)
        self.time_to_int_map = {}
        for i, time_id in enumerate(sorted(self.uni_time_ids)):
            self.time_to_int_map[time_id] = i+1


def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data
