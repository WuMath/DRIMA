import logging
import time
from datetime import datetime
import warnings
import shutil
import json
import pickle
import torch
import sys
from torch.cuda.amp import autocast, GradScaler  # 新增：混合精度训练
import psutil  # 新增：系统监控

sys.path.append("/home/liyang/BioWuYan/dygmamba_project/model/dygmamba/src")

import os
import pandas as pd
import numpy as np
import networkx as nx
import scanpy as sc
import anndata as ad

import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from itertools import islice

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from models.DyGMamba import DyGMamba
from models.modules import MergeLayer, MergeLayerTD

from utils.load_configs import load_link_prediction_args

from utils.DataLoader import get_model_data
from utils.DataLoader import get_idx_data_loader
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.utils import get_parameter_sizes
from utils.utils import set_random_seed
from utils.utils import convert_to_gpu, create_optimizer
from utils.EarlyStopping import EarlyStopping
from utils.metrics import get_link_prediction_metrics
from models.evaluate_models_utils import evaluate_model_link_prediction
from models.inference_grn import model_link_prediction



#################################################################
#
#################################################################

if __name__ == "__main__":
    
    cell_type = "PC3"

    print("********************** start ********************")

    start_time = time.time()  # start the time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start the job")

    # get arguments
    args = load_link_prediction_args(is_evaluation=False)
    
    # ============ 新增：优化配置参数 ============
    USE_AMP = False # True  # 混合精度训练开关
    ACCUMULATION_STEPS = 4  # 梯度累积步数
    MAX_NEIGHBORS = 30  # 最大邻居采样数
    
    print("**********************device********************")
    print(f"Now use device is {args.device}")

    #########################################################################################
    # *************************** load data ******************************************

    data_path = "/home/liyang/BioWuYan/dygmamba_project/data/cell_line/" + cell_type + "/data_dyg/"

    feat_path = data_path + "edge_features.npy"
    edge_label_path = data_path + "edge_labels.npy"

    Edge_feature = np.load(feat_path, mmap_mode="r")
    Edge_feature = Edge_feature.reshape(-1,1).copy()
    Edge_label = np.load(edge_label_path, mmap_mode="r")
    Edge_label = Edge_label.reshape(-1,1).copy()

    with open(data_path + "node_feature_data.pkl", "rb") as f:
        load_data = pickle.load(f)

    Node_feature = load_data['node_feature']

    Node_id = pd.read_pickle(data_path + "node_id.pkl")

    graph_df = pd.read_pickle(data_path + "Graph_df.pkl")
    graph_df["Unnamed"] = graph_df.index
    name_list = ["Unnamed", "source_node", "target_node", "time", "label", "edge_idx"]
    New_Graph = graph_df[name_list].copy()
    New_Graph.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx']

    print("********************** Successfully load data ********************")

    #########################################################################################
    # *************************** process data ******************************************

    node_raw_feature_dict, edge_raw_features, full_data= get_model_data(New_Graph,
                                                            Edge_feature,
                                                            Node_feature,
                                                            feature_dim= 172)
    
    edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(args.device)

    Edge_label = torch.from_numpy(Edge_label.astype(np.float32)).to(args.device)
    
    if torch.isnan(edge_raw_features).any():
        print(f"Edge_feature has Nan values. Please check the data.")
        sys.exit(1)
    else:
        print(f"Edge_feature is successfully converted to tensor without NaN values.")
        
    if np.isnan(list(node_raw_feature_dict.values())).any():
        print(f"node_feature has Nan values. Please check the data.")
        sys.exit(1)
    else:
        print(f"node_feature is successfully converted to dict without NaN values.")
    
    print("********************** Successfully process data ********************")

    ##########################################################
    # get neighbor data

    full_neighbor_sampler = get_neighbor_sampler(data=full_data,
                                                sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                time_scaling_factor=args.time_scaling_factor, 
                                                seed=1,
                                                max_neighbors=MAX_NEIGHBORS) # 新增：限制邻居数量
    
    print("********************* successfully get neighbor **************")
    neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                dst_node_ids=full_data.dst_node_ids,
                                                interact_times=full_data.node_interact_times,
                                                negative_sample_strategy='random', # inductive
                                                seed=2)
    print("********************* successfully get negative neighbor **************")
    
    ###########################################################
    # get index data for batch analysis
    
    effective_batch_size = args.batch_size * ACCUMULATION_STEPS
    
    idx_data_loader = get_idx_data_loader(indices_list=list(range(len(full_data.src_node_ids))),
                                          batch_size=effective_batch_size,  # 修改：使用更大的batch size
                                          shuffle=True)  # 修改：启用数据打乱
    

    metric_all_runs = []

    print("********************** Successfully process neighbor data---- positive and negative ********************")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start the job")
    ###########################################################
    # run data
    for run in range(args.num_runs) :

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'
        ########################################################
        # set up logger
        ######################################

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        os.makedirs(f"{data_path}/logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)

        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"{data_path}/logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        ########################################################
        # create model
        #####################################
        dynamic_backbone = DyGMamba(node_feat_dim= 172,
                                    edge_feat_dim=172,
                                    time_feat_dim=args.time_feat_dim,
                                    channel_embedding_dim=args.channel_embedding_dim,
                                    patch_size=args.patch_size,
                                    num_layers=args.num_layers,
                                    num_heads=args.num_heads,
                                    dropout=args.dropout,
                                    gamma=args.gamma,
                                    max_input_sequence_length=args.max_input_sequence_length,
                                    max_interaction_times=args.max_interaction_times,
                                    device=args.device)

        link_predictor = MergeLayerTD(input_dim1=172,
                                    input_dim2=172,
                                    input_dim3=172,
                                    hidden_dim=172,
                                      output_dim=1)

        model = nn.Sequential(dynamic_backbone, link_predictor)

        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        ########################################################
        # create optimizer
        ######################################
        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer,
                             learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"{data_path}/saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}"

        shutil.rmtree(save_model_folder, ignore_errors=True)

        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                    save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        # loss_func = nn.BCELoss()
        loss_func = nn.BCEWithLogitsLoss()
        
        # ============ 新增：创建混合精度Scaler ============
        scaler = GradScaler() if USE_AMP else None
        if USE_AMP:
            print("混合精度训练已启用")
        # =================================================

        best_acc = 0

        for epoch in range(args.num_epochs):

            ########################################################
            # train model
            ######################################

            model.train()

            model[0].set_neighbor_sampler(full_neighbor_sampler)

            train_losses, train_metrics = [], []

            idx_data_loader_tqdm = tqdm(idx_data_loader, ncols=120)

            for batch_idx, train_data_indices in enumerate(idx_data_loader_tqdm):

                train_data_indices = train_data_indices.numpy()

                batch_src_node_ids , batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    full_data.src_node_ids[train_data_indices], full_data.dst_node_ids[train_data_indices], \
                    full_data.node_interact_times[train_data_indices], full_data.edge_ids[train_data_indices]

                current_batch_start_time = min(batch_node_interact_times)
                current_batch_end_time = max(batch_node_interact_times)

                if neg_edge_sampler.negative_sample_strategy == 'random':
                    batch_neg_src_node_ids, batch_neg_dst_node_ids = neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                batch_src_node_ids=batch_src_node_ids,
                                                                                batch_dst_node_ids=batch_dst_node_ids,
                                                                                current_batch_start_time=current_batch_start_time,
                                                                                current_batch_end_time=current_batch_end_time)

                    # batch_neg_src_node_ids = batch_src_node_ids
                else:
                    batch_neg_src_node_ids, batch_neg_dst_node_ids = neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                batch_src_node_ids=batch_src_node_ids,
                                                                                batch_dst_node_ids=batch_dst_node_ids,
                                                                                current_batch_start_time=current_batch_start_time,
                                                                                current_batch_end_time=current_batch_end_time)

                ########################################################
                # feature embedding
                ######################################
                batch_src_node_ids = torch.tensor(batch_src_node_ids, device=args.device)
                batch_dst_node_ids = torch.tensor(batch_dst_node_ids, device=args.device)  # 新增
                batch_neg_src_node_ids = torch.tensor(batch_neg_src_node_ids, device=args.device)  # 新增
                batch_neg_dst_node_ids = torch.tensor(batch_neg_dst_node_ids, device=args.device)  # 新增
                
                if torch.isnan(batch_src_node_ids).any():
                    print(f"⚠️ NaN in batch_src_node_ids at batch {batch_idx}")
                    
                if torch.isnan(batch_dst_node_ids).any():
                    print(f"⚠️ NaN in batch_dst_node_ids at batch {batch_idx}")
                    
                if torch.isnan(batch_neg_src_node_ids).any():
                    print(f"⚠️ NaN in batch_neg_src_node_ids at batch {batch_idx}")
                    
                if torch.isnan(batch_neg_dst_node_ids).any():
                    print(f"⚠️ NaN in batch_neg_dst_node_ids at batch {batch_idx}")
                    
                # batch_src_node_ids.to(args.device)

                # ========== 新增：混合精度前向传播 ==========
                with autocast(enabled=USE_AMP):
                    batch_src_node_embeddings, batch_dst_node_embeddings, batch_time_diff_emb = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        node_features= node_raw_feature_dict,
                                                                        edge_features=edge_raw_features,
                                                                        edge_ids=batch_edge_ids)
                        
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings, batch_neg_time_diff_emb = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                        dst_node_ids=batch_neg_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        node_features= node_raw_feature_dict,
                                                                        edge_features=edge_raw_features)
                    # ########################################################
                    # # predicted
                    # ######################################
                    # positive_probabilities = model[1](input_1=batch_src_node_embeddings,
                    #                                     input_2=batch_dst_node_embeddings,
                    #                                     input_3=batch_time_diff_emb).squeeze(dim=-1).sigmoid()
                    
                    # negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings,
                    #                                     input_2=batch_neg_dst_node_embeddings,
                    #                                     input_3=batch_neg_time_diff_emb).squeeze(dim=-1).sigmoid()
                    # ########################################################
                    # # loss
                    # ######################################
                    # predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)

                    # labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                    # loss = loss_func(input=predicts, target=labels)/ ACCUMULATION_STEPS
                    
                    # ↓ 新增：检查 embedding 是否有 NaN
                    if torch.isnan(batch_src_node_embeddings).any():
                        print(f"⚠️ NaN in src_embeddings at batch {batch_idx}")

                    if torch.isnan(batch_dst_node_embeddings).any():
                        print(f"⚠️ NaN in dst_embeddings at batch {batch_idx}")

                    if torch.isnan(batch_time_diff_emb).any():
                        print(f"⚠️ NaN in time_diff_emb at batch {batch_idx}")
                        
                    # 预测（移除 .sigmoid()）
                    positive_logits = model[1](
                        input_1=batch_src_node_embeddings,
                        input_2=batch_dst_node_embeddings,
                        input_3=batch_time_diff_emb
                    ).squeeze(dim=-1)  # 不要 sigmoid
                    
                    negative_logits = model[1](
                        input_1=batch_neg_src_node_embeddings,
                        input_2=batch_neg_dst_node_embeddings,
                        input_3=batch_neg_time_diff_emb
                    ).squeeze(dim=-1)  # 不要 sigmoid
                    
                    # 合并 logits（仍在 autocast 内）
                    predicts = torch.cat([positive_logits, negative_logits], dim=0)
                    labels = torch.cat([
                        torch.ones_like(positive_logits), 
                        torch.zeros_like(negative_logits)
                    ], dim=0)
                    
                    # 计算损失（仍在 autocast 内，使用 BCEWithLogitsLoss）
                    loss = loss_func(input=predicts, target=labels) / ACCUMULATION_STEPS
                # ========== autocast 结束 ==========
                
                
                
                # 修改：记录真实的loss（乘回累积步数）
                train_losses.append(loss.item() * ACCUMULATION_STEPS)

                # train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
                # 修改后 ✅
                with torch.no_grad():
                    predicts_prob = torch.sigmoid(predicts.detach().float())  # 强制转为float32
                    predicts_prob = torch.nan_to_num(predicts_prob, nan=0.5, posinf=1.0, neginf=0.0)  # 替换NaN/Inf
                    predicts_prob = torch.clamp(predicts_prob, min=1e-7, max=1-1e-7)  # 限制范围

                train_metrics.append(get_link_prediction_metrics(
                    predicts=predicts_prob,
                    labels=labels
                ))

                # ========== 新增：混合精度反向传播 + 梯度累积 ==========
                if USE_AMP:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 梯度累积：每ACCUMULATION_STEPS步更新一次参数
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    if USE_AMP:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                # ====================================================
                # optimizer.zero_grad()

                # loss.backward()

                # optimizer.step()

                idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, \
                                        train for the {batch_idx + 1}-th batch, train loss: {loss.item()}\n')

            
            # ============ 新增：性能监控 ============
            avg_loss = np.mean(train_losses)

            # 打印GPU显存使用
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f'Epoch {epoch + 1} - GPU Memory: {mem_allocated:.2f}GB / {mem_reserved:.2f}GB')

            # 打印CPU内存
            cpu_mem = psutil.virtual_memory().percent
            logger.info(f'Epoch {epoch + 1} - CPU Memory: {cpu_mem:.1f}%')
            logger.info(f'Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}')
            # ========================================


            train_metric_indicator = []
            for metric_name in train_metrics[0].keys():

                logger.info(f'train {metric_name}, \
                            {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
                
                train_metric_indicator.append((metric_name, np.mean([train_metric[metric_name] for train_metric in train_metrics]), True))
            
            early_stop = early_stopping.step(train_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        result = model_link_prediction(model_name=args.model_name,
                                            model=model,
                                            neighbor_sampler=full_neighbor_sampler,
                                            evaluate_idx_data_loader=idx_data_loader,
                                            evaluate_data=full_data,
                                            node_features=node_raw_feature_dict,
                                            edge_features=edge_raw_features)

        # 1. 确保张量在 CPU 上 (如果它在 GPU 上)
        cpu_result = result.cpu()

        # 2. 转换为 NumPy 数组
        result_numpy = cpu_result.numpy()

        # 3. 保存为 .npy 文件
        np.save(data_path + f'my_result_run{run}.npy', result_numpy)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')
        
        # store the evaluation metrics at the current run
        metric_dict = {}


        logger.info(f'validate loss: {np.mean(train_losses):.4f}')
        for metric_name in train_metrics[0].keys():
            average_metric = np.mean([train_metric[metric_name] for train_metric in train_metrics])
            logger.info(f'validate {metric_name}, {average_metric:.4f}')
            metric_dict[metric_name] = average_metric


        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')


        metric_all_runs.append(metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result

        result_json = {
            "metrics": {metric_name: f'{metric_dict[metric_name]:.4f}' for metric_name in metric_dict},
            }

        result_json = json.dumps(result_json, indent=4)


        save_result_folder = f"{data_path}/saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)

        timestamp = str(time.time())
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}_{timestamp}.json")


        while os.path.exists(save_result_path):
            timestamp = str(time.time())
            save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}_{timestamp}.json")
    
        with open(save_result_path, 'w') as file:
            file.write(result_json)

        # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')


    for metric_name in metric_all_runs[0].keys():
        logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in metric_all_runs]}')
        logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in metric_all_runs]):.4f} '
                    f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in metric_all_runs], ddof=1):.4f}')
    
    end_time = time.time()
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished the job")

    print(f"The total time is: {end_time - start_time:.5f} second")

    sys.exit(0)


