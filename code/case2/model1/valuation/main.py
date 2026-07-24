"""
full_pipeline.py
完整下游流程: 对4个模型 (1) 重新导出带负边的预测分数 (2) 重建稀疏GRN
前提: 4个模型 AUC 均已验证 > 0.85, checkpoint 存在。

两阶段:
  A. export_scores(): 用已训模型对正边+负边打分, 存 my_result_run0.npy + neg_scores_run0.npy
  B. build_grn():     读分数, 收紧motif + 对照边阈值 + per-TF/peak排名稀疏化
"""
import os
import sys
import time
import pickle
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append("/home/wuyan/dygmamba_project/model/dygmamba/src")
from models.DyGMamba import DyGMamba
from models.modules import MergeLayerTD
from utils.DataLoader import get_model_data, get_idx_data_loader
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.utils import set_random_seed, convert_to_gpu
from self_utils.new_regulation import NodeFeatureLookup

# ================================================================
# 全局配置
# ================================================================
DATA_ROOT = "/home/wuyan/dygmamba_project/NewRealPlan/case2/data/AD/process/"
MODELS = ["model1_CRND8_Microglia", "model2_WT_Microglia",
          "model3_CRND8_Astrocyte", "model4_WT_Astrocyte"]

GPU = 0
BATCH_SIZE = 1000
MAX_NEIGHBORS = 30
SEED = 0

# 阶段A: 导出时是否全量 (None=全量, 用于最终构网; 数字=抽样, 仅快速验证AUC)
EXPORT_MAX_BATCHES = None      # 构网必须全量, 否则边不全

# 阶段B: 稀疏化参数 (跑完看诊断打印再调)
MOTIF_VALUE_MIN = 3.0          # -log10(p) 下限, 保留显著motif
MOTIF_SCORE_MIN, MOTIF_SCORE_MAX = 6.0, 17.5
NEG_QUANTILE = 0.99            # 对照边分位数作为正边可信下限
TOP_K_PER_PEAK = 10
TOP_K_PER_TF = 50
N_WINDOWS = 12
WINDOW_OVERLAP = 0.5
TOP_K_PER_TF_DYNAMIC = 50


def load_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_name', type=str, default='mooc')
    p.add_argument('--model_name', type=str, default='DyGMamba')
    p.add_argument('--gpu', type=int, default=GPU)
    p.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    p.add_argument('--sample_neighbor_strategy', type=str, default='uniform')
    p.add_argument('--time_scaling_factor', type=float, default=1e-6)
    p.add_argument('--num_heads', type=int, default=2)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--time_feat_dim', type=int, default=100)
    p.add_argument('--channel_embedding_dim', type=int, default=50)
    p.add_argument('--patch_size', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--max_input_sequence_length', type=int, default=32)
    p.add_argument('--max_interaction_times', type=int, default=10)
    p.add_argument('--negative_sample_strategy', type=str, default='random')
    args = p.parse_args([])
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    args.seed = SEED
    args.save_model_name = f'{args.model_name}_seed{args.seed}'
    return args


def find_checkpoint(data_path, args):
    folder = (f"{data_path}/saved_models/{args.model_name}/"
              f"{args.dataset_name}/{args.save_model_name}")
    if not os.path.isdir(folder):
        return None
    cands = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.endswith(('.pkl', '.pt', '.pth')) or args.save_model_name in f]
    return max(cands, key=os.path.getmtime) if cands else None


# ================================================================
# 阶段 A: 导出正边 + 负边分数
# ================================================================
def export_scores(traj):
    args = load_args()
    data_path = DATA_ROOT + traj + "/process/"
    print("\n" + "=" * 60)
    print(f"[阶段A] 导出分数: {traj}")
    print("=" * 60)

    if not os.path.exists(data_path + "Graph_df.pkl"):
        print(f"  Graph_df.pkl 缺失 -> 跳过 (此模型可能未跑完)")
        return False

    set_random_seed(seed=args.seed)

    # 加载数据
    Edge_feature = np.load(data_path + "edge_features.npy", mmap_mode="r").reshape(-1, 1).copy()
    Node_feature = NodeFeatureLookup.load(
        data_path + "node_feat_matrix.dat", data_path + "node_feat_meta.pkl")
    graph_df = pd.read_pickle(data_path + "Graph_df.pkl")
    graph_df["Unnamed"] = graph_df.index
    New_Graph = graph_df[["Unnamed", "source_node", "target_node",
                          "time", "label", "edge_idx"]].copy()
    New_Graph.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx']

    node_raw_feature_dict, edge_raw_features, full_data = get_model_data(
        New_Graph, Edge_feature, Node_feature, feature_dim=172)
    edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(args.device)

    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor, seed=1, max_neighbors=MAX_NEIGHBORS)
    neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
        interact_times=full_data.node_interact_times,
        negative_sample_strategy='random', seed=2)
    idx_loader = get_idx_data_loader(
        indices_list=list(range(len(full_data.src_node_ids))),
        batch_size=args.batch_size, shuffle=False)

    # 构建模型 + 加载权重
    backbone = DyGMamba(
        node_feat_dim=172, edge_feat_dim=172, time_feat_dim=args.time_feat_dim,
        channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
        num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
        gamma=args.gamma, max_input_sequence_length=args.max_input_sequence_length,
        max_interaction_times=args.max_interaction_times, device=args.device)
    predictor = MergeLayerTD(input_dim1=172, input_dim2=172, input_dim3=172,
                             hidden_dim=172, output_dim=1)
    model = nn.Sequential(backbone, predictor)
    model = convert_to_gpu(model, device=args.device)

    ckpt = find_checkpoint(data_path, args)
    if ckpt is None:
        print(f"  [错误] 无 checkpoint -> 跳过")
        return False
    print(f"  加载权重: {os.path.basename(ckpt)}")
    state = torch.load(ckpt, map_location=args.device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.eval()
    model[0].set_neighbor_sampler(full_neighbor_sampler)

    # 推理: 正边(带edge_ids) + 负边(不带), 与训练一致
    pos_scores, neg_scores = [], []
    nb = len(idx_loader) if EXPORT_MAX_BATCHES is None else min(len(idx_loader), EXPORT_MAX_BATCHES)
    print(f"  推理 {nb} 个 batch ({'全量' if EXPORT_MAX_BATCHES is None else '抽样'})...")

    with torch.no_grad():
        for bi, ids in enumerate(tqdm(idx_loader, total=nb, ncols=100)):
            if EXPORT_MAX_BATCHES is not None and bi >= EXPORT_MAX_BATCHES:
                break
            idx = ids.numpy()
            src, dst = full_data.src_node_ids[idx], full_data.dst_node_ids[idx]
            t, eid = full_data.node_interact_times[idx], full_data.edge_ids[idx]

            neg_src, neg_dst = neg_edge_sampler.sample(
                size=len(src), batch_src_node_ids=src, batch_dst_node_ids=dst,
                current_batch_start_time=min(t), current_batch_end_time=max(t))

            src_t = torch.tensor(src, device=args.device)
            dst_t = torch.tensor(dst, device=args.device)
            nsrc_t = torch.tensor(neg_src, device=args.device)
            ndst_t = torch.tensor(neg_dst, device=args.device)

            pe_s, pe_d, pe_td = model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_t, dst_node_ids=dst_t, node_interact_times=t,
                node_features=node_raw_feature_dict, edge_features=edge_raw_features, edge_ids=eid)
            ne_s, ne_d, ne_td = model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=nsrc_t, dst_node_ids=ndst_t, node_interact_times=t,
                node_features=node_raw_feature_dict, edge_features=edge_raw_features)

            pos = model[1](input_1=pe_s, input_2=pe_d, input_3=pe_td).squeeze(-1).sigmoid()
            neg = model[1](input_1=ne_s, input_2=ne_d, input_3=ne_td).squeeze(-1).sigmoid()
            pos_scores.append(pos.cpu().numpy().ravel())
            neg_scores.append(neg.cpu().numpy().ravel())

    pos = np.concatenate(pos_scores)
    neg = np.concatenate(neg_scores)

    # 报告 AUC
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    scores = np.concatenate([pos, neg])
    auc = roc_auc_score(labels, scores)
    print(f"  正边: 均值={pos.mean():.4f} 中位={np.median(pos):.4f}")
    print(f"  负边: 均值={neg.mean():.4f} 中位={np.median(neg):.4f}")
    print(f"  AUC={auc:.4f}  AP={average_precision_score(labels, scores):.4f}")

    # 保存 (关键: 同时存正边和负边)
    # 注意: 全量时 pos 的顺序与 idx_loader(shuffle=False) 一致, 对应 Graph_df 行序
    np.save(data_path + "my_result_run0.npy", pos)        # 覆盖旧的(纯正边但现在确认可信)
    np.save(data_path + "neg_scores_run0.npy", neg)       # 新增: 负边分数, 供构网定阈值
    print(f"  已存 my_result_run0.npy ({len(pos):,}) + neg_scores_run0.npy ({len(neg):,})")
    return True


# ================================================================
# 阶段 B: 构建稀疏 GRN
# ================================================================
def build_grn(traj):
    data_path = DATA_ROOT + traj + "/process/"
    print("\n" + "=" * 60)
    print(f"[阶段B] 构建 GRN: {traj}")
    print("=" * 60)

    need = ["jaspar_df.pkl", "node_id.pkl", "Graph_df.pkl", "my_result_run0.npy"]
    miss = [f for f in need if not os.path.exists(data_path + f)]
    if miss:
        print(f"  缺失 {miss} -> 跳过")
        return None

    # ---- 1. TF-Peak motif 过滤 (修正方向) ----
    jaspar = pd.read_pickle(data_path + "jaspar_df.pkl")
    jaspar['sequence_name'] = jaspar['sequence_name'].str.replace(':', '-')
    tfp = jaspar[['TF_Symbol', 'sequence_name', 'score', 'p-value', 'q-value']].copy()
    tfp.rename(columns={'TF_Symbol': 'TF', 'sequence_name': 'Peak'}, inplace=True)
    tfp.drop_duplicates(subset=['TF', 'Peak'], keep='first', inplace=True)
    tfp['value'] = -np.log10(tfp['p-value'].clip(lower=1e-300))
    mask = ((tfp['score'] > MOTIF_SCORE_MIN) & (tfp['score'] < MOTIF_SCORE_MAX)
            & (tfp['value'] >= MOTIF_VALUE_MIN))
    tf_peak = tfp[mask].copy()
    print(f"[1] TF-Peak: {len(tfp):,} -> {len(tf_peak):,} "
          f"({tf_peak['TF'].nunique()} TF, {tf_peak['Peak'].nunique()} peaks)")
    tf_peak[['TF', 'Peak', 'value']].to_pickle(data_path + 'pred_tf_peak_v2.pkl')

    # ---- 2. Peak-Gene 加载预测 ----
    Node_id = pd.read_pickle(data_path + "node_id.pkl")
    graph_df = pd.read_pickle(data_path + "Graph_df.pkl")
    graph_df["Unnamed"] = graph_df.index
    NG = graph_df[["Unnamed", "source_node", "target_node",
                   "time", "label", "edge_idx"]].copy()
    NG.columns = ['Unnamed: 0', 'u', 'i', 'ts', 'label', 'idx']

    pred = np.load(data_path + 'my_result_run0.npy').ravel()
    if len(pred) != len(NG):
        print(f"  [警告] 预测长度 {len(pred):,} != 图边数 {len(NG):,}")
        print(f"         说明 my_result_run0.npy 是抽样导出的, 请用 EXPORT_MAX_BATCHES=None 重跑阶段A")
        return None
    NG["predict"] = pred

    mapping = Node_id["name"]
    NG['source'] = (NG['u'] - 1).map(mapping)
    NG['target'] = (NG['i'] - 1).map(mapping)
    pg = NG[['source', 'target', 'ts', 'predict']].rename(
        columns={'source': 'Peak', 'target': 'Gene'})
    pg = pg[~pg["Gene"].astype(str).str.startswith('chr')].copy()
    pg['Peak'] = pg['Peak'].str.replace(':', '-', regex=False)
    print(f"[2] Peak-Gene 预测: {len(pg):,} 行, predict 中位={pg['predict'].median():.4f}")

    # ---- 2a. 对照边定阈值 ----
    neg_path = data_path + "neg_scores_run0.npy"
    if os.path.exists(neg_path):
        neg = np.load(neg_path).ravel()
        thr = np.quantile(neg, NEG_QUANTILE)
        print(f"    对照边 {int(NEG_QUANTILE*100)}% 分位 = {thr:.4f} (正边下限)")
    else:
        thr = 0.5
        print(f"    [无对照边分数] 用默认下限 {thr} -- 建议先跑阶段A")
    pg_sig = pg[pg["predict"] > thr].copy()
    print(f"    过滤后: {len(pg):,} -> {len(pg_sig):,}")
    if len(pg_sig) == 0:
        print("    [错误] 过滤后为空, 阈值太高")
        return None
    pg_sig.to_pickle(data_path + "pred_time_peak_gene_v2.pkl")

    # ---- 2c. per-peak 排名 -> 静态 peak-gene ----
    pg_static = pg_sig.groupby(['Peak', 'Gene'])['predict'].mean().reset_index()
    pg_static = (pg_static.sort_values('predict', ascending=False)
                 .groupby('Peak', group_keys=False).head(TOP_K_PER_PEAK)
                 .reset_index(drop=True))
    print(f"[2c] 静态 Peak-Gene (per-peak top{TOP_K_PER_PEAK}): {len(pg_static):,} 边")
    pg_static.to_pickle(data_path + "pred_peak_gene_v2.pkl")

    # ---- 3. TF-Gene 静态: mean 强度 + per-TF 排名 ----
    merged = pd.merge(tf_peak[['TF', 'Peak']], pg_static, on='Peak')
    tf_gene = (merged.groupby(['TF', 'Gene'])
               .agg(weight=('predict', 'mean'), peak_num=('Peak', 'nunique'))
               .reset_index())
    print(f"[3] TF-Gene 桥接: {len(tf_gene):,} 边 "
          f"({tf_gene['TF'].nunique()} TF, {tf_gene['Gene'].nunique()} 基因)")
    tf_gene_sparse = (tf_gene.sort_values('weight', ascending=False)
                      .groupby('TF', group_keys=False).head(TOP_K_PER_TF)
                      .reset_index(drop=True))
    print(f"    per-TF top{TOP_K_PER_TF}: -> {len(tf_gene_sparse):,} 边")
    # 诊断: 稀疏化后权重还饱和吗?
    w = tf_gene_sparse['weight']
    print(f"    权重分布: 中位={w.median():.4f} 10%={w.quantile(.1):.4f} 90%={w.quantile(.9):.4f}")
    if w.quantile(.9) - w.quantile(.1) < 0.02:
        print(f"    [注意] 权重仍高度集中 -> 下游建议用'边是否存在'(结构)而非权重高低做结论")
    tf_gene_sparse.to_pickle(data_path + "pred_tf_gene_v2.pkl")

    # ---- 4. TF-Gene 动态: 滑窗 ----
    ts_min, ts_max = pg_sig["ts"].min(), pg_sig["ts"].max()
    win = (ts_max - ts_min) / (N_WINDOWS * (1 - WINDOW_OVERLAP) + WINDOW_OVERLAP)
    step = win * (1 - WINDOW_OVERLAP)
    dynamic = {}
    for wi in range(N_WINDOWS):
        lo, hi = ts_min + wi * step, ts_min + wi * step + win
        wpg = pg_sig[(pg_sig["ts"] >= lo) & (pg_sig["ts"] < hi)]
        if len(wpg) == 0:
            continue
        wpg_agg = wpg.groupby(['Peak', 'Gene'])['predict'].mean().reset_index()
        wm = pd.merge(tf_peak[['TF', 'Peak']], wpg_agg, on='Peak')
        if len(wm) == 0:
            continue
        wtg = (wm.groupby(['TF', 'Gene'])['predict'].mean().reset_index()
               .rename(columns={'predict': 'weight'}))
        wtg = (wtg.sort_values('weight', ascending=False)
               .groupby('TF', group_keys=False).head(TOP_K_PER_TF_DYNAMIC)
               .reset_index(drop=True))
        wtg['window'] = wi
        wtg['ts_center'] = (lo + hi) / 2
        dynamic[wi] = wtg
    if dynamic:
        nrec = sum(len(v) for v in dynamic.values())
        print(f"[4] 动态: {len(dynamic)} 窗, {nrec:,} 边记录")
        with open(data_path + "pred_time_tf_gene_v2.pkl", "wb") as f:
            pickle.dump(dynamic, f)
    else:
        print(f"[4] 无有效时间窗")

    print(f"  完成 -> *_v2.pkl")
    return tf_gene_sparse, dynamic


# ================================================================
# 主程序
# ================================================================
if __name__ == "__main__":
    t0 = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始完整流程")

    # 阶段A: 对所有模型重新导出带负边的分数 (全量)
    print("\n########## 阶段 A: 导出分数 ##########")
    exported = {}
    traj = MODELS[0]
    try:
        exported[traj] = export_scores(traj)
    except Exception as e:
        print(f"  {traj} 阶段A失败: {e}")
        exported[traj] = False

    # 阶段B: 构网
    print("\n########## 阶段 B: 构建 GRN ##########")

    if not exported.get(traj):
        print(f"\n{traj}: 阶段A未成功, 跳过构网")
    else:
        build_grn(traj)


    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 全部完成, {time.time()-t0:.0f}s")
    print("\n下一步: 检查每个模型 [3] 的 TF-Gene 边数和权重分布")
    print("  - 边数应在数千~数万 (非91万)")
    print("  - 若权重仍高度集中, 下游用结构差异(Jaccard/边组成)而非权重高低")