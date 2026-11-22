"""
TacoPrompt hidden 变体：dump 测试集每个 query 的 Top-K 排序结果。

说明
----
本脚本**不修改**项目源码，只 import 项目已有模块，复现 inference_scripts/infer_hidden.py
的候选位置集合、prompt 构造与打分口径，额外把每个 query 的 Top-K (parent, child) 结果存盘，
供后续错误分类统计使用。

打分口径与 model/model.py: TacoPromptHidden.scorer 完全一致：
    prompt = "All: [MASK] Parent: [MASK] Chlid: [MASK]. <Parent>: {p_desc} <Child>: {c_desc}[SEP]<Query>: {q_desc}"
    score  = logit(yes) - logit(no) 在**第一个 MASK**（All 位）上的取值
唯一区别：不在全部 256 个位置上跑 MLM head（那会浪费 ~7 倍算力），
只对 3 个 MASK 位置跑 cls head，数学上完全等价（可用 --verify 校验）。

用法
----
python analysis/dump_hidden_rank.py \
    -c config_files/SemEval-V/config.test.hidden.json \
    -m data/saved/wordnet_verb/TacoPromptHidden/<ts>/models/trial1/model_best.pth \
    -o analysis/results/verb_top10.json \
    [--k 10] [--shard 0 --num_shards 1] [--verify]

分片（--shard/--num_shards）用于多卡并行跑同一份推理，最后合并 json 即可。
"""

import argparse
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import data_loader.data_loaders as module_data  # noqa: E402
import model.model as module_arch  # noqa: E402
import more_itertools as mit  # noqa: E402
from tqdm import tqdm  # noqa: E402


def setup_seed(seed=47):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def build_loader(cfg):
    dl_cfg = dict(cfg["train_data_loader"]["args"])
    taxonomy_name = dl_cfg.pop("taxonomy_name")
    loader_cls = getattr(module_data, cfg["train_data_loader"]["type"])
    return loader_cls(data_path=cfg["data_path"], taxonomy_name=taxonomy_name, **dl_cfg)


def build_model(cfg, dataset, model_path, device):
    model_cls = getattr(module_arch, cfg["arch"]["type"])
    model = model_cls(dataset, **cfg["arch"]["args"])
    state = torch.load(model_path, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def fast_scores(model, p_idx, q_idx, c_idx):
    """与 model.scorer 等价，但只对 MASK 位置跑 MLM head。

    返回 (B,) 的 tensor，对应 scorer(...)[:, 0]。
    """
    batch_input_ids, _, batch_attention_mask = model.get_lm_inputs(
        p_idx, q_idx, c_idx,
        torch.ones(p_idx.shape[0], dtype=torch.long),
        torch.ones(p_idx.shape[0], dtype=torch.long),
        torch.ones(p_idx.shape[0], dtype=torch.long),
    )
    batch_input_ids = batch_input_ids.to(next(model.parameters()).device)
    batch_attention_mask = batch_attention_mask.to(next(model.parameters()).device)
    with autocast():
        hidden = model.model.bert(input_ids=batch_input_ids,
                                  attention_mask=batch_attention_mask)[0]
        mask_pos = (batch_input_ids == model.mask_token_id)
        rows = hidden[mask_pos]                      # (B*n_mask, H)
        logits = model.model.cls(rows)               # (B*n_mask, V)
        score = logits[:, model.yes_token_id] - logits[:, model.no_token_id]
        scores = score.view(p_idx.shape[0], -1)
    return scores[:, 0].float()


def _verify_full(model, device, ps_all_t, cs_all_t, q_it, fast, candidate_positions, true_pos, batch_size):
    """用项目原版 scorer 对**全部候选位置**打分，与 fast_scores 对拍（含 top-10 集合/顺序）。"""
    ref = []
    n_cand = len(candidate_positions)
    for start in range(0, n_cand, batch_size):
        end = min(start + batch_size, n_cand)      # 最后一个 chunk 必须截断，否则 label 比切片长
        bs = end - start
        with torch.no_grad(), autocast():
            r = model.scorer(ps_all_t[start:end].to(device), q_it[start:end],
                             cs_all_t[start:end].to(device),
                             torch.ones(bs, dtype=torch.long),
                             torch.ones(bs, dtype=torch.long),
                             torch.ones(bs, dtype=torch.long))[:, 0]
        ref.append(r.cpu().float())
    ref = torch.cat(ref)

    d = (ref - fast).abs()
    print(f"[verify_full] n={len(ref)} |score|max={ref.abs().max():.2f} "
          f"max|ref-fast|={d.max():.3e} mean={d.mean():.3e}", flush=True)

    k = 10
    t_ref = torch.topk(ref, k, largest=True, sorted=True)[1].tolist()
    t_fast = torch.topk(fast, k, largest=True, sorted=True)[1].tolist()
    print(f"[verify_full] top-{k} 集合重合 {len(set(t_ref) & set(t_fast))}/{k}, "
          f"顺序完全一致={t_ref == t_fast}", flush=True)

    pos2idx = {tuple([p.display_name, c.display_name]): i for i, (p, c) in enumerate(candidate_positions)}
    for p, c in true_pos:
        idx = pos2idx.get((p.display_name, c.display_name))
        if idx is None:
            continue
        r_ref = int((ref > ref[idx]).sum().item()) + 1
        r_fast = int((fast > fast[idx]).sum().item()) + 1
        print(f"[verify_full] 真实位置 ({p.display_name} | {c.display_name}) "
              f"rank: 原版={r_ref} 快路径={r_fast}", flush=True)

    for i in [i for i in t_fast if i not in t_ref]:
        p, c = candidate_positions[i]
        print(f"[verify_full] 快路径独有: {p.display_name} | {c.display_name} "
              f"ref={ref[i]:.4f} fast={fast[i]:.4f}", flush=True)
    for i in [i for i in t_ref if i not in t_fast]:
        p, c = candidate_positions[i]
        print(f"[verify_full] 原版独有: {p.display_name} | {c.display_name} "
              f"ref={ref[i]:.4f} fast={fast[i]:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-k", "--topk", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--verify", action="store_true", help="抽样校验 fast_scores 与 model.scorer 一致")
    ap.add_argument("--verify_full", action="store_true",
                    help="第一个 query 用项目原版 scorer 全量打分对拍 top-10（每份数据只跑一次，约 2 min）")
    ap.add_argument("--max_queries", type=int, default=-1, help="smoke test 用：只跑前 N 个 query")
    ap.add_argument("--max_candidates", type=int, default=-1, help="smoke test 用：只用前 N 个候选位置")
    args = ap.parse_args()

    setup_seed(47)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] loading data ...", flush=True)
    loader = build_loader(cfg)
    dataset = loader.dataset
    print(f"[{time.strftime('%H:%M:%S')}] loading model ...", flush=True)
    model = build_model(cfg, dataset, args.model_path, device)

    batch_size = cfg["trainer"]["test_batch_size"]
    candidate_positions = dataset.all_edges           # 与 Tester.__init__ 一致
    if args.max_candidates > 0:
        candidate_positions = candidate_positions[:args.max_candidates]
    print(f"#candidates = {len(candidate_positions)}, batch = {batch_size}", flush=True)

    queries = dataset.test_node_list
    if args.max_queries > 0:
        queries = queries[:args.max_queries]
    node2pos = dataset.test_node2pos
    taxon2all_node_id = model.taxon2id
    taxon2allemb_id = model.train_taxon2allemb_id

    # 与 infer_hidden.py 相同的 query 切分方式
    if args.num_shards > 1:
        queries = [q for i, q in enumerate(queries) if i % args.num_shards == args.shard]
    print(f"#queries(this shard) = {len(queries)}", flush=True)

    # 预先把候选位置转成 allemb_id，省掉每轮的 python 循环
    ps_all = [taxon2allemb_id[p] for p, _ in candidate_positions]
    cs_all = [taxon2allemb_id[c] for _, c in candidate_positions]
    ps_all_t = torch.tensor(ps_all, dtype=torch.long)
    cs_all_t = torch.tensor(cs_all, dtype=torch.long)

    results = {}
    pos2idx = {(p.display_name, c.display_name): i for i, (p, c) in enumerate(candidate_positions)}
    n_missing_pos = 0
    t0 = time.time()
    for qi, query in enumerate(tqdm(queries, desc="infer")):
        q_allemb = taxon2all_node_id[query]
        q_it = torch.full((len(candidate_positions),), q_allemb, dtype=torch.long)
        scores = []
        for start in range(0, len(candidate_positions), batch_size):
            end = start + batch_size
            with torch.no_grad():
                s = fast_scores(model, ps_all_t[start:end], q_it[start:end], cs_all_t[start:end])
            scores.append(s.cpu())
            if args.verify and qi == 0 and start == 0 and torch.cuda.is_available():
                with torch.no_grad(), autocast():
                    ref = model.scorer(
                        ps_all_t[start:end].to(device), q_it[start:end], cs_all_t[start:end].to(device),
                        torch.ones(end - start, dtype=torch.long),
                        torch.ones(end - start, dtype=torch.long),
                        torch.ones(end - start, dtype=torch.long))[:, 0]
                print(f"[verify] max|fast-scorer| = {(ref.cpu().float() - s.cpu()).abs().max().item():.3e}",
                      flush=True)
        scores = torch.cat(scores)
        if args.verify_full and qi == 0 and torch.cuda.is_available():
            _verify_full(model, device, ps_all_t, cs_all_t, q_it, scores,
                         candidate_positions, node2pos[query], batch_size)
        k = min(args.topk, scores.shape[0])
        top_scores, top_idx = torch.topk(scores, k, largest=True, sorted=True)

        true_pos = [[p.display_name, c.display_name] for p, c in node2pos[query]]
        ranks = []
        for name_pair in true_pos:
            idx = pos2idx.get(tuple(name_pair))
            if idx is None:
                n_missing_pos += 1
                continue
            s_true = scores[idx]
            ranks.append(int((scores > s_true).sum().item()) + 1)
        is_leaf = all(c is dataset.pseudo_leaf_node for _, c in node2pos[query])
        topk = []
        for sc, idx in zip(top_scores.tolist(), top_idx.tolist()):
            p, c = candidate_positions[idx]
            topk.append([p.display_name, c.display_name, sc])

        results[query.display_name] = {
            "is_leaf": bool(is_leaf),
            "true_positions": true_pos,
            "ranks": ranks,
            "topk": topk,
        }
        if (qi + 1) % 20 == 0:
            el = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] {qi + 1}/{len(queries)} queries, "
                  f"{el / (qi + 1):.1f}s/query, eta {(len(queries) - qi - 1) * el / (qi + 1) / 3600:.2f}h",
                  flush=True)
            _dump(results, args, cfg, candidate_positions, n_missing_pos, Path(str(args.out) + '.partial'), True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if n_missing_pos:
        print(f"WARNING: {n_missing_pos} true positions not found in candidate list", flush=True)
    _dump(results, args, cfg, candidate_positions, n_missing_pos, out_path, False)
    print(f"[{time.strftime('%H:%M:%S')}] saved {len(results)} queries -> {out_path}", flush=True)


def _dump(results, args, cfg, candidate_positions, n_missing_pos, out_path, partial):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump({
            "config": args.config,
            "model_path": args.model_path,
            "taxonomy_name": cfg["train_data_loader"]["args"]["taxonomy_name"],
            "num_candidates": len(candidate_positions),
            "n_missing_pos": n_missing_pos,
            "shard": args.shard,
            "num_shards": args.num_shards,
            "partial": partial,
            "results": results,
        }, f, ensure_ascii=False)
    os.replace(tmp, out_path)          # 原子替换，避免半截文件


if __name__ == "__main__":
    main()
