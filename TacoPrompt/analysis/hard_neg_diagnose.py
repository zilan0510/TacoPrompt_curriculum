"""
结构化父节点负采样（课程学习）的诊断与合法性校验 —— 对应任务书 §2.3 / §6.5 / §8。

全部在 CPU 上跑，不需要 GPU：
    python analysis/hard_neg_diagnose.py -c config_files/mesh/config.test.hidden.json
    python analysis/hard_neg_diagnose.py -c config_files/SemEval-V/config.test.hidden.json

检查项：
  1) §6.5 分层抽样打印（20 个 leaf query 的 L1~L8 桶）
  2) §2.3 各层可采样数量统计（1000 个 query 的均值 / 中位数 / 为 0 的比例）
  3) 合法性审计：(p', c) ∈ all_edges、p' ∈ train_taxon2allemb_id、
     (p', c) ∉ node2pos[q]、无重复、条数恒定
  4) 配额核验：layer_hist 与 _allocate_quotas 的偏差
  5) RNG 一致性：hard_ratio=0 时 _get_mixed_negatives 与 _get_negative_positions 逐位一致
  6) 时间开销：每个 query 的负采样耗时（含/不含 hard）
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import data_loader.data_loaders as module_data  # noqa: E402

LAYER_NAME = {1: "L1 直接父", 2: "L2 直接子", 3: "L3 兄弟", 4: "L4 二跳父",
              5: "L5 二跳子", 6: "L6 三跳父", 7: "L7 三跳子", 8: "L8 四跳及以上"}


def build(cfg_path, hard_ratio=None, hard_schedule=None):
    cfg = json.load(open(cfg_path))
    args = dict(cfg["train_data_loader"]["args"])
    name = args.pop("taxonomy_name")
    if hard_ratio is not None:
        args["hard_ratio"] = hard_ratio
    if hard_schedule is not None:
        args["hard_schedule"] = hard_schedule
    loader = getattr(module_data, cfg["train_data_loader"]["type"])(
        data_path=cfg["data_path"], taxonomy_name=name, **args)
    return cfg, loader.dataset


def is_leaf(ds, q):
    return all(c is ds.pseudo_leaf_node for _, c in ds.node2pos[q])


def show_layers(ds, n=20):
    print("\n" + "=" * 78)
    print(f"1) §6.5 分层抽样打印（前 {n} 个 leaf query）")
    print("=" * 78)
    leaf_qs = [q for q in ds.node_list if is_leaf(ds, q)][:n]
    for q in leaf_qs:
        p0, c = ds.node2pos[q][0]
        layers = ds._layers_of(p0)
        print(f"query={q.display_name}  leaf=True  P0={p0.display_name}")
        for L in range(1, 9):
            bucket = ds._filter_hard(layers[L], p0, c, set(ds.node2pos[q]), q)
            names = [x.display_name for x in bucket[:3]]
            print(f"  {LAYER_NAME[L]}: {len(bucket)} 个  {names}")


def layer_stats(ds, n=1000):
    print("\n" + "=" * 78)
    print(f"2) §2.3 各层可采样数量（{n} 个 query）")
    print("=" * 78)
    per_layer = {L: [] for L in range(1, 9)}
    n_leaf = n_nonleaf = 0
    qs = ds.node_list[:n]
    t0 = time.time()
    for q in qs:
        leaf = is_leaf(ds, q)
        n_leaf += leaf
        n_nonleaf += (not leaf)
        for (p0, c) in ds.node2pos[q]:
            layers = ds._layers_of(p0)
            pos_set = set(ds.node2pos[q])
            for L in range(1, 9):
                per_layer[L].append(len(ds._filter_hard(layers[L], p0, c, pos_set, q)))
    print(f"（leaf query {n_leaf} 个 / non-leaf {n_nonleaf} 个；建层缓存耗时 {time.time() - t0:.1f}s）")
    print(f"{'层':<16}{'均值':>8}{'中位数':>8}{'为0比例':>10}{'最大':>8}")
    for L in range(1, 9):
        v = per_layer[L]
        zero = sum(1 for x in v if x == 0) / len(v)
        print(f"{LAYER_NAME[L]:<16}{statistics.mean(v):>8.1f}{statistics.median(v):>8.0f}"
              f"{100 * zero:>9.1f}%{max(v):>8}")
    return per_layer


def audit(ds, sched_ratio, sched_layers, n=300, k=15):
    print("\n" + "=" * 78)
    print(f"3)+4) 合法性 / 配额审计  ratio={sched_ratio}  layers={sched_layers}  ({n} 个 query)")
    print("=" * 78)
    ds.hard_schedule = [[10 ** 9, sched_ratio, sched_layers]]
    ds.hard_ratio = sched_ratio
    ds.set_epoch(1)

    pool = set(ds.all_edges)
    bad = {'not_in_pool': 0, 'not_in_allemb': 0, 'hit_pos': 0, 'dup': 0, 'count': 0}
    quota = ds._allocate_quotas(int(round(k * sched_ratio)), sched_layers)
    # 「可达成配额」：某层池子为空时该层配额作废，权重在剩下的层里重新归一化
    achievable = {L: 0.0 for L in sched_layers}
    for q in ds.node_list[:n]:
        pos = [random.choice(ds.node2pos[q])]
        negs = ds._get_mixed_negatives(q, k, pos)
        if len(negs) != k:
            bad['count'] += 1
        if len(set(negs)) != len(negs):
            bad['dup'] += 1
        pos_set = set(ds.node2pos[q])
        for (p, c) in negs:
            if (p, c) not in pool:
                bad['not_in_pool'] += 1
            if p not in ds.train_taxon2allemb_id:
                bad['not_in_allemb'] += 1
            if (p, c) in pos_set:
                bad['hit_pos'] += 1
        # 该 query 上层池子非空的那些层
        p0, c = pos[0]
        layers = ds._layers_of(p0)
        nonempty = [L for L in sched_layers
                    if quota.get(L, 0) > 0 and ds._filter_hard(layers[L], p0, c, pos_set, q)]
        denom = sum(quota[L] for L in nonempty)
        if denom > 0:
            for L in nonempty:
                achievable[L] += quota[L] / denom
    print(f"合法性：{bad}   （应全部为 0）")

    st = ds.hard_stats
    total_hard = st['hard']
    print(f"总计 hard={total_hard} rand={st['rand']} shortfall={st['shortfall']}")
    print(f"{'层':<16}{'配额':>6}{'实际占比':>10}{'可达成占比':>12}{'偏差':>8}{'空层次数':>10}")
    tot_ach = sum(achievable.values())
    for L in sched_layers:
        share = st['layer_hist'][L] / total_hard if total_hard else 0
        ach = achievable[L] / tot_ach if tot_ach else 0
        dev = (share - ach) / ach if ach else 0
        print(f"{LAYER_NAME[L]:<16}{quota.get(L, 0):>6}{100 * share:>9.1f}%{100 * ach:>11.1f}%"
              f"{100 * dev:>7.1f}%{st['empty_layer'][L]:>10}")
    illegal_layers = [L for L in st['layer_hist'] if st['layer_hist'][L] and L not in sched_layers]
    print(f"落在 cur_layers 之外的层：{illegal_layers}   （应为空）")
    print(f"shortfall / query 数 = {100 * st['shortfall'] / n:.1f}%  （任务书要求 < 30%）")


def rng_identity(ds, n=50, k=15):
    print("\n" + "=" * 78)
    print("5) RNG 一致性：hard_ratio=0 时与改造前逐位一致")
    print("=" * 78)
    ds.hard_ratio = 0.0
    ds.hard_schedule = []
    ds.set_epoch(1)
    same = 0
    same_rng = 0
    orig = list(ds.all_edges)
    for q in ds.node_list[:n]:
        pos = [random.choice(ds.node2pos[q])]      # 先把正位算出来，别影响下面的 RNG 比对
        # 两次调用都从同一份 all_edges 顺序出发（shuffle 是对当前顺序做置换，必须还原）
        ds.all_edges[:] = orig
        ds.pointer = 0
        random.seed(7)
        a = ds._get_negative_positions(q, k)
        st_a = random.getstate()

        ds.all_edges[:] = orig
        ds.pointer = 0
        random.seed(7)
        b = ds._get_mixed_negatives(q, k, pos)
        st_b = random.getstate()

        same += (a == b)
        same_rng += (st_a == st_b)
    ds.all_edges[:] = orig
    print(f"{same}/{n} 个 query 的负样本序列完全一致（应为 {n}/{n}）")
    print(f"{same_rng}/{n} 个 query 的 RNG 状态完全一致（说明 rho=0 时没有多消耗随机数）")
    print(f"hard_stats（应为全 0）：{ds.hard_stats['hard']} / {ds.hard_stats['rand']}")


def timing(ds, n=200, k=15):
    print("\n" + "=" * 78)
    print(f"6) 负采样耗时（{n} 个 query）")
    print("=" * 78)
    qs = ds.node_list[:n]
    for label, ratio, layers in [("hard=0（baseline）", 0.0, []),
                                 ("hard=0.40 全 8 层", 0.40, list(range(1, 9)))]:
        ds.hard_ratio = ratio
        ds.hard_schedule = [[10 ** 9, ratio, layers]] if layers else []
        ds.set_epoch(1)
        t0 = time.time()
        for q in qs:
            ds._get_mixed_negatives(q, k, [random.choice(ds.node2pos[q])])
        dt = time.time() - t0
        print(f"  {label:<22}: {dt * 1000 / n:.2f} ms / query")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--n_show", type=int, default=20)
    ap.add_argument("--n_stat", type=int, default=1000)
    args = ap.parse_args()

    cfg, ds = build(args.config)
    print(f"dataset={cfg['train_data_loader']['args']['taxonomy_name']}  "
          f"#train_queries={len(ds.node_list)}  #candidates={len(ds.all_edges)}  "
          f"hard_ratio={ds.hard_ratio}  schedule={ds.hard_schedule}")

    show_layers(ds, args.n_show)
    layer_stats(ds, args.n_stat)
    audit(ds, 0.30, [1, 2, 3])
    audit(ds, 0.30, list(range(1, 9)))
    rng_identity(ds)
    timing(ds)


if __name__ == "__main__":
    main()
