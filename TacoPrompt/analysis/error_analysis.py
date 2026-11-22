"""
TacoPrompt hidden 变体：测试集 Top-10 错误分类统计。

输入
----
1. dump_hidden_rank.py 产出的 json（每个 query 的 Top-10 位置 + 正确位置集合）
2. 数据集原始层级 <TAXO>.taxo / <TAXO>.terms（真实 taxonomy，用于计算祖先/后代/兄弟与 LCA）

统计口径（已与用户确认）
--------
* 预测标签 = 位置对 (p, c)；正确标签集合 = node2pos[q]
* 错误样本 = Top-10 中不属于正确位置集合的条目，记为三元组 (q, p′, c)
* 「父错子对」= c 属于 q 的正确子标签集合、但 p′ 不属于正确父标签集合
  · 错误节点 = p′；基准 P₀ = q 的正确父节点（多父时取与 p′ 层级关系最近的那个）
* 不要求「正确位置必须出现在 Top-10 内」
* 叶子 query = node2pos[q] 中所有位置的子标签都是 pseudo leaf（与项目 infer_hidden.py 一致）
* 前置统计项「正确标签为 pseudo_leaf」→ 解释为「正确子标签为 pseudo leaf（即叶子 query）」的占比
* 10 类细分只统计叶子 query 的父错子对样本；层级关系在**原始 taxonomy 全图**上计算
  （core 子图里有跳过未参与训练节点的 jump edge，会扭曲层级距离）

用法
----
python analysis/error_analysis.py \
    -d analysis/results/verb_top10.json \
    -t data/SemEval-Verb/wordnet_verbs.taxo -n data/SemEval-Verb/wordnet_verb.terms \
    -o analysis/results/verb_error_analysis.md --title "Wordnet-Verb"
"""

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

PSEUDO_LEAF = "pseudo leaf"
PSEUDO_ROOT = "pseudo root"

CLASS_NAMES = {
    1: "兄弟节点类(2跳)",
    2: "直接父节点类(1跳)",
    3: "直接子节点类(1跳)",
    4: "二跳父节点类",
    5: "二跳子节点类",
    6: "三跳父节点类",
    7: "三跳子节点类",
    8: "四跳及以上父节点类",
    9: "四跳及以上子节点类",
    10: "其他类",
}


# ---------------------------------------------------------------- taxonomy
class Taxonomy:
    def __init__(self, taxo_path, terms_path):
        id2name = {}
        with open(terms_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                segs = line.split("\t")
                if len(segs) < 2:
                    continue
                id2name[segs[0]] = segs[1]
        self.parents = defaultdict(set)
        self.children = defaultdict(set)
        self.nodes = set()
        with open(taxo_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                segs = line.split("\t")
                if len(segs) < 2:
                    continue
                p = id2name.get(segs[0], segs[0])
                c = id2name.get(segs[1], segs[1])
                if p == c:
                    continue
                self.parents[c].add(p)
                self.children[p].add(c)
                self.nodes.add(p)
                self.nodes.add(c)
        # depth: 从所有入度为 0 的根做 BFS 的最短距离
        self.depth = {}
        q = deque()
        for n in self.nodes:
            if not self.parents[n]:
                self.depth[n] = 0
                q.append(n)
        while q:
            n = q.popleft()
            for ch in self.children[n]:
                if ch not in self.depth:
                    self.depth[ch] = self.depth[n] + 1
                    q.append(ch)
        for n in self.nodes:            # 理论上不会残留（DAG 有环时兜底）
            self.depth.setdefault(n, 0)
        self._up_cache = {}
        self._down_cache = {}

    def up_dist(self, node, max_d=None):
        """node 向上 BFS，返回 {祖先: 距离}（含 node 自身，距离 0）。默认不截断深度。"""
        if node in self._up_cache:
            return self._up_cache[node]
        res = {node: 0}
        frontier = [node]
        d = 0
        while frontier and (max_d is None or d < max_d):
            d += 1
            nxt = []
            for n in frontier:
                for p in self.parents[n]:
                    if p not in res:
                        res[p] = d
                        nxt.append(p)
            frontier = nxt
        self._up_cache[node] = res
        return res

    def down_dist(self, node, max_d=None):
        if node in self._down_cache:
            return self._down_cache[node]
        res = {node: 0}
        frontier = [node]
        d = 0
        while frontier and (max_d is None or d < max_d):
            d += 1
            nxt = []
            for n in frontier:
                for c in self.children[n]:
                    if c not in res:
                        res[c] = d
                        nxt.append(c)
            frontier = nxt
        self._down_cache[node] = res
        return res

    def classify(self, p0, err):
        """返回 (class_id, x1, x2, mx)：p0→mx 距离 x1，err→mx 距离 x2。

        优先级：祖先/后代（父/子类）> 兄弟 > 其他。
        """
        up0 = self.up_dist(p0)
        if err in up0 and err != p0:                      # err 是 p0 的祖先
            d = up0[err]
            cls = 2 if d == 1 else (4 if d == 2 else (6 if d == 3 else 8))
            return cls, d, 0, err
        down0 = self.down_dist(p0)
        if err in down0 and err != p0:                    # err 是 p0 的后代
            d = down0[err]
            cls = 3 if d == 1 else (5 if d == 2 else (7 if d == 3 else 9))
            return cls, 0, d, p0
        # 兄弟：存在共同父节点
        common_parents = self.parents[p0] & self.parents[err]
        if common_parents:
            mx = max(common_parents, key=lambda x: (self.depth.get(x, 0), x))
            return 1, 1, 1, mx
        # 其他：最近公共祖先
        up_err = self.up_dist(err)
        commons = set(up0.keys()) & set(up_err.keys())
        if not commons:
            return 10, None, None, None
        mx = max(commons, key=lambda x: (self.depth.get(x, 0), -(up0[x] + up_err[x]), x))
        return 10, up0[mx], up_err[mx], mx


# ---------------------------------------------------------------- analysis
def pct(a, b):
    return 0.0 if b == 0 else 100.0 * a / b


PAPER_TABLE1 = {
    "MeSH": {
        "total": {"MR": 49.140, "MRR": 0.674, "R@1": 17.9, "R@5": 42.4, "R@10": 55.9,
                  "H@1": 40.7, "H@5": 74.6, "H@10": 84.6},
        "leaf": {"MRR": 0.868, "H@5": 79.0, "R@10": 81.5},
        "nonleaf": {"MRR": 0.554, "H@5": 65.1, "R@10": 40.2},
    },
    "WordNet-Verb": {
        "total": {"MR": 436.799, "MRR": 0.557, "R@1": 18.3, "R@5": 36.9, "R@10": 46.5,
                  "H@1": 28.0, "H@5": 52.3, "H@10": 62.5},
        "leaf": {"MRR": 0.762, "H@5": 56.5, "R@10": 65.8},
        "nonleaf": {"MRR": 0.370, "H@5": 35.2, "R@10": 25.3},
    },
}


def paper_alignment(recs, paper):
    """返回与论文 Table 1（Ours-Hidden 行）的对齐表行（markdown）。"""
    import math
    m_all = compute_metrics(recs)
    m_leaf = compute_metrics([r for r in recs if r["is_leaf"]])
    m_non = compute_metrics([r for r in recs if not r["is_leaf"]])
    rows = [
        ("Total", "MR↓", paper["total"]["MR"], m_all["macro_mr"]),
        ("Total", "MRR", paper["total"]["MRR"], m_all["mrr_scaled_10"]),
        ("Total", "R@1", paper["total"]["R@1"], 100 * m_all["recall@1"]),
        ("Total", "R@5", paper["total"]["R@5"], 100 * m_all["recall@5"]),
        ("Total", "R@10", paper["total"]["R@10"], 100 * m_all["recall@10"]),
        ("Total", "H@1", paper["total"]["H@1"], 100 * m_all["hit@1"]),
        ("Total", "H@5", paper["total"]["H@5"], 100 * m_all["hit@5"]),
        ("Total", "H@10", paper["total"]["H@10"], 100 * m_all["hit@10"]),
        ("Leaf", "MRR", paper["leaf"]["MRR"], m_leaf["mrr_scaled_10"]),
        ("Leaf", "H@5", paper["leaf"]["H@5"], 100 * m_leaf["hit@5"]),
        ("Leaf", "R@10", paper["leaf"]["R@10"], 100 * m_leaf["recall@10"]),
        ("Non-leaf", "MRR", paper["nonleaf"]["MRR"], m_non["mrr_scaled_10"]),
        ("Non-leaf", "H@5", paper["nonleaf"]["H@5"], 100 * m_non["hit@5"]),
        ("Non-leaf", "R@10", paper["nonleaf"]["R@10"], 100 * m_non["recall@10"]),
    ]
    L = ["| 分组 | 指标 | 论文 Table 1 (Ours-Hidden) | 本次复现 | 差值 |",
         "|---|---|---|---|---|"]
    for grp, k, pv, ov in rows:
        nd = 3 if k in ("MR↓", "MRR") else 1
        L.append(f"| {grp} | {k} | {pv:.{nd}f} | {ov:.{nd}f} | {ov - pv:+.{nd}f} |")

    # 排名分布（解释 MR 的数值差异：MR 由重尾主导）
    ranks = sorted(r for rec in recs for r in rec.get("ranks", []))
    if ranks:
        n = len(ranks)
        q = lambda p: ranks[min(n - 1, int(p * n))]
        L += ["", "真实位置排名分布（用于解释 MR 的波动 —— MR 由重尾主导）：", "",
              "| 统计量 | 数值 |", "|---|---|",
              f"| 中位数 | {q(0.5)} |", f"| 90 分位 | {q(0.9)} |",
              f"| 99 分位 | {q(0.99)} |", f"| 最大 | {ranks[-1]} |",
              f"| macro 均值（每 query 先平均 = 论文 MR 口径） | {sum(ranks)/n:.1f}（按 query 平均为 "
              f"{sum(sum(rec['ranks'])/len(rec['ranks']) for rec in recs if rec.get('ranks'))/max(1,len(recs)):.1f}） |",
              f"| micro 均值（全部真实位置直接平均） | {sum(ranks)/n:.1f} |",
              f"| micro 均值去掉 top-1% 极端排名后 | {sum(ranks[:int(n*0.99)])/max(1,int(n*0.99)):.1f} |"]
    return L


def compute_metrics(recs):
    """用 dump 中存下的真实位置完整排名复算项目里的指标（口径同 model/metric.py）。"""
    if not recs:
        return None
    all_ranks = [r for rec in recs for r in rec.get("ranks", [])]
    per_query = [rec.get("ranks", []) for rec in recs]
    if not all_ranks:
        return None
    n_rank = len(all_ranks)
    n_q = len(per_query)
    import math

    def hit_at(k):
        return sum(1 for rs in per_query if any(r <= k for r in rs)) / n_q

    def recall_at(k):
        return sum(1 for r in all_ranks if r <= k) / n_rank

    return {
        "hit@1": hit_at(1), "hit@5": hit_at(5), "hit@10": hit_at(10),
        "recall@1": recall_at(1), "recall@5": recall_at(5), "recall@10": recall_at(10),
        "mrr": sum(1.0 / r for r in all_ranks) / n_rank,
        "mrr_scaled_10": sum(1.0 / math.ceil(r / 10) for r in all_ranks) / n_rank,
        "macro_mr": sum((sum(rs) / len(rs)) for rs in per_query if rs) / n_q,
        "micro_mr": sum(all_ranks) / n_rank,
    }


def analyse(dump, taxo, title):
    results = dump["results"]
    n_query = len(results)
    n_correct = n_err = 0
    err_types = Counter()                  # 父错子对 / 父对子错 / 双错
    n_pcp = 0                              # 父错子对错误条目数
    parents_fail_both = 0
    # 父错子对明细
    pcp_leaf = 0                           # 叶子 query 的父错子对条目
    pcp_nonleaf = 0
    class_cnt = Counter()
    class_dist = defaultdict(Counter)      # cls -> Counter((x1,x2))
    class_mx = defaultdict(Counter)        # cls -> Counter(mx)
    multi_parent_samples = 0
    err_node_types = Counter()             # 错误节点是真实节点/pseudo root
    query_level_leaf = 0
    hit10_correct = 0                      # 正确位置进 top10 的 query 数
    other_reasons = Counter()
    c10_reason = Counter()

    for q, rec in results.items():
        true_pos = [tuple(x) for x in rec["true_positions"]]
        correct_pairs = set(true_pos)
        correct_parents = {p for p, _ in true_pos}
        correct_children = {c for _, c in true_pos}
        is_leaf = bool(rec["is_leaf"])
        query_level_leaf += is_leaf
        if correct_pairs & {tuple([t[0], t[1]]) for t in rec["topk"]}:
            hit10_correct += 1

        for p, c, _score in rec["topk"]:
            if (p, c) in correct_pairs:
                n_correct += 1
                continue
            n_err += 1
            if c in correct_children and p not in correct_parents:
                err_types["父错子对"] += 1
                n_pcp += 1
                if is_leaf:
                    pcp_leaf += 1
                else:
                    pcp_nonleaf += 1
                    continue        # 10 类细分只统计叶子 query
                # ---- 选基准 P₀ 并分类 ----
                if p == PSEUDO_ROOT or p not in taxo.nodes:
                    if p == PSEUDO_ROOT:
                        c10_reason["错误节点 = pseudo root"] += 1
                    else:
                        c10_reason["错误节点为孤立节点(不在 .taxo 中)"] += 1
                    err_node_types["非真实节点(pseudo root 等)"] += 1
                    class_cnt[10] += 1
                    class_dist[10][(None, None)] += 1
                    class_mx[10]["(无)"] += 1
                    continue
                err_node_types["真实节点"] += 1
                if len(correct_parents) > 1:
                    multi_parent_samples += 1
                best = None
                for p0 in sorted(correct_parents):
                    if p0 == PSEUDO_ROOT or p0 not in taxo.nodes:
                        cls, x1, x2, mx = 10, None, None, None
                    else:
                        cls, x1, x2, mx = taxo.classify(p0, p)
                    key = (0 if cls != 10 else 1,
                           (x1 or 0) + (x2 or 0),
                           -taxo.depth.get(p0, 0) if p0 in taxo.nodes else 0)
                    if best is None or key < best[0]:
                        best = (key, cls, x1, x2, mx, p0)
                _key, cls, x1, x2, mx, p0 = best
                class_cnt[cls] += 1
                class_dist[cls][(x1, x2)] += 1
                class_mx[cls][mx if mx else "(无公共祖先)"] += 1
                if cls == 10:
                    if mx is None:
                        c10_reason["与 P₀ 无公共祖先（跨顶级树）"] += 1
                    else:
                        c10_reason["有公共祖先但非 1-9 类（远亲/跨分支）"] += 1
            elif p in correct_parents and c not in correct_children:
                err_types["父对子错"] += 1
            else:
                err_types["父子皆错"] += 1

    # ---------------- 生成报告 ----------------
    L = []
    A = L.append
    A(f"# {title} — TacoPrompt hidden 变体 Top-10 错误分类统计")
    A("")
    A(f"- 数据集：`{title}`；测试 query 数：**{n_query}**（其中叶子 query {query_level_leaf} 个，"
      f"占 {pct(query_level_leaf, n_query):.2f}%）")
    A(f"- 候选位置总数：{dump.get('num_candidates')}；每个 query 取 Top-10，共 {n_query * 10} 个预测条目")
    A(f"- 其中正确位置 {n_correct} 条（{pct(n_correct, n_query * 10):.2f}%），"
      f"错误位置 **{n_err}** 条（{pct(n_err, n_query * 10):.2f}%）")
    A(f"- 正确位置出现在 Top-10 的 query：{hit10_correct}（{pct(hit10_correct, n_query):.2f}%）")
    A("")
    A("## 零、推理口径校验（复算项目指标）")
    A("")
    m_all = compute_metrics(list(results.values()))
    m_leaf = compute_metrics([r for r in results.values() if r["is_leaf"]])
    m_non = compute_metrics([r for r in results.values() if not r["is_leaf"]])
    if m_all:
        keys = ["hit@1", "hit@5", "hit@10", "recall@1", "recall@5", "recall@10",
                "mrr", "mrr_scaled_10", "macro_mr", "micro_mr"]
        A("| 指标 | 全部 query | 叶子 query | 非叶子 query |")
        A("|---|---|---|---|")
        for k in keys:
            v1 = f"{m_all[k]:.4f}" if m_all else "-"
            v2 = f"{m_leaf[k]:.4f}" if m_leaf else "-"
            v3 = f"{m_non[k]:.4f}" if m_non else "-"
            A(f"| {k} | {v1} | {v2} | {v3} |")
        A("")
        A("（应与训练日志中的 `test_*` / `test_leaf_*` / `test_nonleaf_*` 一致）")
        A("")
    paper_key = None
    for k in PAPER_TABLE1:
        if k.lower() in title.lower() or (k == "WordNet-Verb" and "verb" in title.lower()) or \
           (k == "MeSH" and "mesh" in title.lower()):
            paper_key = k
            break
    if paper_key:
        A(f"### 与论文 Table 1（`Ours-Hidden` 行）对齐（{paper_key}）")
        A("")
        L.extend(paper_alignment(list(results.values()), PAPER_TABLE1[paper_key]))
        A("")
    A("## 一、全局错误大类占比")
    A("")
    A("| 错误大类 | 条目数 | 占全部错误 |")
    A("|---|---|---|")
    for k in ["父错子对", "父对子错", "父子皆错"]:
        A(f"| {k} | {err_types.get(k, 0)} | {pct(err_types.get(k, 0), n_err):.2f}% |")
    A(f"| **合计** | **{n_err}** | 100.00% |")
    A("")
    A(f"**父错子对（父子插入类错误）占全部错误的比例：{pct(n_pcp, n_err):.2f}%**")
    A("")
    A("## 二、父错子对前置统计项")
    A("")
    A(f"- 父错子对总错误条目：**{n_pcp}**")
    A(f"- 其中 query 为叶子（正确子标签 = pseudo leaf）的条目：{pcp_leaf}，"
      f"占父错子对总错误的 **{pct(pcp_leaf, n_pcp):.2f}%**")
    A(f"- 其中 query 为非叶子（正确子标签为真实节点）的条目：{pcp_nonleaf}，"
      f"占 {pct(pcp_nonleaf, n_pcp):.2f}%（不参与下面的 10 类细分）")
    A(f"- 多父 query 样本数：{multi_parent_samples}；错误节点类型分布："
      + "，".join(f"{k} {v}" for k, v in err_node_types.items()))
    A("")
    A("## 三、10 类细分错误占比（仅叶子 query 的父错子对样本）")
    A("")
    A(f"叶子子集样本数 = **{pcp_leaf}**")
    A("")
    A("| # | 类别 | 条目数 | 占叶子子集父错子对 | 占父错子对总错误 | 占全部错误 |")
    A("|---|---|---|---|---|---|")
    for cls in range(1, 11):
        c = class_cnt.get(cls, 0)
        A(f"| {cls} | {CLASS_NAMES[cls]} | {c} | {pct(c, pcp_leaf):.2f}% | "
          f"{pct(c, n_pcp):.2f}% | {pct(c, n_err):.2f}% |")
    A(f"| - | **合计** | **{sum(class_cnt.values())}** | "
      f"{pct(sum(class_cnt.values()), pcp_leaf):.2f}% | "
      f"{pct(sum(class_cnt.values()), n_pcp):.2f}% | - |")
    A("")
    A("## 四、各类别的 LCA 距离分布（x₁ = P₀→Mx 跳数，x₂ = 错误节点→Mx 跳数）")
    A("")
    for cls in range(1, 11):
        c = class_cnt.get(cls, 0)
        if c == 0:
            A(f"### {cls}. {CLASS_NAMES[cls]}：0 条（跳过）")
            A("")
            continue
        A(f"### {cls}. {CLASS_NAMES[cls]}：{c} 条（占叶子子集 {pct(c, pcp_leaf):.2f}%）")
        A("")
        A("| (x₁, x₂) | 条目数 | 占比 |")
        A("|---|---|---|")
        for (x1, x2), k in class_dist[cls].most_common():
            A(f"| ({x1}, {x2}) | {k} | {pct(k, c):.2f}% |")
        A("")
        A("Mx 分组（Top-10 最近公共祖先）：")
        A("")
        A("| Mx | 条目数 | 占比 |")
        A("|---|---|---|")
        for mx, k in class_mx[cls].most_common(10):
            A(f"| {mx} | {k} | {pct(k, c):.2f}% |")
        A(f"| （共 {len(class_mx[cls])} 个不同 Mx） | {c} | 100.00% |")
        A("")
        if cls == 10 and c10_reason:
            A("第 10 类（其他类）的构成：")
            A("")
            A("| 子类 | 条目数 | 占其他类 | 占叶子子集父错子对 |")
            A("|---|---|---|---|")
            for k, v in c10_reason.most_common():
                A(f"| {k} | {v} | {pct(v, c):.2f}% | {pct(v, pcp_leaf):.2f}% |")
            A("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dump", required=True, nargs="+", help="可传多个分片 json，会合并")
    ap.add_argument("-t", "--taxo", required=True)
    ap.add_argument("-n", "--terms", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    dump = None
    for path in args.dump:
        with open(path) as f:
            d = json.load(f)
        if dump is None:
            dump = dict(d)
            dump["results"] = dict(d["results"])
        else:
            dump["results"].update(d["results"])
        print(f"loaded {path}: {len(d['results'])} queries")
    taxo = Taxonomy(args.taxo, args.terms)
    print(f"taxonomy: {len(taxo.nodes)} nodes, "
          f"{sum(len(v) for v in taxo.children.values())} edges")
    report = analyse(dump, taxo, args.title or Path(args.taxo).stem)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
