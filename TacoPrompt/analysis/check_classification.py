"""对 error_analysis 的分类做抽样人工核验 + 统计 '其他类' 的构成。"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_analysis import Taxonomy, CLASS_NAMES, PSEUDO_LEAF, PSEUDO_ROOT  # noqa: E402

PSEUDO_LEAF_NAME = "pseudo leaf"


def main(dumps, taxo_path, terms_path, n_show=3):
    dump = {"results": {}}
    for p in dumps:
        d = json.load(open(p))
        dump["results"].update(d["results"])
    tx = Taxonomy(taxo_path, terms_path)

    stats = Counter()
    samples = defaultdict(list)
    for q, rec in dump["results"].items():
        if not rec["is_leaf"]:
            continue
        correct_pairs = {(p, c) for p, c in rec["true_positions"]}
        correct_parents = {p for p, _ in rec["true_positions"]}
        correct_children = {c for _, c in rec["true_positions"]}
        for p, c, _s in rec["topk"]:
            if (p, c) in correct_pairs:
                continue
            if not (c in correct_children and p not in correct_parents):
                continue
            # 父错子对
            if p == PSEUDO_ROOT:
                stats["错误节点=pseudo root"] += 1
                cls, x1, x2, mx = 10, None, None, None
            elif p not in tx.nodes:
                stats["错误节点=不在taxo中的孤立节点"] += 1
                cls, x1, x2, mx = 10, None, None, None
            else:
                if any(p0 in tx.nodes and p0 != PSEUDO_ROOT for p0 in correct_parents):
                    stats["错误节点=真实节点"] += 1
                best = None
                for p0 in sorted(correct_parents):
                    if p0 in tx.nodes and p0 != PSEUDO_ROOT:
                        cls_, x1_, x2_, mx_ = tx.classify(p0, p)
                    else:
                        cls_, x1_, x2_, mx_ = 10, None, None, None
                    key = (0 if cls_ != 10 else 1, (x1_ or 0) + (x2_ or 0),
                           -tx.depth.get(p0, 0) if p0 in tx.nodes else 0)
                    if best is None or key < best[0]:
                        best = (key, cls_, x1_, x2_, mx_, p0)
                _k, cls, x1, x2, mx, p0 = best
                if cls == 10 and mx is None:
                    stats["其他类: 与P0无公共祖先(跨顶级树)"] += 1
            if len(samples[cls]) < n_show:
                samples[cls].append((q, sorted(correct_parents), p, c, x1, x2, mx))

    total = sum(v for k, v in stats.items())
    print(f"=== 父错子对（叶子 query）构成，总 {total} ===")
    for k, v in stats.most_common():
        print(f"  {k}: {v} ({100*v/total:.2f}%)")
    print("\n=== 每类抽样 ===")
    for cls in range(1, 11):
        print(f"\n[{cls}] {CLASS_NAMES[cls]}")
        for q, p0s, p, c, x1, x2, mx in samples[cls]:
            print(f"  q={q!r}  P0={p0s}  错误节点p'={p!r}  子c={c!r}  "
                  f"(x1,x2)=({x1},{x2})  Mx={mx!r}")
            if p in tx.nodes:
                p0 = p0s[0]
                print(f"     parents(p')={sorted(tx.parents[p])[:4]}   "
                      f"parents(P0)={sorted(tx.parents[p0])[:4]}")


if __name__ == "__main__":
    main(sys.argv[1:-2], sys.argv[-2], sys.argv[-1])
