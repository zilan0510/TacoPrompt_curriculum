# TacoPrompt 父节点负采样改造 · 实现任务书

> 约束：**只改父节点的负采样**，不动模型结构、不动评测口径、不动子节点采样。

---

## 0. 一句话任务

在 `TacoPrompt/data_loader/dataset.py` 里，把「全局洗牌 + 游标切段」的纯随机负采样，改成 **按父节点与正确父的图距离分层采样的课程学习**；分层依据来自两份错误分析中「父错子对」的第 1~9 类。

---

## 1. 数据依据（两份错误分析的结论）

### 1.1 错误大类分布

| 数据集 | 父错子对 | 父对子错 | 父子皆错 | 父错子对占比 |
|---|---|---|---|---|
| WordNet-Verb（1000 query，804 leaf） | 6916 | 442 | 1946 | **74.33%** |
| MeSH（819 query，558 leaf） | 4402 | 931 | 1835 | **61.41%** |

→ 支持「先只改父节点」这个决定。

### 1.2 父错子对里，叶子 query 占绝对多数

| 数据集 | 父错子对总数 | 其中 leaf query | 占比 |
|---|---|---|---|
| WordNet-Verb | 6916 | 6738 | **97.43%** |
| MeSH | 4402 | 3901 | **88.62%** |

→ 改造只需要重点保证 **leaf query** 这条路径有效；非叶 query 是次要目标。

### 1.3 第 1~9 类（可归因）的分布

以「叶子 query 的父错子对」为分母：

| # | 类别 | (x₁,x₂) | Verb 条目 | Verb 占比 | MeSH 条目 | MeSH 占比 |
|---|---|---|---|---|---|---|
| 1 | 兄弟节点 | (1,1) | 271 | 4.02% | 248 | 6.36% |
| 2 | 直接父 | (1,0) | 174 | 2.58% | 310 | 7.95% |
| 3 | 直接子 | (0,1) | 498 | 7.39% | 360 | 9.23% |
| 4 | 二跳父 | (2,0) | 62 | 0.92% | 130 | 3.33% |
| 5 | 二跳子 | (0,2) | 152 | 2.26% | 106 | 2.72% |
| 6 | 三跳父 | (3,0) | 18 | 0.27% | 45 | 1.15% |
| 7 | 三跳子 | (0,3) | 30 | 0.45% | 18 | 0.46% |
| 8 | 四跳+父 | (4+,0) | 4 | 0.06% | 20 | 0.51% |
| 9 | 四跳+子 | (0,4+) | 22 | 0.33% | 5 | 0.13% |
| **1~9 合计** | | | **1231** | **18.27%** | **1242** | **31.84%** |
| 10 | 其他（无公共祖先等） | — | 5507 | 81.73% | 2659 | 68.16% |

其中 x₁ = 「正确父 P₀ → 最近公共祖先 Mx」的跳数，x₂ = 「错误父 → Mx」的跳数。

### 1.4 为什么可以忽略第 10 类

- 第 10 类是「其他」，其中 60~77% 是「与 P₀ 无任何公共祖先（跨顶级树）」，13~22% 是「错误节点 = pseudo root」。这两类**没有可定位的图结构特征**，无法针对性设计采样。
- 1~9 类虽然只占 18~32%，但它们是**有明确目标的错误样本**：错误父与正确父之间存在确定的图距离。这类错误能被针对性训练排除，收益上限更高。
- 第 10 类占比高是**分母效应**：全树其他节点在数量上占绝大多数（WordNet-Verb 51159 个候选位置里，与 P₀ 无公共祖先的远超一万），随机采样天然会命中大量简单负例。这不代表模型「主要在犯低级错误」。

---

## 2. 难度分层（核心设计）

### 2.1 分层定义

以正例 `(P₀, c)` 为参照，把候选父 `p'` 按图关系分成 7 层：

| 层 | 含义 | 对应错误分析类别 | 图操作（`self.node2parents` / `self.node2children` 现成的） |
|---|---|---|---|
| **L1** | P₀ 的直接父 | 第 2 类 (1,0) | `node2parents[P0]` |
| **L2** | P₀ 的直接子 | 第 3 类 (0,1) | `node2children[P0]` |
| **L3** | P₀ 的兄弟 | 第 1 类 (1,1) | `∪ node2children[pp] for pp in node2parents[P0]`，去掉 P₀ |
| **L4** | 二跳父（祖父） | 第 4 类 (2,0) | `∪ node2parents[pp] for pp in node2parents[P0]` |
| **L5** | 二跳子（孙） | 第 5 类 (0,2) | `∪ node2children[ch] for ch in node2children[P0]` |
| **L6** | 三跳父 | 第 6 类 (3,0) | 对 L4 再上溯一层 |
| **L7** | 三跳子 | 第 7 类 (0,3) | 对 L5 再下钻一层 |
| **L8** | 四跳及以上（父或子） | 第 8、9 类 | 在候选池里但不在 L1~L7 |
| **L9** | 其他（无公共祖先 / pseudo root） | 第 10 类 | 兜底，**不主动采** |

难度从 L1 到 L9 单调递减：**L1/L2/L3 是最难也是最有信息量的负例**。

因此课程学习的开放顺序必须**由远及近**（L6~L8 先开，L1 最后开），详见 §4。

### 2.2 分层必须满足的合法性

候选父 `p'` 要能替换 P₀ 而不破坏「位置」的定义，必须同时满足：

1. **`p' != P0`**
2. **`p'` 必须是 `c` 的严格祖先** —— 否则 `(p', c)` 根本不在候选池 `all_edges` 里
   - 对 **leaf query**（`c is pseudo_leaf`）：TacoPrompt 把**每个节点都直连了 pseudo_leaf**，所以 `ancestors(pseudo_leaf) = 全图节点` → 任意 `p'` 合法，池子 = 11937 个（实测）
   - 对 **非 leaf query**（`c` 是真子）：`p'` 只能从 `nx.ancestors(core_subgraph, c)` 里取，池子小得多
3. **`(p', c) not in node2pos[q]`** —— 不能撞上该 query 的其它正例位置
4. **`p'` 在 `train_taxon2allemb_id` 里** —— 否则拼 prompt 时查不到描述


### 2.3 各层的实际池子大小（实现时必须先打印出来）

对 WordNet-Verb 的 leaf query：
- L1（直接父）：1~3 个
- L2（直接子）：0~hundreds（`grow||grow.v.02` 有 39 个）
- L3（兄弟）：2~50 个（视父的扇出）
- L4/L5：数量不定
- L6~L8：通常个位数

**必须在实现后加一段诊断打印**，统计 1000 个 leaf query 各层的平均可用数量。如果某一层平均 < 2 个，就走 §3.4 的降级（层内放回采样 / 整层为空时配额作废）。

---

## 3. 采样策略

### 3.1 总体配额

设 `negative_size = k`（config 里是 15），`hard_ratio = ρ`（课程控制，见 §4）：

```
n_hard = round(k * ρ)             # 名义配额：结构化难负样本数量
n_rand = k - len(hard_negs)      # 实际剩余：某层池子为空时配额作废，转给随机部分
```

### 3.2 hard 内部的层间配额（按错误占比加权）

`n_hard` 个难负例**不是**在各层之间平均分配，而是按两份错误分析里各层的实际错误占比分配。取 Verb 与 MeSH 的均值作为权重：

| 层 | Verb 占比 | MeSH 占比 | 平均（权重 w） | 8 层归一化 |
|---|---|---|---|---|
| L1 直接父 | 2.58% | 7.95% | 5.27 | **21.0%** |
| L2 直接子 | 7.39% | 9.23% | 8.31 | **33.1%** |
| L3 兄弟 | 4.02% | 6.36% | 5.19 | **20.7%** |
| L4 二跳父 | 0.92% | 3.33% | 2.13 | 8.5% |
| L5 二跳子 | 2.26% | 2.72% | 2.49 | 9.9% |
| L6 三跳父 | 0.27% | 1.15% | 0.71 | 2.8% |
| L7 三跳子 | 0.45% | 0.46% | 0.46 | 1.8% |
| L8 四跳+ | 0.39% | 0.64% | 0.52 | 2.1% |
| 合计 | — | — | **25.08** | 100% |

**为什么按错误占比加权，而不是等概率**：
- 错误占比本身就是「这个层有多容易被模型搞错」的经验估计，直接拿来当采样权重最省事；
- 层内数量差异是数量级的（L2 可能有 39 个，L1 常常只有 1 个），如果按候选级等概率抽，**最难的 L1 会被 L2 完全淹没**；
- 加权能保证 L1（直接父，候选最少但最难）一定拿到约 1/5 的配额。

**关键规则：权重只在「当前课程允许的层子集」内重新归一化。**

例如阶段 1 只开放 L6/L7/L8，则：

$$w'_{6} = \frac{0.71}{0.71+0.46+0.52}=0.42,\quad
w'_{7}=0.27,\quad w'_{8}=0.31$$

而不是拿全 8 层的归一化值直接截断（那样三层加起来只有 6.7%，配额会几乎全是 0）。

**配额拆分用最大余数法**（保证各层配额之和恰好等于 `n_hard`）：

```python
LAYER_WEIGHTS = {1: 5.27, 2: 8.31, 3: 5.19, 4: 2.13, 5: 2.49, 6: 0.71, 7: 0.46, 8: 0.52}

def _allocate_quotas(self, n_hard, layers):
    ws = {l: LAYER_WEIGHTS[l] for l in layers}
    total = sum(ws.values())
    raw = {l: n_hard * w / total for l, w in ws.items()}
    quota = {l: int(raw[l]) for l in ws}
    rest = n_hard - sum(quota.values())
    for l in sorted(ws, key=lambda x: raw[x] - quota[x], reverse=True)[:rest]:
        quota[l] += 1
    return quota
```

举例：`n_hard = 5`，开放 L1~L8 → 配额约 `{L1:1, L2:2, L3:1, L5:1}`（其余 0）。

### 3.3 层内怎么抽

层内**均匀随机、不放回**（`random.sample`）。若该层池子不足配额，则允许放回补齐（`random.choices`）并记 `shortfall`。

**该层池子为空时不跨界补**，把作废的配额还给随机负例（见 §3.4），这样：
- 不会因为 L1 常常为空就把配额挪给 L8（那会污染难度分层）；
- `negative_size` 总数仍然恒定。

### 3.4 池子不足 / 为空的降级

```
1) 层内候选数 < 该层配额  → random.choices 放回补齐，记 shortfall += 1
2) 该层候选数为 0        → 配额作废、转给随机负例，记 empty_layer[layer] += 1
3) 绝不允许为了凑数去「第 9 层（其他类）」或其它未开放的层里补
   —— 那会污染难度分层，等于把结构化采样退化成随机采样
```

### 3.5 随机部分保持不变

`n_rand = negative_size - len(hard_negs)`（用实际拿到的 hard 数量，不是名义配额），仍然走原来的 `_get_exactly_k_negatives`（全局池 + 游标），并传入 `ignore=hard_negs` 去重。

这样 `negative_size` 恒定，且「简单负例」的分布不被破坏。

---

## 4. 课程调度（由远及近）

### 4.1 为什么是「由远及近」

L1~L8 按迷惑性排序是 **L1 最难、L8 最易**（错误分析里的「难度倍率」：1 跳负例被排进 top-10 的概率是随机候选的几十倍，4 跳以上则接近随机）。课程学习必须**从易到难**，所以层的开放顺序必须是 **L6→L8 先开，L1 最后开**。

> ⚠️ 早期版本的任务书写成「L1,L2,L3 先开」，那是**反向课程**，已修正。

### 4.2 课程表

TacoPrompt 的验证从 **epoch 40** 才开始（`if self.do_validation and epoch >= 40`），所以阶段切换要在 40 附近就完成到「已包含 1 跳」的状态，否则第一次验证看不到改造效果。

| 阶段 | epoch 区间 | `hard_ratio` ρ | 开放的层 | 难度定位 |
|---|---|---|---|---|
| 0 · warmup | 1 – 10 | 0.00 | — | 纯随机，先学会「父是什么」 |
| 1 · 远亲 | 11 – 20 | 0.15 | **L6, L7, L8** | 3~4 跳，与随机分布最接近，扰动最小 |
| 2 · 二跳 | 21 – 32 | 0.25 | **L4 ~ L8** | 收到 2 跳 |
| 3 · 一跳（子/兄弟） | 33 – 45 | 0.35 | **L2 ~ L8** | 引入 1 跳里较易的两类 |
| 4 · 直接父 | 46 – 100 | 0.40 | **L1 ~ L8** | 最后才放最难的直接父 |

### 4.3 设计理由

- **前 10 个 epoch 完全随机**：模型此时还分不清父的概念，直接喂兄弟节点会让 loss 无意义地波动。
- **每次只向「更近一层」收缩**：L6~L8 → L4 → L2 → L1，对应难度倍率从「接近随机」到「几十倍」。
- **L1（直接父）最后开**：它是唯一一个「候选经常只有 1 个」的层（单父节点只有 1 个父），过早引入会让每个 batch 反复看同一个负例。
- **epoch 40 落在阶段 3**：首次验证时已经吃掉 1 跳的子/兄弟，能看出效果；又还没上 L1，loss 不会炸。
- **ρ 封顶 0.40**：开放全部 8 层后，L1~L3 的候选池平均只有 10~50 个，ρ 再高就只能反复采样同几个节点。

### 4.4 两个可单独配置的开关

- `hard_schedule` 里的层列表：可把课程钉死在某一档做消融（例如只开 L1~L3），见 §7。
- `hard_ratio = 0.0`：完全关闭，行为与论文一致（用于跑 baseline）。

**做「只验证学姐假设」的对照实验时**：直接给 `hard_schedule = [[100, 0.30, [1,2,3]]]` —— 不设课程，全程固定 1 跳、固定比例。

---

## 5. 代码改动清单

> 只改 4 个文件，按顺序执行。**每改完一个文件必须 `python -m py_compile <file>` 通过。**

### 5.1 `data_loader/dataset.py`（主要改动）

#### (a) `RawDataset.__init__` 末尾新增状态

```python
# ===== 结构化父节点负采样（课程学习）=====
# 依据：错误分析里「父错子对」的 1~9 类（详见 _code_reading/TacoPrompt_父节点负采样改造_Prompt.md §2）
# 第 10 类（无公共祖先 / pseudo root）无法定位结构特征，不主动采样。
self.hard_ratio = hard_ratio                     # 由 config 传入，默认 0.0 = 关闭（保持原行为）
self.hard_schedule = hard_schedule               # [(end_epoch, ratio, [layer, ...]), ...]
self.cur_epoch = 1
self.cur_hard_ratio = 0.0
self.cur_layers = []                             # 当前课程开放的层，例如 [6, 7, 8]
self._hard_pool_cache = {}                       # query -> {(p0,c): {layer: [p, ...]}}
self.hard_stats = {
    'hard': 0,
    'rand': 0,
    'shortfall': 0,                              # 某层池子不足配额、被迫放回的次数
    'empty_layer': {i: 0 for i in range(1, 9)},  # 某层池子为空的次数
    'layer_hist': {i: 0 for i in range(1, 9)},   # 实际采到的各层数量
}
```

#### (b) 新增 `set_epoch`

```python
def set_epoch(self, epoch):
    """由 trainer 在每个 epoch 开头调用，推进课程阶段（由远及近）。"""
    self.cur_epoch = epoch
    self.cur_hard_ratio = 0.0
    self.cur_layers = []
    for end_epoch, ratio, layers in self.hard_schedule:
        if epoch <= end_epoch:
            self.cur_hard_ratio = ratio
            self.cur_layers = list(layers)
            break
    else:
        # 超出最后一段：沿用最后一段的配置
        if self.hard_schedule:
            _, self.cur_hard_ratio, layers = self.hard_schedule[-1]
            self.cur_layers = list(layers)

    # 兜底：没配 schedule 但给了 hard_ratio → 全程固定比例 + 全 8 层（方便快速跑通）
    if not self.hard_schedule and self.hard_ratio > 0:
        self.cur_hard_ratio = self.hard_ratio
        self.cur_layers = list(range(1, 9))

    if not self.cur_layers:          # warmup 阶段：层列表为空 → 强制关掉 hard
        self.cur_hard_ratio = 0.0

    print(f'[hard-neg] epoch {epoch}: ratio={self.cur_hard_ratio}, layers={self.cur_layers}')
```

#### (c) 新增 `_classify_parent`

```python
def _classify_parent(self, p0, p_prime):
    """返回 p_prime 相对正确父 p0 的难度层（1..8），不在 L1~L8 则返回 None。"""
    if p_prime == p0:
        return 0

    # L1 直接父 / L2 直接子
    if p_prime in self.node2parents.get(p0, set()):
        return 1
    if p_prime in set(self.node2children.get(p0, [])):
        return 2

    # L3 兄弟：所有父的孩子并集，去掉自己
    sib = set()
    for pp in self.node2parents.get(p0, set()):
        sib.update(self.node2children.get(pp, []))
    sib.discard(p0)
    if p_prime in sib:
        return 3

    # L4 二跳父
    gp = set()
    for pp in self.node2parents.get(p0, set()):
        gp.update(self.node2parents.get(pp, set()))
    if p_prime in gp:
        return 4

    # L5 二跳子
    gc = set()
    for ch in self.node2children.get(p0, []):
        gc.update(self.node2children.get(ch, []))
    if p_prime in gc:
        return 5

    # L6 三跳父
    ggp = set()
    for x in gp:
        ggp.update(self.node2parents.get(x, set()))
    if p_prime in ggp:
        return 6

    # L7 三跳子
    ggc = set()
    for x in gc:
        ggc.update(self.node2children.get(x, []))
    if p_prime in ggc:
        return 7

    # L8 四跳及以上：在候选池里但不在上面任何一层
    return 8
```

#### (d) 新增 `_build_hard_pool`

```python
def _build_hard_pool(self, query_node):
    """
    为 query 的【每个正例位置】建一份分层候选父池。
    返回：{ (p0, c): {layer: [p_prime, ...]} }

    合法性（缺一不可）：
      1) p_prime != p0
      2) p_prime 是 c 的严格祖先（否则 (p_prime, c) 不在 all_edges 里）
      3) (p_prime, c) not in node2pos[query_node]
      4) p_prime 在 train_taxon2allemb_id 里
    """
    if query_node in self._hard_pool_cache:
        return self._hard_pool_cache[query_node]

    all_pool = set(self.all_edges)
    pos_set = set(self.node2pos[query_node])
    ancestors_cache = {}

    def ancestors_of(node):
        if node not in ancestors_cache:
            ancestors_cache[node] = nx.ancestors(self.core_subgraph, node)
        return ancestors_cache[node]

    result = {}
    for (p0, c) in self.node2pos[query_node]:
        if c is self.pseudo_leaf_node:
            # leaf query：TacoPrompt 把所有节点都直连 pseudo_leaf，所以全图都是合法父
            candidates = list(self.core_subgraph.nodes())
        else:
            candidates = list(ancestors_of(c))

        buckets = {i: [] for i in range(1, 9)}
        for p_prime in candidates:
            if p_prime == p0:
                continue
            if p_prime not in self.train_taxon2allemb_id:
                continue
            if (p_prime, c) in pos_set:
                continue
            if (p_prime, c) not in all_pool:      # 双重保险：必须真在池子里
                continue
            layer = self._classify_parent(p0, p_prime)
            if layer and 1 <= layer <= 8:
                buckets[layer].append(p_prime)
        result[(p0, c)] = buckets

    self._hard_pool_cache[query_node] = result
    return result
```

#### (e) 新增 `_allocate_quotas` 与 `_get_mixed_negatives`

```python
# 各层权重 = 两份错误分析里「叶子 query 父错子对」各层占比的平均值（§3.2）
LAYER_WEIGHTS = {1: 5.27, 2: 8.31, 3: 5.19, 4: 2.13, 5: 2.49, 6: 0.71, 7: 0.46, 8: 0.52}


def _allocate_quotas(self, n_hard, layers):
    """
    在【当前开放层】内按 LAYER_WEIGHTS 重新归一化，用最大余数法拆出每层配额。
    返回 {layer: quota}，总和恰为 n_hard。
    """
    ws = {l: LAYER_WEIGHTS[l] for l in layers if l in LAYER_WEIGHTS}
    if not ws:
        return {}
    total = sum(ws.values())
    raw = {l: n_hard * w / total for l, w in ws.items()}
    quota = {l: int(raw[l]) for l in ws}
    rest = n_hard - sum(quota.values())
    for l in sorted(ws, key=lambda x: raw[x] - quota[x], reverse=True)[:rest]:
        quota[l] += 1
    return quota


def _get_mixed_negatives(self, query_node, negative_size, positive_positions):
    """
    先按层配额取 hard 负例，再补齐随机负例，保证恰好 negative_size 个位置。
    """
    rho = self.cur_hard_ratio if self.hard_ratio > 0 else 0.0
    if rho <= 0 or not self.cur_layers:
        return self._get_exactly_k_negatives(query_node, negative_size)

    n_hard = int(round(negative_size * rho))
    if n_hard <= 0:
        return self._get_exactly_k_negatives(query_node, negative_size)

    p0, c = random.choice(positive_positions)         # 与本次正例对齐，避免正负位置互相矛盾
    buckets = self._build_hard_pool(query_node)[(p0, c)]
    quota = self._allocate_quotas(n_hard, self.cur_layers)

    hard_negs = []
    for layer in self.cur_layers:                     # 严格按层遍历，不跨界补
        need = quota.get(layer, 0)
        if need <= 0:
            continue
        pool = buckets.get(layer, [])
        if not pool:
            self.hard_stats['empty_layer'][layer] += 1
            continue                                  # 作废，配额交给随机部分
        if len(pool) >= need:
            picked = random.sample(pool, need)
        else:
            picked = random.choices(pool, k=need)     # 池子不足：放回补齐
            self.hard_stats['shortfall'] += 1
        hard_negs.extend((p, c) for p in picked)
        self.hard_stats['layer_hist'][layer] += len(picked)

    hard_negs = list(dict.fromkeys(hard_negs))        # 放回采样可能产生重复，这里去重
    self.hard_stats['hard'] += len(hard_negs)

    # 随机部分用「实际 remaining」而不是名义配额，保证总数恰好 negative_size
    n_rand = negative_size - len(hard_negs)
    rand_negs = self._get_exactly_k_negatives(
        query_node, n_rand, ignore=hard_negs) if n_rand > 0 else []
    self.hard_stats['rand'] += len(rand_negs)

    return hard_negs + rand_negs
```

#### (f) `__getitem__` 里改一行

```python
# 原：negative_positions = self._get_negative_positions(q, negative_size)
negative_positions = self._get_mixed_negatives(q, negative_size, positive_positions)
```

并把 `positive_positions` 提前到负采样之前（现状已经是这个顺序，确认即可）。

#### (g) `__init__` 签名新增参数

```python
def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100,
             expand_factor=64, cache_refresh_time=128, test_topk=-1, tokenizer=None,
             hard_ratio=0.0, hard_schedule=None):
```

### 5.2 `data_loader/data_loaders.py`

把 `hard_ratio` / `hard_schedule` 透传给 `RawDataset`：

```python
def __init__(self, ..., hard_ratio=0.0, hard_schedule=None):
    ...
    self.dataset = RawDataset(..., hard_ratio=hard_ratio, hard_schedule=hard_schedule)
```

### 5.3 `trainer/trainer.py`

在 `TrainerHidden._train_epoch`、`TrainerChain._train_epoch`、`TrainerTEMP._train_epoch` 的**开头**（`self.model.train()` 之后）加：

```python
# 推进负采样课程（只对实现了 set_epoch 的 dataset 生效）
if hasattr(self.data_loader.dataset, 'set_epoch'):
    self.data_loader.dataset.set_epoch(epoch)
```

并在每个 epoch 结束时（`log = {'epoch avg loss': ...}` 之前）打印统计：

```python
if hasattr(self.data_loader.dataset, 'hard_stats'):
    st = self.data_loader.dataset.hard_stats
    if st['hard'] > 0:
        self.logger.info(f"[hard-neg] hard={st['hard']} rand={st['rand']} "
                         f"shortfall={st['shortfall']} layer_hist={st['layer_hist']}")
```

### 5.4 `train.py` + config

`train.py` 的 `options` 只加一项（总开关）：

```python
CustomArgs(['--hard_ratio'], type=float, target=('train_data_loader', 'args', 'hard_ratio')),
```

> 消融实验（固定层、固定比例）**不靠 CLI 传参**，而是复制一份 config 文件改 `hard_schedule`—— 层列表用 CLI 表达太别扭，且容易传错。

config（`config_files/SemEval-V/config.test.hidden.json`）的 `train_data_loader.args` 里加：

```json
"hard_ratio": 0.0,
"hard_schedule": [
    [10,  0.00, []],
    [20,  0.15, [6, 7, 8]],
    [32,  0.25, [4, 5, 6, 7, 8]],
    [45,  0.35, [2, 3, 4, 5, 6, 7, 8]],
    [100, 0.40, [1, 2, 3, 4, 5, 6, 7, 8]]
]
```

格式说明：每个元素 `[end_epoch, ratio, [允许的层]]`，表示「从上一个 end_epoch+1 到本 end_epoch」使用这个配置。

> `hard_ratio = 0.0` 是**默认关闭**，保证不传参时行为与论文完全一致。
> 若 `hard_schedule = []` 但 `hard_ratio > 0`，则退化为「全程固定比例 + 全 8 层」，方便快速跑通。

---

## 6. 关键约束与必做校验

### 6.1 合法性（重复强调）

新造出来的 `(p', c)` **必须真的在 `self.all_edges` 里**。在 `_build_hard_pool` 里加了 `(p_prime, c) not in all_pool: continue` 这一行，**不要删**。

原因：test/valid 的评测候选池就是 `all_edges`，如果训练时喂了池子外的位置，模型会学到「打分函数在池外也要有区分度」，与评测口径不一致。

### 6.2 不能引入假负例

`(p', c) not in node2pos[q]` 必须检查。多父 query（MeSH 694 个、Verb 0 个）尤其要注意：P₀ 可能不止一个。

### 6.3 性能

- `_build_hard_pool` 里对每个 query 算一次 `nx.ancestors`，**必须缓存**（`self._hard_pool_cache`）。
- 缓存后每个 epoch 的额外开销应该 < 5%。超过就说明缓存没生效。
- 如果内存吃紧，给缓存加容量上限（LRU），或者只在 `mode == 'train'` 时建。

### 6.4 随机性

改造会改变 RNG 消耗序列，**导致与原论文实验不可比**。所以：
- 必须**重跑 baseline**（`hard_ratio=0`），不要拿论文表格里的数字当对照。
- 报告时要写清「本次 baseline 是重跑的」。

### 6.5 诊断输出（实现完成后必跑）

跑 20 个 query 的 smoke test，打印：

```
query=read||read.v.01  leaf=True
  L1 直接父: ['perceive||perceive.v.01']
  L2 直接子: []
  L3 兄弟  : ['interpret||interpret.v.01', 'look||look.v.02', ...]  (7 个)
  L4 二跳父: [...]
  L5 二跳子: [...]
  L6~L8   : 12 个
```

**如果 L1/L2/L3 大面积为空，说明 `node2parents` / `node2children` 的键对不上**（注意 `node_list` 不含 `pseudo_root`，查它要防 KeyError）。

---

## 7. 实验设计

四组**单变量对照**，都从 A 出发只改一项：

| 组 | `hard_ratio` | 课程 | `hard_schedule` 里的层 | 回答什么问题 | A vs 谁 |
|---|---|---|---|---|---|
| **A · baseline** | 0.0 | — | — | 本次改造的对照基准（**必须重跑**） | — |
| **B · 只 1 跳** | 0.30 | 无（全程固定） | `[1,2,3]` | 学姐的假设：只靠 1 跳错误能不能被消除 | B−A |
| **C · 全层加权** | 0.30 | 无（全程固定） | `[1..8]` | 按错误占比加权本身有多少收益 | C−A |
| **D · 课程（远→近）** | 见 §4 | **有** | L6→L4→L2→L1 逐步 | 课程是否比「一上来就上全层」更稳 | D−C |

对应的 `hard_schedule` 写法：

```jsonc
// B：只要 1 跳，不分阶段
"hard_schedule": [[100, 0.30, [1, 2, 3]]]

// C：全层固定，不分阶段
"hard_schedule": [[100, 0.30, [1, 2, 3, 4, 5, 6, 7, 8]]]

// D：课程（同 §4.2）
"hard_schedule": [[10,0.00,[]], [20,0.15,[6,7,8]], [32,0.25,[4,5,6,7,8]],
                  [45,0.35,[2,3,4,5,6,7,8]], [100,0.40,[1,2,3,4,5,6,7,8]]]
```

每组跑 3 个 seed（对齐论文的 `-n 3`），报告 `mean ± std`。

**重点看的指标**（按敏感度排序）：
1. `mrr_scaled_10`（连续、最稳）
2. `recall_at_10`（真实位置被压到前排的程度 —— 正是难负样本该改善的）
3. `hit_at_1`
4. Leaf / Non-leaf 分开看 —— 非叶 query 的提升应该更明显（因为它们的父节点更难猜）

**同时必须统计的辅助指标**：用同一份 `dump_hidden_rank.py` 重跑错误分类，看第 1~9 类的条目数有没有下降、第 10 类是否基本不变。

这条比总体指标重要得多：
- 若 **1~3 类下降、第 10 类不变** → 改造确实打中了目标；
- 若 **两类一起下降** → 可能只是整体打分变好了，不一定是负采样的功劳；
- 若 **1~3 类下降但总体指标没涨** → 说明这三类错误本身对 rank 的拖累有限（因为它们数量少）。

---

## 8. 验收标准

- [ ] `python -m py_compile` 全部通过
- [ ] `hard_ratio=0.0` 时，与改造前的日志逐 epoch 对比：loss 曲线完全一致（说明默认路径没被改动）
- [ ] smoke test 能打印出 L1~L3 的分层结果，非空
- [ ] 跑 3 个 epoch 不报错、不 OOM、单 epoch 耗时增幅 < 10%
- [ ] `layer_hist` 各层比例与 `_allocate_quotas` 给出的配额偏差 < 10%（说明加权生效，而不是退化成候选级均匀）
- [ ] `layer_hist` 里**没有落在 `cur_layers` 之外**的层（说明课程开关有效，没漏到 L9）
- [ ] `empty_layer[1]` 在 leaf query 上 < 5%（若很高，说明 `node2parents` 配错或 `p0` 取错了）
- [ ] `shortfall` / query 数 < 30%（否则说明层池太小，ρ 或配额要降）
- [ ] 每个 epoch 的 `[hard-neg] epoch N: ratio=..., layers=[...]` 打印顺序符合 §4.2 的阶段表

---

## 9. 明确不要做的事

1. **不要改 `model/model.py`**。这是纯采样侧改动，模型和 prompt 一个字都不动。
2. **不要改评测代码**（`trainer._test`、`metric.py`）。否则指标不可比。
3. **不要动 `sampling_mode=0` 分支**。
4. **不要删 `_get_exactly_k_negatives`**。`n_rand` 部分还要用它。
5. **不要为了「覆盖第 10 类」而扩大候选父集合**。第 10 类不做。
6. **不要改 `pseudo_leaf` 的全连接逻辑**。那是 TacoPrompt 的位置表示基础，动了全盘皆崩。
7. **不要引入新依赖**（不需要 scipy / 新的图库）。
8. **不要顺手「修正」prompt 里的 `Chlid` 拼写**。会让实验与已发布结果不可比。

---

## 10. 实现顺序（建议）

1. 先只加 `hard_ratio` 参数并透传，跑通「开关能关」（默认 0.0 行为不变）—— **这一步必须先提交、先验证**
2. 加 `set_epoch` + 打印，验证课程阶段能正确切换（可以先手写几个假的 schedule 试）
3. 加 `_classify_parent` + `_build_hard_pool`，只**打印**分层结果，不参与采样
4. 加 `_allocate_quotas` + `_get_mixed_negatives`，先用 `hard_schedule = [[100, 0.15, [6,7,8]]]`（临时跳过 warmup）跑 2 个 epoch，看 `layer_hist` 是否落在 L6~L8
5. 换成完整课程（§4.2），确认 `[hard-neg]` 打印的阶段切换符合预期
6. 跑 A/B/C/D 四组实验（§7）

每一步都单独 commit，便于回滚。
