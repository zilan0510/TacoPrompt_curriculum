# MeSH-B — TacoPrompt hidden 变体 Top-10 错误分类统计

- 数据集：`MeSH-B`；测试 query 数：**819**（其中叶子 query 558 个，占 68.13%）
- 候选位置总数：42970；每个 query 取 Top-10，共 8190 个预测条目
- 其中正确位置 864 条（10.55%），错误位置 **7326** 条（89.45%）
- 正确位置出现在 Top-10 的 query：628（76.68%）

## 零、推理口径校验（复算项目指标）

| 指标 | 全部 query | 叶子 query | 非叶子 query |
|---|---|---|---|
| hit@1 | 0.3248 | 0.3728 | 0.2222 |
| hit@5 | 0.6337 | 0.6649 | 0.5670 |
| hit@10 | 0.7668 | 0.7724 | 0.7548 |
| recall@1 | 0.1428 | 0.2930 | 0.0503 |
| recall@5 | 0.3408 | 0.5831 | 0.1917 |
| recall@10 | 0.4643 | 0.6972 | 0.3209 |
| mrr | 0.2439 | 0.4262 | 0.1317 |
| mrr_scaled_10 | 0.5797 | 0.7521 | 0.4735 |
| macro_mr | 83.7786 | 98.4180 | 52.4804 |
| micro_mr | 97.6044 | 117.6831 | 85.2402 |

（应与训练日志中的 `test_*` / `test_leaf_*` / `test_nonleaf_*` 一致）

### 与论文 Table 1（`Ours-Hidden` 行）对齐（MeSH）

| 分组 | 指标 | 论文 Table 1 (Ours-Hidden) | 本次复现 | 差值 |
|---|---|---|---|---|
| Total | MR↓ | 49.140 | 83.779 | +34.639 |
| Total | MRR | 0.674 | 0.580 | -0.094 |
| Total | R@1 | 17.9 | 14.3 | -3.6 |
| Total | R@5 | 42.4 | 34.1 | -8.3 |
| Total | R@10 | 55.9 | 46.4 | -9.5 |
| Total | H@1 | 40.7 | 32.5 | -8.2 |
| Total | H@5 | 74.6 | 63.4 | -11.2 |
| Total | H@10 | 84.6 | 76.7 | -7.9 |
| Leaf | MRR | 0.868 | 0.752 | -0.116 |
| Leaf | H@5 | 79.0 | 66.5 | -12.5 |
| Leaf | R@10 | 81.5 | 69.7 | -11.8 |
| Non-leaf | MRR | 0.554 | 0.474 | -0.080 |
| Non-leaf | H@5 | 65.1 | 56.7 | -8.4 |
| Non-leaf | R@10 | 40.2 | 32.1 | -8.1 |

真实位置排名分布（用于解释 MR 的波动 —— MR 由重尾主导）：

| 统计量 | 数值 |
|---|---|
| 中位数 | 13 |
| 90 分位 | 222 |
| 99 分位 | 1439 |
| 最大 | 5409 |
| macro 均值（每 query 先平均 = 论文 MR 口径） | 97.6（按 query 平均为 83.8） |
| micro 均值（全部真实位置直接平均） | 97.6 |
| micro 均值去掉 top-1% 极端排名后 | 70.8 |

## 一、全局错误大类占比

| 错误大类 | 条目数 | 占全部错误 |
|---|---|---|
| 父错子对 | 2957 | 40.36% |
| 父对子错 | 1595 | 21.77% |
| 父子皆错 | 2774 | 37.87% |
| **合计** | **7326** | 100.00% |

**父错子对（父子插入类错误）占全部错误的比例：40.36%**

## 二、父错子对前置统计项

- 父错子对总错误条目：**2957**
- 其中 query 为叶子（正确子标签 = pseudo leaf）的条目：2514，占父错子对总错误的 **85.02%**
- 其中 query 为非叶子（正确子标签为真实节点）的条目：443，占 14.98%（不参与下面的 10 类细分）
- 多父 query 样本数：394；错误节点类型分布：非真实节点(pseudo root 等) 608，真实节点 1906

## 三、10 类细分错误占比（仅叶子 query 的父错子对样本）

叶子子集样本数 = **2514**

| # | 类别 | 条目数 | 占叶子子集父错子对 | 占父错子对总错误 | 占全部错误 |
|---|---|---|---|---|---|
| 1 | 兄弟节点类(2跳) | 84 | 3.34% | 2.84% | 1.15% |
| 2 | 直接父节点类(1跳) | 94 | 3.74% | 3.18% | 1.28% |
| 3 | 直接子节点类(1跳) | 59 | 2.35% | 2.00% | 0.81% |
| 4 | 二跳父节点类 | 35 | 1.39% | 1.18% | 0.48% |
| 5 | 二跳子节点类 | 41 | 1.63% | 1.39% | 0.56% |
| 6 | 三跳父节点类 | 6 | 0.24% | 0.20% | 0.08% |
| 7 | 三跳子节点类 | 7 | 0.28% | 0.24% | 0.10% |
| 8 | 四跳及以上父节点类 | 5 | 0.20% | 0.17% | 0.07% |
| 9 | 四跳及以上子节点类 | 3 | 0.12% | 0.10% | 0.04% |
| 10 | 其他类 | 2180 | 86.71% | 73.72% | 29.76% |
| - | **合计** | **2514** | 100.00% | 85.02% | - |

## 四、各类别的 LCA 距离分布（x₁ = P₀→Mx 跳数，x₂ = 错误节点→Mx 跳数）

### 1. 兄弟节点类(2跳)：84 条（占叶子子集 3.34%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (1, 1) | 84 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| proteins | 8 | 9.52% |
| health services | 3 | 3.57% |
| public health | 3 | 3.57% |
| dentistry | 3 | 3.57% |
| behavior | 3 | 3.57% |
| surgical procedures , operative | 3 | 3.57% |
| genetic phenomena | 3 | 3.57% |
| pathological conditions , signs and symptoms | 3 | 3.57% |
| physical therapy modalities | 3 | 3.57% |
| nervous system diseases | 2 | 2.38% |
| （共 54 个不同 Mx） | 84 | 100.00% |

### 2. 直接父节点类(1跳)：94 条（占叶子子集 3.74%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (1, 0) | 94 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| health facilities | 2 | 2.13% |
| cell physiological phenomena | 2 | 2.13% |
| biochemical phenomena | 2 | 2.13% |
| pharmaceutical preparations | 2 | 2.13% |
| complex mixtures | 2 | 2.13% |
| geological phenomena | 2 | 2.13% |
| beverages | 2 | 2.13% |
| medicine | 2 | 2.13% |
| diagnostic techniques , cardiovascular | 2 | 2.13% |
| genes | 2 | 2.13% |
| （共 79 个不同 Mx） | 94 | 100.00% |

### 3. 直接子节点类(1跳)：59 条（占叶子子集 2.35%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (0, 1) | 59 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| publication formats | 3 | 5.08% |
| peptides | 2 | 3.39% |
| pathologic processes | 2 | 3.39% |
| clinical laboratory techniques | 2 | 3.39% |
| occupational diseases | 2 | 3.39% |
| patient care | 2 | 3.39% |
| therapeutics | 2 | 3.39% |
| technology , pharmaceutical | 2 | 3.39% |
| behavior | 1 | 1.69% |
| central nervous system diseases | 1 | 1.69% |
| （共 50 个不同 Mx） | 59 | 100.00% |

### 4. 二跳父节点类：35 条（占叶子子集 1.39%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (2, 0) | 35 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| health care economics and organizations | 2 | 5.71% |
| diet , food , and nutrition | 2 | 5.71% |
| computing methodologies | 2 | 5.71% |
| epidemiologic study characteristics | 2 | 5.71% |
| food and beverages | 2 | 5.71% |
| health care evaluation mechanisms | 2 | 5.71% |
| health behavior | 1 | 2.86% |
| mathematical concepts | 1 | 2.86% |
| genetic phenomena | 1 | 2.86% |
| genome components | 1 | 2.86% |
| （共 29 个不同 Mx） | 35 | 100.00% |

### 5. 二跳子节点类：41 条（占叶子子集 1.63%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (0, 2) | 41 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| health facilities | 3 | 7.32% |
| clinical laboratory techniques | 3 | 7.32% |
| central nervous system diseases | 3 | 7.32% |
| epidemiologic methods | 3 | 7.32% |
| proteins | 3 | 7.32% |
| ganglia | 2 | 4.88% |
| health care evaluation mechanisms | 2 | 4.88% |
| technology , industry , and agriculture | 2 | 4.88% |
| inorganic chemicals | 2 | 4.88% |
| genetic techniques | 2 | 4.88% |
| （共 26 个不同 Mx） | 41 | 100.00% |

### 6. 三跳父节点类：6 条（占叶子子集 0.24%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (3, 0) | 6 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| delivery of health care | 1 | 16.67% |
| carrier proteins | 1 | 16.67% |
| peptides | 1 | 16.67% |
| nervous system physiological phenomena | 1 | 16.67% |
| health care evaluation mechanisms | 1 | 16.67% |
| information science | 1 | 16.67% |
| （共 6 个不同 Mx） | 6 | 100.00% |

### 7. 三跳子节点类：7 条（占叶子子集 0.28%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (0, 3) | 7 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| membrane proteins | 3 | 42.86% |
| behavior | 1 | 14.29% |
| central nervous system diseases | 1 | 14.29% |
| organization and administration | 1 | 14.29% |
| inorganic chemicals | 1 | 14.29% |
| （共 5 个不同 Mx） | 7 | 100.00% |

### 8. 四跳及以上父节点类：5 条（占叶子子集 0.20%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (4, 0) | 4 | 80.00% |
| (5, 0) | 1 | 20.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| population characteristics | 2 | 40.00% |
| peptides | 1 | 20.00% |
| environment and public health | 1 | 20.00% |
| epidemiologic methods | 1 | 20.00% |
| （共 4 个不同 Mx） | 5 | 100.00% |

### 9. 四跳及以上子节点类：3 条（占叶子子集 0.12%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (0, 4) | 2 | 66.67% |
| (0, 5) | 1 | 33.33% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| epidemiologic methods | 1 | 33.33% |
| fluids and secretions | 1 | 33.33% |
| genetic phenomena | 1 | 33.33% |
| （共 3 个不同 Mx） | 3 | 100.00% |

### 10. 其他类：2180 条（占叶子子集 86.71%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (None, None) | 1937 | 88.85% |
| (1, 2) | 36 | 1.65% |
| (2, 2) | 29 | 1.33% |
| (2, 1) | 26 | 1.19% |
| (2, 3) | 24 | 1.10% |
| (3, 2) | 21 | 0.96% |
| (3, 1) | 13 | 0.60% |
| (4, 2) | 12 | 0.55% |
| (1, 3) | 11 | 0.50% |
| (3, 3) | 11 | 0.50% |
| (3, 4) | 9 | 0.41% |
| (4, 1) | 8 | 0.37% |
| (4, 3) | 6 | 0.28% |
| (1, 4) | 6 | 0.28% |
| (2, 4) | 5 | 0.23% |
| (5, 3) | 5 | 0.23% |
| (5, 2) | 4 | 0.18% |
| (5, 1) | 4 | 0.18% |
| (2, 5) | 4 | 0.18% |
| (4, 4) | 3 | 0.14% |
| (5, 4) | 2 | 0.09% |
| (1, 6) | 1 | 0.05% |
| (3, 5) | 1 | 0.05% |
| (7, 3) | 1 | 0.05% |
| (6, 1) | 1 | 0.05% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| (无公共祖先) | 1329 | 60.96% |
| (无) | 608 | 27.89% |
| information science | 22 | 1.01% |
| glycoproteins | 19 | 0.87% |
| social sciences | 16 | 0.73% |
| behavior and behavior mechanisms | 13 | 0.60% |
| amino acids , peptides , and proteins | 10 | 0.46% |
| behavioral sciences | 9 | 0.41% |
| biological factors | 8 | 0.37% |
| therapeutics | 6 | 0.28% |
| （共 72 个不同 Mx） | 2180 | 100.00% |

第 10 类（其他类）的构成：

| 子类 | 条目数 | 占其他类 | 占叶子子集父错子对 |
|---|---|---|---|
| 与 P₀ 无公共祖先（跨顶级树） | 1329 | 60.96% | 52.86% |
| 错误节点 = pseudo root | 557 | 25.55% | 22.16% |
| 有公共祖先但非 1-9 类（远亲/跨分支） | 243 | 11.15% | 9.67% |
| 错误节点为孤立节点(不在 .taxo 中) | 51 | 2.34% | 2.03% |
