# MeSH-C — TacoPrompt hidden 变体 Top-10 错误分类统计

- 数据集：`MeSH-C`；测试 query 数：**819**（其中叶子 query 558 个，占 68.13%）
- 候选位置总数：42970；每个 query 取 Top-10，共 8190 个预测条目
- 其中正确位置 832 条（10.16%），错误位置 **7358** 条（89.84%）
- 正确位置出现在 Top-10 的 query：613（74.85%）

## 零、推理口径校验（复算项目指标）

| 指标 | 全部 query | 叶子 query | 非叶子 query |
|---|---|---|---|
| hit@1 | 0.2711 | 0.3190 | 0.1686 |
| hit@5 | 0.6264 | 0.6810 | 0.5096 |
| hit@10 | 0.7485 | 0.7796 | 0.6820 |
| recall@1 | 0.1192 | 0.2507 | 0.0382 |
| recall@5 | 0.3290 | 0.6042 | 0.1596 |
| recall@10 | 0.4466 | 0.7000 | 0.2905 |
| mrr | 0.2248 | 0.4025 | 0.1155 |
| mrr_scaled_10 | 0.5720 | 0.7611 | 0.4555 |
| macro_mr | 80.5838 | 95.1576 | 49.4261 |
| micro_mr | 91.9066 | 115.6930 | 77.2593 |

（应与训练日志中的 `test_*` / `test_leaf_*` / `test_nonleaf_*` 一致）

### 与论文 Table 1（`Ours-Hidden` 行）对齐（MeSH）

| 分组 | 指标 | 论文 Table 1 (Ours-Hidden) | 本次复现 | 差值 |
|---|---|---|---|---|
| Total | MR↓ | 49.140 | 80.584 | +31.444 |
| Total | MRR | 0.674 | 0.572 | -0.102 |
| Total | R@1 | 17.9 | 11.9 | -6.0 |
| Total | R@5 | 42.4 | 32.9 | -9.5 |
| Total | R@10 | 55.9 | 44.7 | -11.2 |
| Total | H@1 | 40.7 | 27.1 | -13.6 |
| Total | H@5 | 74.6 | 62.6 | -12.0 |
| Total | H@10 | 84.6 | 74.8 | -9.8 |
| Leaf | MRR | 0.868 | 0.761 | -0.107 |
| Leaf | H@5 | 79.0 | 68.1 | -10.9 |
| Leaf | R@10 | 81.5 | 70.0 | -11.5 |
| Non-leaf | MRR | 0.554 | 0.456 | -0.098 |
| Non-leaf | H@5 | 65.1 | 51.0 | -14.1 |
| Non-leaf | R@10 | 40.2 | 29.1 | -11.1 |

真实位置排名分布（用于解释 MR 的波动 —— MR 由重尾主导）：

| 统计量 | 数值 |
|---|---|
| 中位数 | 14 |
| 90 分位 | 210 |
| 99 分位 | 1194 |
| 最大 | 11157 |
| macro 均值（每 query 先平均 = 论文 MR 口径） | 91.9（按 query 平均为 80.6） |
| micro 均值（全部真实位置直接平均） | 91.9 |
| micro 均值去掉 top-1% 极端排名后 | 64.2 |

## 一、全局错误大类占比

| 错误大类 | 条目数 | 占全部错误 |
|---|---|---|
| 父错子对 | 3194 | 43.41% |
| 父对子错 | 1520 | 20.66% |
| 父子皆错 | 2644 | 35.93% |
| **合计** | **7358** | 100.00% |

**父错子对（父子插入类错误）占全部错误的比例：43.41%**

## 二、父错子对前置统计项

- 父错子对总错误条目：**3194**
- 其中 query 为叶子（正确子标签 = pseudo leaf）的条目：2706，占父错子对总错误的 **84.72%**
- 其中 query 为非叶子（正确子标签为真实节点）的条目：488，占 15.28%（不参与下面的 10 类细分）
- 多父 query 样本数：437；错误节点类型分布：非真实节点(pseudo root 等) 619，真实节点 2087

## 三、10 类细分错误占比（仅叶子 query 的父错子对样本）

叶子子集样本数 = **2706**

| # | 类别 | 条目数 | 占叶子子集父错子对 | 占父错子对总错误 | 占全部错误 |
|---|---|---|---|---|---|
| 1 | 兄弟节点类(2跳) | 93 | 3.44% | 2.91% | 1.26% |
| 2 | 直接父节点类(1跳) | 116 | 4.29% | 3.63% | 1.58% |
| 3 | 直接子节点类(1跳) | 81 | 2.99% | 2.54% | 1.10% |
| 4 | 二跳父节点类 | 39 | 1.44% | 1.22% | 0.53% |
| 5 | 二跳子节点类 | 33 | 1.22% | 1.03% | 0.45% |
| 6 | 三跳父节点类 | 6 | 0.22% | 0.19% | 0.08% |
| 7 | 三跳子节点类 | 7 | 0.26% | 0.22% | 0.10% |
| 8 | 四跳及以上父节点类 | 6 | 0.22% | 0.19% | 0.08% |
| 9 | 四跳及以上子节点类 | 2 | 0.07% | 0.06% | 0.03% |
| 10 | 其他类 | 2323 | 85.85% | 72.73% | 31.57% |
| - | **合计** | **2706** | 100.00% | 84.72% | - |

## 四、各类别的 LCA 距离分布（x₁ = P₀→Mx 跳数，x₂ = 错误节点→Mx 跳数）

### 1. 兄弟节点类(2跳)：93 条（占叶子子集 3.44%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (1, 1) | 93 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| proteins | 8 | 8.60% |
| dentistry | 6 | 6.45% |
| behavior | 3 | 3.23% |
| inorganic chemicals | 3 | 3.23% |
| specialty uses of chemicals | 2 | 2.15% |
| heart diseases | 2 | 2.15% |
| information science | 2 | 2.15% |
| health occupations | 2 | 2.15% |
| surgical procedures , operative | 2 | 2.15% |
| behavior and behavior mechanisms | 2 | 2.15% |
| （共 63 个不同 Mx） | 93 | 100.00% |

### 2. 直接父节点类(1跳)：116 条（占叶子子集 4.29%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (1, 0) | 116 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| biochemical phenomena | 4 | 3.45% |
| environment | 4 | 3.45% |
| communication | 3 | 2.59% |
| equipment and supplies | 2 | 1.72% |
| environmental exposure | 2 | 1.72% |
| persons | 2 | 1.72% |
| pharmaceutical preparations | 2 | 1.72% |
| complex mixtures | 2 | 1.72% |
| geological phenomena | 2 | 1.72% |
| beverages | 2 | 1.72% |
| （共 93 个不同 Mx） | 116 | 100.00% |

### 3. 直接子节点类(1跳)：81 条（占叶子子集 2.99%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (0, 1) | 81 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| publication formats | 3 | 3.70% |
| genes | 3 | 3.70% |
| technology , pharmaceutical | 3 | 3.70% |
| carcinoma | 2 | 2.47% |
| peptides | 2 | 2.47% |
| genetic phenomena | 2 | 2.47% |
| mouth diseases | 2 | 2.47% |
| clinical laboratory techniques | 2 | 2.47% |
| drug evaluation , preclinical | 2 | 2.47% |
| chemical phenomena | 2 | 2.47% |
| （共 66 个不同 Mx） | 81 | 100.00% |

### 4. 二跳父节点类：39 条（占叶子子集 1.44%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (2, 0) | 39 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| health care economics and organizations | 2 | 5.13% |
| computing methodologies | 2 | 5.13% |
| environment | 2 | 5.13% |
| sociology | 2 | 5.13% |
| epidemiologic study characteristics | 2 | 5.13% |
| food and beverages | 2 | 5.13% |
| health behavior | 1 | 2.56% |
| laboratory chemicals | 1 | 2.56% |
| mathematical concepts | 1 | 2.56% |
| genome components | 1 | 2.56% |
| （共 33 个不同 Mx） | 39 | 100.00% |

### 5. 二跳子节点类：33 条（占叶子子集 1.22%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (0, 2) | 33 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| clinical laboratory techniques | 4 | 12.12% |
| epidemiologic methods | 3 | 9.09% |
| health facilities | 2 | 6.06% |
| central nervous system diseases | 2 | 6.06% |
| health care economics and organizations | 2 | 6.06% |
| proteins | 2 | 6.06% |
| equipment and supplies | 2 | 6.06% |
| carcinoma | 1 | 3.03% |
| pathologic processes | 1 | 3.03% |
| blood physiological phenomena | 1 | 3.03% |
| （共 23 个不同 Mx） | 33 | 100.00% |

### 6. 三跳父节点类：6 条（占叶子子集 0.22%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (3, 0) | 6 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| delivery of health care | 1 | 16.67% |
| peptides | 1 | 16.67% |
| cytoplasmic structures | 1 | 16.67% |
| signs and symptoms | 1 | 16.67% |
| nervous system physiological phenomena | 1 | 16.67% |
| health care evaluation mechanisms | 1 | 16.67% |
| （共 6 个不同 Mx） | 6 | 100.00% |

### 7. 三跳子节点类：7 条（占叶子子集 0.26%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (0, 3) | 7 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| central nervous system diseases | 2 | 28.57% |
| membrane proteins | 1 | 14.29% |
| organization and administration | 1 | 14.29% |
| proteins | 1 | 14.29% |
| inorganic chemicals | 1 | 14.29% |
| technology , industry , and agriculture | 1 | 14.29% |
| （共 6 个不同 Mx） | 7 | 100.00% |

### 8. 四跳及以上父节点类：6 条（占叶子子集 0.22%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (4, 0) | 4 | 66.67% |
| (5, 0) | 2 | 33.33% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| population characteristics | 2 | 33.33% |
| hormones | 1 | 16.67% |
| neoplasms | 1 | 16.67% |
| health care evaluation mechanisms | 1 | 16.67% |
| epidemiologic methods | 1 | 16.67% |
| （共 5 个不同 Mx） | 6 | 100.00% |

### 9. 四跳及以上子节点类：2 条（占叶子子集 0.07%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (0, 4) | 2 | 100.00% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| fluids and secretions | 1 | 50.00% |
| genetic phenomena | 1 | 50.00% |
| （共 2 个不同 Mx） | 2 | 100.00% |

### 10. 其他类：2323 条（占叶子子集 85.85%）

| (x₁, x₂) | 条目数 | 占比 |
|---|---|---|
| (None, None) | 2070 | 89.11% |
| (1, 2) | 43 | 1.85% |
| (2, 1) | 29 | 1.25% |
| (2, 2) | 26 | 1.12% |
| (2, 3) | 24 | 1.03% |
| (3, 2) | 21 | 0.90% |
| (3, 3) | 16 | 0.69% |
| (4, 2) | 13 | 0.56% |
| (3, 4) | 12 | 0.52% |
| (3, 1) | 11 | 0.47% |
| (1, 3) | 11 | 0.47% |
| (4, 1) | 8 | 0.34% |
| (1, 4) | 7 | 0.30% |
| (2, 4) | 7 | 0.30% |
| (2, 5) | 6 | 0.26% |
| (4, 3) | 4 | 0.17% |
| (5, 3) | 3 | 0.13% |
| (5, 2) | 3 | 0.13% |
| (5, 1) | 2 | 0.09% |
| (4, 4) | 2 | 0.09% |
| (1, 6) | 1 | 0.04% |
| (8, 2) | 1 | 0.04% |
| (8, 4) | 1 | 0.04% |
| (3, 5) | 1 | 0.04% |
| (4, 5) | 1 | 0.04% |

Mx 分组（Top-10 最近公共祖先）：

| Mx | 条目数 | 占比 |
|---|---|---|
| (无公共祖先) | 1451 | 62.46% |
| (无) | 619 | 26.65% |
| information science | 28 | 1.21% |
| social sciences | 20 | 0.86% |
| glycoproteins | 18 | 0.77% |
| behavior and behavior mechanisms | 15 | 0.65% |
| amino acids , peptides , and proteins | 11 | 0.47% |
| behavioral sciences | 10 | 0.43% |
| biological factors | 8 | 0.34% |
| health occupations | 6 | 0.26% |
| （共 74 个不同 Mx） | 2323 | 100.00% |

第 10 类（其他类）的构成：

| 子类 | 条目数 | 占其他类 | 占叶子子集父错子对 |
|---|---|---|---|
| 与 P₀ 无公共祖先（跨顶级树） | 1451 | 62.46% | 53.62% |
| 错误节点 = pseudo root | 558 | 24.02% | 20.62% |
| 有公共祖先但非 1-9 类（远亲/跨分支） | 253 | 10.89% | 9.35% |
| 错误节点为孤立节点(不在 .taxo 中) | 61 | 2.63% | 2.25% |
