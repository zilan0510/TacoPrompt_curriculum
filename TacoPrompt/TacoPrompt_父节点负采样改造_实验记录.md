# 父节点负采样改造 · 实现与实验记录

> 依据：`TacoPrompt_父节点负采样改造_Prompt.md`（下称「任务书」）
> 原则：只改父节点负采样，不动模型结构 / 评测口径 / 子节点采样。

---

## 一、代码改动清单

| 文件 | 改动 |
|---|---|
| `data_loader/dataset.py` | `RawDataset.__init__` 新增 `hard_ratio` / `hard_schedule` 参数与状态；新增 `set_epoch` / `_layers_of` / `_classify_parent` / `_filter_hard` / `_allocate_quotas` / `_get_mixed_negatives`；`__getitem__` 的负采样调用改为 `_get_mixed_negatives` |
| `data_loader/data_loaders.py` | `hard_ratio` / `hard_schedule` 透传给 `RawDataset` |
| `trainer/trainer.py` | 三个 trainer（TEMP / Hidden / Chain）的 `_train_epoch` 开头调用 `dataset.set_epoch(epoch)`；epoch 末尾打印 `[hard-neg] hard=… rand=… shortfall=… layer_hist=…` |
| `train.py` | 新增 CLI `--hard_ratio`（层级配置走 config 的 `hard_schedule`） |
| `config_files/{mesh,SemEval-V}/config.test.hidden.json` | 加 `"hard_ratio": 0.0, "hard_schedule": []`（**默认关闭，行为与论文一致**） |
| `config_files/*/config.test.hidden.hard{B,C,D}.json` | B/C/D 三组消融配置 |
| `analysis/hard_neg_diagnose.py` | 新增：分层抽样打印、各层可采样量统计、合法性审计、配额核验、RNG 一致性、耗时 |
| `slurm/train_mesh_{B,C,D}.slurm`、`slurm/train_verb_D.slurm`、`slurm/run_dump_generic.slurm`、`slurm/watch_and_dump.sh` | 实验与推理脚本 |

### 1.1 与任务书实现的偏差（都是为了不出错/不爆内存，语义未变）

| 偏差 | 原因 |
|---|---|
| pseudo 节点用 `==`（`Taxon.__eq__` 按 display_name）而不是 `is` 判等 | `core_subgraph` 来自 `subgraphs.pickle`，反序列化出的 `pseudo_leaf` / `pseudo_root` **不是同一个对象实例**，用 `is` 会漏过滤（实测会把 `pseudo root` 采成 L6 负例） |
| `_filter_hard` 额外排除 **`p′ == query` 自身** | query 是 P₀ 的子节点，会落进 L2；测试时候选池里根本不存在 `(q, ·)` 这种位置（q 已被移出 core 子图），喂它等于教模型一个测试期不存在的模式 |
| 层候选集合按 **`p0`** 缓存（不是按 query 缓存候选列表） | leaf query 的合法父 ≈ core 子图全部 1.2 万节点，按 task book 写法 `{(p0,c): {layer: [...]}}` 逐 query 缓存会把 8 个桶全部实例化，Verb 上约 1 GB+/epoch 级内存；改成「p0 → 8 个层集合」缓存后总量只有 ~15 万条引用，且采样时再套 query 级过滤 |
| `set_epoch` 里把 `hard_stats` **按 epoch 归零** | 任务书的打印是累计值，无法核对「本 epoch 的 layer_hist 是否等于配额」 |
| `_get_mixed_negatives` 的唯一开关是 `self.cur_hard_ratio` | 任务书写的是 `rho = self.cur_hard_ratio if self.hard_ratio > 0 else 0.0`，这会让「只配 `hard_schedule`、`hard_ratio=0`」的用法整条失效 |

### 1.2 一个必须知道的口径细节

`n_hard = int(round(k * ρ))`，Python 的 `round` 是银行家舍入：`round(15 × 0.30) = round(4.5) = 4`（不是 5），`round(15 × 0.40) = 6`。因此 ρ=0.30 时每个 query 只有 **4** 个 hard 负例，各层占比的最小粒度是 25%。

---

## 二、诊断与验收结果（`analysis/hard_neg_diagnose.py`，CPU）

### 2.1 合法性（300 个 query × 15 负例，B 组与 C 组各跑一遍）

```
合法性：{'not_in_pool': 0, 'not_in_allemb': 0, 'hit_pos': 0, 'dup': 0, 'count': 0}
落在 cur_layers 之外的层：[]
shortfall / query 数 = 6.7%（Verb）/ 9.3%（MeSH）        ← 要求 < 30%
```

- 所有 `(p′, c)` 都在 `all_edges` 里（评测口径一致）
- 没有撞正例、没有重复、条数恒为 `negative_size=15`
- 课程关闭的层一个都没被采到

### 2.2 各层可采样数量（任务书 §2.3）

MeSH（300 query，leaf 221 / non-leaf 79）：

| 层 | 均值 | 中位数 | 为 0 比例 | 最大 |
|---|---|---|---|---|
| L1 直接父 | 0.7 | 1 | 43.0% | 4 |
| L2 直接子 | 3.2 | 0 | 66.9% | 55 |
| L3 兄弟 | 4.9 | 0 | 70.1% | 77 |
| L4 二跳父 | 0.5 | 0 | 65.5% | 5 |
| L5 二跳子 | 3.8 | 0 | 80.6% | 173 |
| L6 三跳父 | 0.2 | 0 | 83.2% | 4 |
| L7 三跳子 | 3.0 | 0 | 87.0% | 232 |
| L8 四跳+ | 2.6 | 0 | 86.7% | 314 |

Wordnet-Verb（200 query，leaf 153 / non-leaf 47）：

| 层 | 均值 | 中位数 | 为 0 比例 | 最大 |
|---|---|---|---|---|
| L1 直接父 | 0.6 | 1 | 37.7% | 1 |
| L2 直接子 | 9.4 | 0 | 57.9% | 451 |
| L3 兄弟 | 24.9 | 0 | 64.2% | 451 |
| L4 二跳父 | 0.4 | 0 | 64.8% | 1 |
| L5 二跳子 | 9.3 | 0 | 73.8% | 553 |
| L6 三跳父 | 0.1 | 0 | 89.1% | 1 |
| L7 三跳子 | 5.3 | 0 | 87.9% | 328 |
| L8 四跳+ | 5.1 | 0 | 94.4% | 261 |

**关键事实**：L1 的平均可采量 ≈ 0.6~0.7（≤1 个），且 38~43% 的 query 上层池为空（因为唯一父是 `pseudo root`，按口径被排除）。所以 L1 无法承担 21% 的配额——这也是「实际占比 vs 可达成占比」仍有 ±15~25% 偏差的原因（ρ=0.3 时每格 25%，一格的差异就是 25%）。

### 2.3 与改造前的一致性（对应验收标准第 2 条）

```
50/50 个 query 的负样本序列完全一致
50/50 个 query 的 RNG 状态完全一致（说明 rho=0 时没有多消耗随机数）
hard_stats（应为全 0）：0 / 0
```

→ `hard_ratio=0.0` 时**逐位等价**于改造前代码，因此
**已有的 MeSH（`1108_153108`，best=epoch 43）与 Verb（`1110_211557`，epoch 41）训练结果可直接作为 A 组 baseline**，不需要重跑。

### 2.4 时间开销

| 场景 | 每次负采样 |
|---|---|
| hard=0（baseline） | 0.06 ms / query |
| hard=0.40 全 8 层 | 0.28 ms / query |

每 query 约 +0.22 ms，而训练每 query ≈ 100 ms（含 1 次 BERT forward+backward），开销 **< 0.3%**（要求 < 5%）。层缓存建一次 0.3 s。

---

## 三、实验矩阵

| 组 | 配置 | hard_ratio | 层 / 课程 | 作业 | 状态 |
|---|---|---|---|---|---|
| **A · baseline** | `config.test.hidden.json` | 0.0 | — | 复用 `1108_153108`(MeSH) / `1110_211557`(Verb) | 已完成（RNG 等价性已验证） |
| **B · 只 1 跳** | `config.test.hidden.hardB.json` | 0.30 | `[1,2,3]` 固定 | 133628 | 排队中 |
| **C · 全层加权** | `config.test.hidden.hardC.json` | 0.30 | `[1..8]` 固定 | 133629 | 排队中 |
| **D · 课程** | `config.test.hidden.hardD.json` | 0.40 | L6→L4→L2→L1 四段 | 133630 (MeSH) / 133627 (Verb) | Verb 在跑，MeSH 排队 |

**已废弃的提交**：133624/133625/133626 —— 落到 main-12（RTX2080Ti 10.57 GiB）后 mesh 训练第一个 batch 就 OOM（PyTorch 分配 10.27 GiB），已把 mesh 脚本的 partition 收窄成 `RTX4090,RTX3090` 重投。Verb 因 `batch_size=2` 显存够，留在 2080Ti 上跑。

每个作业都挂了守卫 `slurm/watch_and_dump.sh`：日志出现 `Training stops.` 或 `Testing with best model...` 就 `scancel`（跳过 trainer 那遍 20~40 h 的冗余全量测试），并自动按 2/4 分片提交 `analysis/dump_hidden_rank.py`。

---

## 四、已知风险 / 待观察

1. **课程第 4 段（L1）可能到不了**：两个数据集的 baseline 都在 epoch 45~46 早停，而任务书的课程表把 L1 放在 46–100。若 D 组同样在 46 前停，则实际只走到「L2~L8」；届时 D−C 的差异反映的是 L4/L6→L2 的收缩，而不是 L1。
2. **ρ 与 L1 池子**：ρ 封顶 0.40（n_hard=6）而 L1 常有 0 或 1 个候选，L1 的真实权重被稀释。
3. **单 seed**：任务书要求 3 个 seed，但每个 run 在 4090 上约 20 h 训练 + dump，集群 GPU 长期满载，本轮先做单 seed（47，与论文一致）。要复现 mean±std 需要另排期。

---

## 五、复现命令

```bash
# 诊断（CPU，秒级）
python analysis/hard_neg_diagnose.py -c config_files/mesh/config.test.hidden.json --n_show 20 --n_stat 1000

# 训练（B/C/D）
sbatch slurm/train_mesh_B.slurm    # 或 _C / _D / train_verb_D.slurm

# 守卫（早停即停 + 自动 dump）
nohup bash slurm/watch_and_dump.sh <jobid> slurm/out/train_mesh_B_<jobid>.out \
      data/saved/mesh/TacoPromptHidden config_files/mesh/config.test.hidden.hardB.json mesh_B 2 &

# 统计（等 dump 完成后）
python analysis/error_analysis.py -d analysis/results/mesh_B_top10_shard0.json analysis/results/mesh_B_top10_shard1.json \
      -t data/mesh/mesh.taxo -n data/mesh/mesh.terms -o analysis/results/mesh_B_error_analysis.md --title "MeSH-B"
```

---

## 六、2026-11-22 进展与事故记录

### 6.1 已完成 / 在跑的作业

| 组 | 作业 | 节点 | 状态 | `val_mrr_scaled_10` |
|---|---|---|---|---|
| A baseline (MeSH) | 126630 | main-5 | 完成 | **0.7195**（best ep43） |
| A baseline (Verb) | 1110_211557 | main-9 | 完成 | **0.6177**（best ep41） |
| B (MeSH) | 133628 | main-6 | 完成（早停 ep43） | **0.6043** |
| C (MeSH) | 133629 | main-8 | 训练中（ep45+） | 0.5860 → 0.6022（ep44） |
| D (MeSH) | 133630 | main-8 | 完成（早停 ep43） | **0.6630** |
| D (Verb) | 133627 → 134107 续训 | main-12 → main-8 | epoch40 验证时 2080Ti OOM，已从 `checkpoint-epoch40` 按 baseline 同口径续训 | ep41 起 |
| Dc (MeSH, 压缩课程) | 134108 | — | 排队 | — |
| Dc (Verb, 压缩课程) | 134109 | main-12 | 运行中（warmup） | — |

**中期结论（仅验证集 MRR@10，尚未看测试集与错误分类）**：MeSH 上 A(0.7195) > D(0.6630) > B(0.6043) ≈ C(0.6022)，即三种硬负采样配置的验证 MRR 都低于 baseline。需要等测试集指标 + 第 1~9 类错误条目数后再判断是「难负例压迫导致排序整体变差」还是「局部改善但被尾部拖累」。

### 6.2 事故 1：代码被外部回退

2026-11-22 09:47:54，`data_loader/dataset.py`、`data_loader/data_loaders.py`、`trainer/trainer.py`、`train.py` 四个文件被回退到 HEAD 版本（无 stash、无备份），导致：

- 134103/134104/134106 三个作业在 `config.initialize()` 处 `TypeError: UnifiedDataset.__init__() got an unexpected keyword argument 'hard_ratio'` 秒退；
- 已按原内容重新应用全部改动，`py_compile` 通过，并复验：合法性 0 违规、`hard_ratio=0` 与改造前**逐位一致**（50/50 query 负样本序列 + RNG 状态相同）；
- 已重投：134107（Verb-D 续训）、134108（MeSH-Dc）、134109（Verb-Dc）。

> 正在跑的 133629（MeSH-C）在回退前已 import 旧模块，内存中的代码不受影响，结果仍有效。

### 6.3 事故 2：dump 的 verify 路径没套 autocast

`--verify` / `--verify_full` 直接调 `model.scorer(...)` 而没有 `with autocast()`，于是 `BertForMaskedLM` 用 fp32 算全词表 logits：$775\times256\times30522\times4\text{B}\approx24$ GiB → 24G 卡也 OOM（134041/134042 因此失败）。已改为 `with torch.no_grad(), autocast():`，与 trainer 的评测口径一致。

### 6.4 事故 3：守卫脚本挑错 run 目录

`ls -dt <dataset_dir>/*/models/trial1 | head -1` 在多 run 并行时会拿到**别人的** `model_best.pth`：mesh_B 的守卫曾取到 mesh_D 的目录，导致 `mesh_B_top10_shard*.json` 实际是 D 的结果（已删除重跑）。现守卫支持第 7 个参数 `EXP`，只认 `models/config.json` 里 `exp_name` 匹配的目录。

### 6.5 压缩课程表（Dc）

任务书的课程表是按 100 epoch 设计的，但两个数据集的 baseline 都在 **epoch 45~46 早停**，照原表 L1 要到 epoch 46 才开放 → 等于永远开不到。按实际预算（≈48 epoch）等比压缩 0.48：

| 阶段 | 原（100 epoch） | 压缩后（Dc） | ρ | 层 |
|---|---|---|---|---|
| 0 warmup | 1–10 | **1–5** | 0.00 | — |
| 1 远亲 | 11–20 | **6–10** | 0.15 | L6–L8 |
| 2 二跳 | 21–32 | **11–15** | 0.25 | L4–L8 |
| 3 一跳 | 33–45 | **16–21** | 0.35 | L2–L8 |
| 4 直接父 | 46–100 | **22–100** | 0.40 | L1–L8 |

好处：**首次验证（epoch 40）之前课程已完整开启**，早停窗口（40~46）内模型始终处于「全 8 层」状态，L1 有 ~20 个 epoch 的训练量。配置见 `config_files/*/config.test.hidden.hardDc.json`。

### 6.6 事故 4：同秒启动 → 同一目录互相覆盖（严重）

`parse_config.py` 的保存目录是 `save_dir/<name>/<%m%d_%H%M%S>/models`，**没有唯一化**。
两个同类 run 若在同一秒创建 ConfigParser，就会写进同一个目录，`model_best.pth` 后写者覆盖先写者。

实测两处：

| 数据集 | 作业 | 结果 |
|---|---|---|
| MeSH | 133629 (C) 与 133630 (D)，均 `Start=2026-11-20T22:04:44` → 同写 `1120_220500` | 该目录唯一留下的 `model_best.pth` 指纹是 `epoch=44, monitor_best=0.60219` → **属于 C**；**D 的权重已丢失**；`mesh_D_top10_shard*.json` 实为 C 的结果（已改名 `mesh_C_*`） |
| Verb | 134107 (D 续训) 与 134109 (Dc)，均 `job start: Sun Nov 22 09:54:15` → 同写 `1122_095425` | 该目录 `info.log` 里既有 `epoch: 1,2,3…`（Dc）也有 `epoch: 41…`（D 续训），`model_best.pth`(11-22 13:43:52) 属 **D**；已把目录整体改名为 `1122_095425_verbD_134107` 并为仍在跑的 134109 重建空目录（其 `mkdir/保存` 走相对旧路径 → 之后写进新空目录，不再覆盖 D） |

**修复**：

1. 所有 `slurm/train_*.slurm` 的 `python train.py` 追加 `--suffix <tag>`（`parse_config.py` 已支持把 suffix 拼进时间戳目录名），例如 `1120_220500_mesh_D`；
2. `slurm/watch_and_dump.sh` 新增第 8 个参数 `SUF`，按目录名后缀精确定位 `model_best.pth`（比 `exp_name` 可靠，因为同名 exp_name 会出现在多个 run 里）；
3. 判权重归属的可靠指纹只有 checkpoint 里的 `epoch` + `monitor_best`（`config.json` 的 `exp_name` 会被后写者覆盖，日志里也不一定打印保存路径）。

### 6.7 MeSH 测试集结果（A / B / C）

| 组 | 作业 | hit@10 | MRR | MRR@10(scaled) | 父错子对 | 父对子错 | 父子皆错 | 报告 |
|---|---|---|---|---|---|---|---|---|
| A baseline | 126630 | **0.8339** | 0.6695 | 0.6695 | 4402 (61.41%) | 931 (12.99%) | 1835 (25.60%) | `mesh_error_analysis.md` |
| B ρ=0.30 层[1,2,3] | 133628 | 0.7668 | 0.2439 | 0.5797 | 2957 (40.36%) | 1595 (21.77%) | 2774 (37.87%) | `mesh_B_error_analysis.md` |
| C ρ=0.30 层[1..8] | 133629 | 0.7485 | 0.2248 | 0.5720 | 3194 (43.41%) | 1520 (20.66%) | 2644 (35.93%) | `mesh_C_error_analysis.md` |

（B/C 报告的“零”节只有 hit@k / MRR / MR，因为 `--title` 未带 baseline 时不给论文对齐列；
A 的 `mrr_scaled_10` 与 `mrr` 数值相同是 A 报告早期版本的算法口径差异，需以 B/C 格式重跑后再横向比。）

**结论（暂定）**：随机插负例的 baseline 在 hit@10 上明显更好；B/C 的**父错子对占比从 61.4% 降到 40~43%**（正是改造目标），
但整体排序被拖差（hit@10 -0.07 ~ -0.09），说明「用真父节点当负例」确实压低了父节点得分，也伤到了同父兄弟的相对次序。
需 D（ρ=0.40 随机 10/20/32/45/100 层可用）与 Dc 结果才能判断是否存在能兼顾两头的配比。

### 6.8 口径澄清（重要，防止横向对比出错）

1. **Dc 是独立的第 5 组**，不是 D 的替代版本。D＝任务书原课程表（10/20/32/45/100），Dc＝按实际预算压缩的课程表（5/10/15/21/100）；
   两者是两条独立曲线，比较它们本身就是 "压缩课程是否值得" 的问题。
2. **有效训练预算只有 ~43 epoch**：`trainer.early_stop = 2`、`monitor = max val_mrr_scaled_10`，
   而验证从 **epoch 40** 才开始 → 所有组都在 ep42~45 早停。`trainer.epochs`（mesh=60 / verb=100）实际用不到。
3. **D 原表在本预算下从未开启 L1**：日志显示 `epoch 43: layers=[2,...,8]`、`layer_hist={1: 0, ...}`，
   因为原表的 L1 要到 epoch 46 才开放 → D 是一条 "分阶段但无直接父负例" 的曲线。Dc 在 ep22 就开 L1，这是压缩的唯一目的。
4. **同一列才能比**：`val_mrr_scaled_10`（训练日志）与测试集 `mrr_scaled_10`（错误分析报告）是两个口径，不可混用。
   - 只有 val：MeSH A 0.7195 > D 0.6630（ep40 峰值 `0.66299`）> Dc 0.6259(仍在升) > B 0.6043 > C 0.6022；
   - 只有 test：MeSH A 0.6695 > B 0.5797 > C 0.5720；**D 无权重 → 无测试集结果**。
5. `1120_220500` 的 checkpoint `epoch=44, monitor_best=0.6021956705908927` 与 C 日志的 val 峰值完全一致 → 再次确认该权重属于 C。

### 6.9 实验矩阵现状（1 seed）

| 组 | 定义 | MeSH | Verb |
|---|---|---|---|
| A | baseline，`hard_ratio=0` | ✅ `1108_153108` | ✅ `1110_211557` |
| B | ρ=0.30，固定 L1–L3 | ✅ `1120_215306` | 🔄 `1123_110658_verb_B`（134937） |
| C | ρ=0.30，固定 L1–L8 | ✅ `1120_220500` | 🔄 `1123_110658_verb_C`（134938） |
| D | 课程原表 10/20/32/45/100 | 🔄 `1123_110658_mesh_D`（134936，重跑） | ✅ `1122_095425_verbD_134107` |
| Dc | 课程压缩 5/10/15/21/100（第 5 组） | 🔄 `1122_111036` | 🔄 `1122_095425` |

**2026-11-23 11:06 已补齐三个 run**（均带 `--suffix`，目录唯一）：

| 作业 | 组 | 节点 | 配置 | ETA |
|---|---|---|---|---|
| 134936 | MeSH-D（重跑，权重曾丢失） | main-6 (4090) | `hardD` | ≈25–28 h |
| 134937 | Verb-B | main-12 (2080Ti，`--tbs 300`) | `hardB` | ≈37 h 训练 + 6.5 h dump |
| 134938 | Verb-C | main-12 (2080Ti，`--tbs 300`) | `hardC` | 同上 |

均已挂 `watch_and_dump.sh` 守卫（第 8 参 `SUF`），早停后自动 `scancel` 并提交 dump。
任务书另要求每组 3 seed（`-n 3`），当前全部为 1 seed。
