# [导读] trainer.py —— 训练循环 + 全量测试（文件里其实有 4 个 Trainer 类）
#   公共部分：
#     rearrange()        把一个 query 的候选分数重排成「正例在前、负例在后」+ labels
#     Trainer            基类：存 metrics / pre_metric，以及 _eval_metrics
#   三个子类（config['arch']['trainer'] 选择用哪个）：
#     TrainerTEMP      → 只有 L_a（整体对不对），score 取 [CLS]+MLP，装的是 TEMPCompletion
#     TrainerHidden    → 三任务 L_a + L_p + L_c，装 TacoPromptHidden ← 论文主模型的默认选择
#     TrainerChain     → 三段式：先 Parent、再 Child（拿父的硬答案拼接）、最后 All，装 TacoPromptChain
#   一个容易踩的坑：三个类各自拷了一份 _test()，代码 95% 重复但打分调用不同：
#     TEMP   → model.scorer(p_idx, q_idx, c_idx, test=True)
#     Hidden → model.scorer(p_idx, q_idx, c_idx, pseudo_pct, pseudo_pct, pseudo_pct)
#     Chain  → 一个 query 跑 3 轮 scorer（p → c → a），所以链式推理比其他两个慢 3 倍
#   验证从 epoch 40 才开始（前 40 epoch 先让模型把基础结构学会），所以 epochs 必须 >= 40，
#   否则 BaseTrainer.train() 结尾的 best_state 根本没被赋值，末尾 load_state_dict(best_state) 会报 UnboundLocalError。
#   与 TaxoDiff 的差别：TaxoDiff 的 trainer 在这里多接了扩散负样本分支（L_DM）和 requires_grad 开关。

import itertools
import random
import time

import wandb
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch

from base import BaseTrainer
from model.loss import *
from functools import partial

MAX_CANDIDATE_NUM = 100000


def rearrange(energy_scores, candidate_position_idx, true_position_idx):
    # [旁批] ★ 两个作用，都是为了给 obtain_ranks 腾路：
    #   ① 生成 labels：哪个候选是真值 → 1，其余 → 0
    #   ② 把正例整块挤到最前面（energy_scores[correct] 在前），这样 labels 就是 [1,1,...,0,0]，
    #      obtain_ranks 里用 re.finditer([0,1]) 切块时才不会「把一个 query 拆成多块」。
    #   复杂度：|candidates| × |true| 的 Python 两层循环 —— 51159 × 1.5 ≈ 7.7 万次比较，每 query 一次。
    #   [旁批?] 若真值是空列表（检索阶段全部漏检），下面 .any(0) 得到全 False → correct 为空 →
    #     labels 全 0 → positive_relations 为空 → ranks 为空且不报错。infer_hidden_topk.py 里
    #     还额外加了 `if true_position_idx == []: true_position_idx = [[]]` 的补丁。
    tmp = np.array([[x == y for x in candidate_position_idx] for y in true_position_idx]).any(0)
    correct = np.where(tmp)[0]
    incorrect = np.where(~tmp)[0]
    labels = torch.cat((torch.ones(len(correct)), torch.zeros(len(incorrect)))).int()
    energy_scores = torch.cat((energy_scores[correct], energy_scores[incorrect]))
    return energy_scores, labels


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.test_batch_size = config['trainer']['test_batch_size']
        self.is_infonce_training = config['loss'].startswith("info_nce")
        self.is_focal_loss = config['loss'].startswith("FocalLoss")
        self.data_loader = data_loader
        self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_mode = self.config['lr_scheduler']['args']['mode']  # "min" or "max"
        self.log_step = len(data_loader) // 10
        # self.log_step = 1
        self.pre_metric = pre_metric
        self.writer.add_text('Text', 'Model Architecture: {}'.format(self.config['arch']), 0)
        self.writer.add_text('Text', 'Training Data Loader: {}'.format(self.config['train_data_loader']), 0)
        self.writer.add_text('Text', 'Loss Function: {}'.format(self.config['loss']), 0)
        self.writer.add_text('Text', 'Optimizer: {}'.format(self.config['optimizer']), 0)

    def _eval_metrics(self, output, target):
        # [旁批] 两步走：先把分数+标签变成「每个真值的名次」（all_ranks），再把名次交给各个指标函数。
        #   这套代码里所有指标都只吃 all_ranks，不看分数，所以指标完全由「排序」决定。
        acc_metrics = np.zeros(len(self.metrics))
        all_ranks = self.pre_metric(output, target)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(all_ranks)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics


class TrainerTEMP(Trainer):
    """
    Trainer class, for our proposed model on taxonomy completion task

    Note:
        Inherited from Trainer and overrides training/evaluation methods.
    """
    # [旁批] TEMP 的 Trainer（baseline）。相比论文主模型，它只用了 L_a（整体对不对），
    #   完全没用 pt / ct 两个细粒度标签 —— 这是它作为 baseline 的主要短板。
    #   它的 _test 里打分调用是 model.scorer(..., test=True)，取 [CLS] 向量过 MLP，不是 [MASK] logits。

    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerTEMP, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader,
                                          lr_scheduler)
        self.config = config
        self.train_data_loader = None
        self.l_p = config['trainer']['lp']
        self.l_c = config['trainer']['lc']
        self.l_b = config['trainer']['lb']
        self.l_w = config['trainer']['lw']
        self.l_angle = config['trainer']['l_angle']
        self.anneal = config['trainer']['anneal']
        # [旁批?] 三个 Trainer 类都把这行写死成 2，而 config 里 accumulation_iters 是 8。
        #   后果：train.py 算 LR warmup 的 total_steps 时用的是 8（以为每 8 个 batch 才走一步），
        #   而真实优化步数是它的 4 倍 → 余弦退火曲线与真实步数错位。不影响能否跑通，影响收敛形状。
        self.accumulation_iters = 2
        dataset = self.data_loader.dataset
        self.candidate_positions = data_loader.dataset.all_edges
        if len(self.candidate_positions) > MAX_CANDIDATE_NUM:
            valid_pos = set(itertools.chain.from_iterable(dataset.valid_node2pos.values()))
            valid_neg = list(set(self.candidate_positions).difference(valid_pos))
            valid_sample_size = max(MAX_CANDIDATE_NUM - len(valid_pos), 0)
            self.valid_candidate_positions = random.sample(valid_neg, valid_sample_size) + list(valid_pos)
        else:
            self.valid_candidate_positions = self.candidate_positions
        self.valid_candidate_positions = self.candidate_positions

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        scaler = GradScaler()
        self.optimizer.zero_grad()
        cached_hgcn_emb = False
        # 推进负采样课程（只对实现了 set_epoch 的 dataset 生效）
        if hasattr(self.data_loader.dataset, 'set_epoch'):
            self.data_loader.dataset.set_epoch(epoch)
        for batch_idx, batch in enumerate(self.data_loader):
            qs, ps, cs, bs, ws, pct, pt, ct, bt, wt, pe, ce, be, we, pit, qit, cit = batch
            qs = {k: v.to(self.device) for k, v in qs.items()}
            ps = {k: v.to(self.device) for k, v in ps.items()}
            cs = {k: v.to(self.device) for k, v in cs.items()}
            bs = {k: v.to(self.device) for k, v in bs.items()}
            ws = {k: v.to(self.device) for k, v in ws.items()}
            pe = pe.to(self.device)
            ce = ce.to(self.device)
            be = be.to(self.device)
            we = we.to(self.device)
            pit = pit.to(self.device)
            qit = qit.to(self.device)
            cit = cit.to(self.device)
            with autocast():
                score = self.model(qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit)
                
                pct = pct.to(self.device)
                pt = pt.to(self.device)
                ct = ct.to(self.device)
                bt = bt.to(self.device)
                wt = wt.to(self.device)
                # [旁批?] TEMP 只用了第二个 loss：self.loss 是一个 tuple（从 model/loss.py 看的），
                #   索引 [2] 取的是 bce_loss。而且它只算 pct（整体对不对），
                #   没有用 pt / ct 两个细粒度标签 —— 这是 TEMP 比 TacoPrompt 弱的原因之一。
                loss_label = self.loss[2](score.squeeze(1), pct)
                loss = loss_label / self.accumulation_iters

            scaler.scale(loss).backward()
            if (batch_idx + 1) % self.accumulation_iters == 0 or batch_idx == len(self.data_loader):
                total_loss += loss.item()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Total: {:.6f} Loss: {:.6f}'.format(
                        epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                               100.0 * batch_idx / len(self.data_loader), total_loss, loss.item()))
                wandb.log(
                    {'total_loss': total_loss,
                     'loss': loss.item(),
                     'step': (epoch - 1) * len(self.data_loader) + batch_idx})

        log = {'epoch avg loss': total_loss / len(self.data_loader)}

        # 本 epoch 的结构化负采样统计（hard_ratio=0 时 hard 恒为 0，不打印）
        if getattr(self.data_loader.dataset, 'hard_stats', None):
            st = self.data_loader.dataset.hard_stats
            if st['hard'] > 0:
                self.logger.info(f"[hard-neg] hard={st['hard']} rand={st['rand']} "
                                 f"shortfall={st['shortfall']} layer_hist={st['layer_hist']}")

        # Validation stage
        if self.do_validation and epoch >= 40:           #前 40epoch 模型快速学习分类图的基础结构，从第40epoch才开始跑验证集（判断是否早停），节省算力。
            if epoch == 40:
                self._save_checkpoint(epoch=epoch, save_best=False)
            val, leaf_val, nonleaf_val = self._test('validation')
            val_log = {'val_metrics': val}
            leaf_val_log = {'leaf_val_metrics': leaf_val}
            nonleaf_val_log = {'nonleaf_val_metrics': nonleaf_val}
            log = {**log, **val_log, **leaf_val_log, **nonleaf_val_log}
            wandb_log = {k.__name__: v for (k, v) in zip(self.metrics, val_log['val_metrics'])}
            wandb_log.update({f'leaf_{k.__name__}': v for (k, v) in zip(self.metrics, leaf_val)})
            wandb_log.update({f'nonleaf_{k.__name__}': v for (k, v) in zip(self.metrics, nonleaf_val)})
            wandb_log.update({'epoch': epoch})
            wandb.log(wandb_log)

        return log

    def _test(self, mode, gpu=True):
        assert mode in ['test', 'validation']
        model = self.model if gpu else self.model.cpu()

        para = isinstance(model, torch.nn.DataParallel)
        if para:
            model = model.module

        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            if mode == 'test':
                queries = dataset.test_node_list
                node2pos = dataset.test_node2pos
                graph = dataset.test_holdout_subgraph
                candidate_positions = self.candidate_positions
                taxon2id = dataset.test_taxon2id
                id2taxon = dataset.test_id2taxon
            else:
                queries = dataset.valid_node_list
                node2pos = dataset.valid_node2pos
                graph = dataset.valid_holdout_subgraph
                candidate_positions = self.valid_candidate_positions
                taxon2id = dataset.valid_taxon2id
                id2taxon = dataset.valid_id2taxon
            taxon2id.keys()
            pseudo_leaf_id = taxon2id[dataset.pseudo_leaf_node]
            pseudo_root_id = taxon2id[dataset.pseudo_root_node]
            taxon2allemb_id = model.train_taxon2allemb_id  
            taxon2all_node_id = model.taxon2id  
            self.logger.info(f'number of candidate positions: {len(candidate_positions)}')

            # start per query prediction
            all_ranks = []
            leaf_queries = []
            leaf_ranks, nonleaf_ranks = [], []

            # begin per query prediction
            debug = False
            eval_queries = queries[:100] if mode == 'validation' else queries
            # generate desc representations for all query nodes
            q_id = torch.tensor([taxon2all_node_id[query] for i, query in enumerate(eval_queries)])

            # find leaf and nonleaf
            # [旁批] ★ 叶子 / 非叶子的判定：node2pos[q] 里【所有】位置的 child 都是 pseudo_leaf → leaf query。
            #   因为 leaf 只需要猜对父（子固定是伪叶），非叶要同时猜对父子，两个子任务难度差很多，
            #   所以论文表格分成 Leaf / Non-leaf 两栏分别报。
            num = 0 
            for i, query in enumerate(eval_queries):
                poses = node2pos[query]
                flag = True
                for pos in poses:
                    if pos[1] != dataset.pseudo_leaf_node:
                        flag = False
                        break
                if flag:
                    leaf_queries.append(query)
                    num += 1
            print(num)

            for i, query in tqdm(enumerate(eval_queries), desc=mode, total=len(eval_queries)):
                batched_energy_scores = []
                for edges in mit.sliced(candidate_positions, batch_size):
                    edges = list(edges)
                    ps, cs = zip(*edges)

                    with autocast():
                        p_idx = torch.tensor([taxon2allemb_id[n] for n in ps]).to(self.device)
                        c_idx = torch.tensor([taxon2allemb_id[n] for n in cs]).to(self.device)
                        q_idx = torch.tensor([q_id[i] for j in range(p_idx.shape[0])])
                        # [旁批] TEMP 的打分：prompt 是 "路径,q,c [SEP] q描述"，取 [CLS] 向量过 MLP。
                        #   注意 test=True 时 get_text_emb 内部把 batch 改成 4096（只对本方法生效），
                        #   而外面仍按 test_batch_size 切候选 —— 两个 batch 概念不同，别搞混。
                        scores = model.scorer(p_idx, q_idx, c_idx, test=True).squeeze(1)
                    batched_energy_scores.append(scores)
                batched_energy_scores_cat = torch.cat(batched_energy_scores)

                # Total
                batched_energy_scores_cat, labels = rearrange(batched_energy_scores_cat, candidate_positions,
                                                              node2pos[query])
                ranks = self.pre_metric(batched_energy_scores_cat, labels)
                all_ranks.extend(ranks)

                if query in leaf_queries:
                    leaf_ranks.extend(ranks)
                else:
                    nonleaf_ranks.extend(ranks)

            print(all_ranks)
            total_metrics = [metric(all_ranks) for metric in self.metrics]
            leaf_metrics = [metric(leaf_ranks) for metric in self.metrics]
            nonleaf_metrics = [metric(nonleaf_ranks) for metric in self.metrics]

            print(f'leaf_metrics:{leaf_metrics}')
            print(f'nonleaf_metrics:{nonleaf_metrics}')
        return total_metrics, leaf_metrics, nonleaf_metrics


class TrainerHidden(Trainer):
    """
    Trainer class, for our proposed model on taxonomy completion task

    Note:
        Inherited from Trainer and overrides training/evaluation methods.
    """
    # [旁批] ★ 这个类的 _test() 就是论文表格的产出地，读的时候抓五个点：
    #   ① candidate_positions = self.candidate_positions —— 测试用的候选池是【训练子图】的 all_edges，
    #      不是 test 自己的。因为 test 节点已被从 core_subgraph 移除，它们不会出现在候选里。
    #   ② leaf / nonleaf 划分：node2pos[q] 里所有位置的 child 都是 pseudo_leaf → leaf query。
    #      （因为 leaf 只需要猜对父，难negative分布和非叶完全不同，所以分开报。）
    #   ③ 打分只取 score[:, 0]（All 位），Parent/Child 两列在测试时丢掉。
    #   ④ rearrange 把真值位置挤到前面生成 labels，再交给 obtain_ranks 算名次。
    #   ⑤ print(all_ranks) 是唯一能拿到「每个 query 的名次」的地方，不会落盘成文件。
    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerHidden, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader,
                                                    lr_scheduler)
        self.config = config
        self.train_data_loader = None
        self.l_main = config['trainer']['lm']
        self.l_p = config['trainer']['lp']
        self.l_c = config['trainer']['lc']
        self.accumulation_iters = 2
        dataset = self.data_loader.dataset
        self.candidate_positions = data_loader.dataset.all_edges
        if len(self.candidate_positions) > MAX_CANDIDATE_NUM:
            valid_pos = set(itertools.chain.from_iterable(dataset.valid_node2pos.values()))
            valid_neg = list(set(self.candidate_positions).difference(valid_pos))
            valid_sample_size = max(MAX_CANDIDATE_NUM - len(valid_pos), 0)
            self.valid_candidate_positions = random.sample(valid_neg, valid_sample_size) + list(valid_pos)
        else:
            self.valid_candidate_positions = self.candidate_positions
        self.valid_candidate_positions = self.candidate_positions

    def _inconsistency_loss(self, pc_score, p_score, c_score, pe, ce):
        # [旁批] 死代码：整个文件搜 _inconsistency_loss 只有这一处定义，没有任何调用。
        #   它的想法是「让 p_prob * c_prob 与 pc_prob 保持一致」（父对×子对 ≍ 整体对），
        #   用一致性正则约束三个子任务。但实际训练根本没启用。
        #   [旁批?] 注意下面第三行写成了 c_prob = torch.sigmoid(c_prob)（自引用），
        #   即使启用也会报 UnboundLocalError。要试这个想法的话，先改成 torch.sigmoid(c_score)。
        pc_prob = torch.sigmoid(pc_score)
        p_prob = torch.sigmoid(p_score)
        c_prob = torch.sigmoid(c_score)

        all = pe & ce
        p_only = pe & ~ce
        c_only = ce & ~pe
        
        pc_prob_all = torch.masked_select(pc_prob, all)
        p_prob_all = torch.masked_select(p_prob, all)
        c_prob_all = torch.masked_select(c_prob, all)
        inconsistency_loss_all = torch.abs(p_prob_all * c_prob_all - pc_prob_all) * (all.shape[0] / pc_score.shape[0])

        pc_prob_p_only = torch.masked_select(pc_prob, p_only)
        p_prob_p_only = torch.masked_select(p_prob, p_only)
        inconsistency_loss_p_only = torch.abs(p_prob_p_only - pc_prob_p_only) * (p_only.shape[0] / pc_score.shape[0])

        pc_prob_c_only = torch.masked_select(pc_prob, c_only)
        c_prob_c_only = torch.masked_select(c_prob, c_only)
        inconsistency_loss_c_only = torch.abs(c_prob_c_only - pc_prob_c_only) * (c_only.shape[0] / pc_score.shape[0])
        
        inconsistency_loss = torch.cat([inconsistency_loss_all, inconsistency_loss_p_only, inconsistency_loss_c_only])
        return torch.mean(inconsistency_loss)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        scaler = GradScaler()
        self.optimizer.zero_grad()
        # 推进负采样课程（只对实现了 set_epoch 的 dataset 生效）
        if hasattr(self.data_loader.dataset, 'set_epoch'):
            self.data_loader.dataset.set_epoch(epoch)
        for batch_idx, batch in enumerate(self.data_loader):
            qs, ps, cs, bs, ws, pct, pt, ct, bt, wt, pe, ce, be, we, pit, qit, cit = batch
            qs = {k: v.to(self.device) for k, v in qs.items()}
            ps = {k: v.to(self.device) for k, v in ps.items()}
            cs = {k: v.to(self.device) for k, v in cs.items()}
            bs = {k: v.to(self.device) for k, v in bs.items()}
            ws = {k: v.to(self.device) for k, v in ws.items()}
            pe = pe.to(self.device)
            ce = ce.to(self.device)
            be = be.to(self.device)
            we = we.to(self.device)
            pit = pit.to(self.device)
            qit = qit.to(self.device)
            cit = cit.to(self.device)
            pct = pct.to(self.device)
            pt = pt.to(self.device)
            ct = ct.to(self.device)
            with autocast():
                score = self.model(qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct)
                pc_score = score[:, 0]
                p_score = score[:, 1]
                c_score = score[:, 2]
                loss_pc = self.loss(pc_score, pct)
                loss_p = self.loss(torch.masked_select(p_score, pe),
                                   torch.masked_select(pt, pe))
                loss_c = self.loss(torch.masked_select(c_score, ce),
                                  torch.masked_select(ct, ce))
                # [旁批] ★★ 这就是论文 Eq.10：L = λa·La + λp·Lp + λc·Lc（λ 来自 config 的 lm/lp/lc，默认都是 1）。
                #   La：全部行都算（整体对不对）
                #   Lp / Lc：只用 masked_select 保留 pe / ce 为 True 的行（即 p / c 有描述文本的行）
                #   为什么要掩码：pseudo_root / pseudo_leaf 的 description 是空串，
                #   在 prompt 里体现为 <Parent>: （空），拿这种样本学「父对不对」没意义。
                mt_loss = (self.l_main * loss_pc + self.l_p * loss_p + self.l_c * loss_c) / self.accumulation_iters
                loss = mt_loss
            scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_iters == 0 or batch_idx == len(self.data_loader):
                total_loss += loss.item()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                               100.0 * batch_idx / len(self.data_loader), loss.item()
                    ))
                wandb.log(
                    {'total_loss': total_loss,
                     'loss': loss.item(),
                     'step': (epoch - 1) * len(self.data_loader) + batch_idx})
        log = {'epoch avg loss': total_loss / len(self.data_loader)}

        # 本 epoch 的结构化负采样统计
        if getattr(self.data_loader.dataset, 'hard_stats', None):
            st = self.data_loader.dataset.hard_stats
            if st['hard'] > 0:
                self.logger.info(f"[hard-neg] hard={st['hard']} rand={st['rand']} "
                                 f"shortfall={st['shortfall']} layer_hist={st['layer_hist']}")

        # Validation stage
        if epoch in [40]:
            self._save_checkpoint(epoch=epoch, save_best=False)
        if self.do_validation and epoch >= 40:
            if epoch == 40:
                self._save_checkpoint(epoch=epoch, save_best=False)
            val, leaf_val, nonleaf_val = self._test('validation')
            val_log = {'val_metrics': val}
            leaf_val_log = {'leaf_val_metrics': leaf_val}
            nonleaf_val_log = {'nonleaf_val_metrics': nonleaf_val}
            log = {**log, **val_log, **leaf_val_log, **nonleaf_val_log}
            wandb_log = {k.__name__: v for (k, v) in zip(self.metrics, val_log['val_metrics'])}
            wandb_log.update({f'leaf_{k.__name__}': v for (k, v) in zip(self.metrics, leaf_val)})
            wandb_log.update({f'nonleaf_{k.__name__}': v for (k, v) in zip(self.metrics, nonleaf_val)})
            wandb_log.update({'epoch': epoch})
            wandb.log(wandb_log)
        return log

    def _test(self, mode, gpu=True):
        assert mode in ['test', 'validation']
        model = self.model if gpu else self.model.cpu()

        para = isinstance(model, torch.nn.DataParallel)
        if para:
            model = model.module

        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            if mode == 'test':
                queries = dataset.test_node_list
                node2pos = dataset.test_node2pos
                graph = dataset.test_holdout_subgraph
                candidate_positions = self.candidate_positions
                taxon2id = dataset.test_taxon2id
                id2taxon = dataset.test_id2taxon
            else:
                queries = dataset.valid_node_list
                node2pos = dataset.valid_node2pos
                graph = dataset.valid_holdout_subgraph
                candidate_positions = self.valid_candidate_positions
                taxon2id = dataset.valid_taxon2id
                id2taxon = dataset.valid_id2taxon
            taxon2id.keys()
            pseudo_leaf_id = taxon2id[dataset.pseudo_leaf_node]
            pseudo_root_id = taxon2id[dataset.pseudo_root_node]
            # id for test
            taxon2allemb_id = model.train_taxon2allemb_id 
            taxon2all_node_id = model.taxon2id  
            self.logger.info(f'number of candidate positions: {len(candidate_positions)}')
            # [旁批] 上面这个日志就是你判断「测试有多重」的依据：SemEval-Verb 是 51159。
            #   每个 query 都要对 51159 个位置跑一次 forward → 1000 个 query 就是 5000 多万次打分。
            #   而训练时每个 query 只看 16 个候选（1 正 + 15 负），两个量级差了 3000 倍。

            all_ranks = []
            leaf_queries = []
            leaf_ranks, nonleaf_ranks = [], []

            eval_queries = queries[:100] if mode == 'validation' else queries
            # generate desc representations for all query nodes
            q_id = torch.tensor([taxon2all_node_id[query] for i, query in enumerate(eval_queries)])

            # find leaf and nonleaf
            # redefine nonleaf
            num = 0
            for i, query in enumerate(eval_queries):
                poses = node2pos[query]
                flag = True
                for pos in poses:
                    if pos[1] != dataset.pseudo_leaf_node:
                        flag = False
                        break
                if flag:
                    leaf_queries.append(query)
                    num+=1
            print(num)
            
            for i, query in tqdm(enumerate(eval_queries), desc=mode, total=len(eval_queries)):
                batched_energy_scores = []
                for edges in mit.sliced(candidate_positions, batch_size):
                    edges = list(edges)
                    ps, cs = zip(*edges)

                    with autocast():
                        p_idx = torch.tensor([taxon2allemb_id[n] for n in ps]).to(self.device)
                        c_idx = torch.tensor([taxon2allemb_id[n] for n in cs]).to(self.device)
                        q_idx = torch.tensor([q_id[i] for j in range(p_idx.shape[0])])
                        # [旁批] ★ 三个 pseudo_pct 分别对应 prompt 里的 All / Parent / Child 三个 [MASK]。
                        #   测试时这三个值根本不影响分数（它们只影响 ground-truth 那句的拼接，而测试不用 labels）。
                        #   然后只取第 0 列（All 位），Parent/Child 两列丢弃 —— 因为排序只需要一个分数。
                        pseudo_pct = torch.tensor([1 for j in range(p_idx.shape[0])])
                        scores = model.scorer(p_idx, q_idx, c_idx, pseudo_pct, pseudo_pct, pseudo_pct)
                        scores = scores[:, 0]
                    batched_energy_scores.append(scores)
                batched_energy_scores_cat = torch.cat(batched_energy_scores)

                # Total
                batched_energy_scores_cat, labels = rearrange(batched_energy_scores_cat, candidate_positions,
                                                              node2pos[query])
                ranks = self.pre_metric(batched_energy_scores_cat, labels)
                all_ranks.extend(ranks)

                if query in leaf_queries:
                    leaf_ranks.extend(ranks)
                else:
                    nonleaf_ranks.extend(ranks)

            print(all_ranks)
            total_metrics = [metric(all_ranks) for metric in self.metrics]
            leaf_metrics = [metric(leaf_ranks) for metric in self.metrics]
            nonleaf_metrics = [metric(nonleaf_ranks) for metric in self.metrics]

            print(f'leaf_metrics:{leaf_metrics}')
            print(f'nonleaf_metrics:{nonleaf_metrics}')
        return total_metrics, leaf_metrics, nonleaf_metrics
    
    
class TrainerChain(Trainer):
    """
    Trainer class, for our proposed model on taxonomy completion task

    Note:
        Inherited from Trainer and overrides training/evaluation methods.
    """
    # [旁批] TrainerChain 与 TrainerHidden 的差别只在「怎么算出三个分数」：
    #   训练：Hidden 是 1 次 forward 拿到 (B,3) 三列；Chain 是 3 次 forward（p → c → a），
    #        后两次依赖前一次输出的硬答案（yes/no 文本拼进 prompt）。
    #   测试：Hidden 只取 score[:,0]；Chain 要跑三遍 scorer，所以慢约 3 倍。
    #   _train_epoch 里三个 loss 是分别 backward 的（不是相加后一次 backward），
    #   梯度在参数上自然累加，数学上与「先求和再 backward」等价，但显存峰值更低。

    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerChain, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader,
                                                    lr_scheduler)
        self.config = config
        self.train_data_loader = None
        self.l_main = config['trainer']['lm']
        self.l_p = config['trainer']['lp']
        self.l_c = config['trainer']['lc']
        self.accumulation_iters = 2
        dataset = self.data_loader.dataset
        self.candidate_positions = data_loader.dataset.all_edges
        if len(self.candidate_positions) > MAX_CANDIDATE_NUM:
            valid_pos = set(itertools.chain.from_iterable(dataset.valid_node2pos.values()))
            valid_neg = list(set(self.candidate_positions).difference(valid_pos))
            valid_sample_size = max(MAX_CANDIDATE_NUM - len(valid_pos), 0)
            self.valid_candidate_positions = random.sample(valid_neg, valid_sample_size) + list(valid_pos)
        else:
            self.valid_candidate_positions = self.candidate_positions
        self.valid_candidate_positions = self.candidate_positions

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        scaler = GradScaler()
        self.optimizer.zero_grad()
        # 推进负采样课程（只对实现了 set_epoch 的 dataset 生效）
        if hasattr(self.data_loader.dataset, 'set_epoch'):
            self.data_loader.dataset.set_epoch(epoch)
        for batch_idx, batch in enumerate(self.data_loader):
            qs, ps, cs, bs, ws, pct, pt, ct, bt, wt, pe, ce, be, we, pit, qit, cit = batch
            qs = {k: v.to(self.device) for k, v in qs.items()}
            ps = {k: v.to(self.device) for k, v in ps.items()}
            cs = {k: v.to(self.device) for k, v in cs.items()}
            bs = {k: v.to(self.device) for k, v in bs.items()}
            ws = {k: v.to(self.device) for k, v in ws.items()}
            pe = pe.to(self.device)
            ce = ce.to(self.device)
            be = be.to(self.device)
            we = we.to(self.device)
            pit = pit.to(self.device)
            qit = qit.to(self.device)
            cit = cit.to(self.device)
            pct = pct.to(self.device)
            pt = pt.to(self.device)
            ct = ct.to(self.device)
            mt_loss = 0
            with autocast():
                # [旁注] ★ chain 的训法：同一个 batch 前向 3 次、反传 3 次，梯度累加在同一个 optimizer 上。
                #   顺序不能变：p_ans 要喂给 'c'，p_ans + c_ans 都要喂给 'a'。
                p_score, p_ans = self.model(qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct, subtask='p')
                loss_p = self.l_p * self.loss(p_score.squeeze(1),
                                   pt) / self.accumulation_iters
                mt_loss += loss_p.item()
                scaler.scale(loss_p).backward()

                c_score, c_ans = self.model(qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct, p_ans=p_ans.squeeze(1), subtask='c')
                # [旁批] 注意这次 forward 把第一步的硬答案 p_ans 当作文本拼进了 "Child: ... Parent: {yes/no} ..."
                loss_c = self.l_c * self.loss(c_score.squeeze(1),
                                   ct) / self.accumulation_iters
                mt_loss += loss_c.item()
                scaler.scale(loss_c).backward()

                pc_score, pc_ans = self.model(qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct, p_ans=p_ans.squeeze(1), c_ans=c_ans.squeeze(1), subtask='a')
                loss_pc = self.l_main * self.loss(pc_score.squeeze(1), pct) / self.accumulation_iters
                mt_loss += loss_pc.item()
                scaler.scale(loss_pc).backward()
                
            if (batch_idx + 1) % self.accumulation_iters == 0 or batch_idx == len(self.data_loader):
                total_loss += mt_loss
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                               100.0 * batch_idx / len(self.data_loader),  mt_loss
                    ))
                wandb.log(
                    {'total_loss': total_loss,
                     'loss': mt_loss,
                     'step': (epoch - 1) * len(self.data_loader) + batch_idx})
        log = {'epoch avg loss': total_loss / len(self.data_loader)}

        # 本 epoch 的结构化负采样统计
        if getattr(self.data_loader.dataset, 'hard_stats', None):
            st = self.data_loader.dataset.hard_stats
            if st['hard'] > 0:
                self.logger.info(f"[hard-neg] hard={st['hard']} rand={st['rand']} "
                                 f"shortfall={st['shortfall']} layer_hist={st['layer_hist']}")

        # Validation stage
        if self.do_validation and epoch >= 40:
            if epoch == 40:
                self._save_checkpoint(epoch=epoch, save_best=False)
            val, leaf_val, nonleaf_val = self._test('validation')
            val_log = {'val_metrics': val}
            leaf_val_log = {'leaf_val_metrics': leaf_val}
            nonleaf_val_log = {'nonleaf_val_metrics': nonleaf_val}
            log = {**log, **val_log, **leaf_val_log, **nonleaf_val_log}
            wandb_log = {k.__name__: v for (k, v) in zip(self.metrics, val_log['val_metrics'])}
            wandb_log.update({f'leaf_{k.__name__}': v for (k, v) in zip(self.metrics, leaf_val)})
            wandb_log.update({f'nonleaf_{k.__name__}': v for (k, v) in zip(self.metrics, nonleaf_val)})
            wandb_log.update({'epoch': epoch})
            wandb.log(wandb_log)
        return log

    def _test(self, mode, q_val=100, gpu=True):
        assert mode in ['test', 'validation']
        model = self.model if gpu else self.model.cpu()

        para = isinstance(model, torch.nn.DataParallel)
        if para:
            model = model.module

        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            if mode == 'test':
                queries = dataset.test_node_list
                node2pos = dataset.test_node2pos
                graph = dataset.test_holdout_subgraph
                candidate_positions = self.candidate_positions
                taxon2id = dataset.test_taxon2id
                id2taxon = dataset.test_id2taxon
            else:
                queries = dataset.valid_node_list
                node2pos = dataset.valid_node2pos
                graph = dataset.valid_holdout_subgraph
                candidate_positions = self.valid_candidate_positions
                taxon2id = dataset.valid_taxon2id
                id2taxon = dataset.valid_id2taxon
            taxon2id.keys()
            # id for test
            taxon2allemb_id = model.train_taxon2allemb_id  
            taxon2all_node_id = model.taxon2id  
            
            pseudo_leaf_id = taxon2allemb_id[dataset.pseudo_leaf_node]
            pseudo_root_id = taxon2allemb_id[dataset.pseudo_root_node]
            self.logger.info(f'number of candidate positions: {len(candidate_positions)}')

            
            all_ranks = []
            leaf_queries = []
            leaf_ranks, nonleaf_ranks = [], []

            eval_queries = queries[:q_val] if mode == 'validation' else queries
            q_id = torch.tensor([taxon2all_node_id[query] for i, query in enumerate(eval_queries)])

            # find leaf and nonleaf
            # redefine nonleaf
            num = 0 
            for i, query in enumerate(eval_queries):
                poses = node2pos[query]
                flag = True
                for pos in poses:
                    if pos[1] != dataset.pseudo_leaf_node:
                        flag = False
                        break
                if flag:
                    leaf_queries.append(query)
                    num+=1
            print(num)

            for i, query in tqdm(enumerate(eval_queries), desc=mode, total=len(eval_queries)):
                batched_energy_scores = []
                for edges in mit.sliced(candidate_positions, batch_size):
                    edges = list(edges)
                    ps, cs = zip(*edges)

                    with autocast():
                        p_idx = torch.tensor([taxon2allemb_id[n] for n in ps]).to(self.device)
                        c_idx = torch.tensor([taxon2allemb_id[n] for n in cs]).to(self.device)
                        pe, ce = map(lambda x: torch.logical_and(x != pseudo_leaf_id, x != pseudo_root_id),
                                             [p_idx, c_idx])
                        q_idx = torch.tensor([q_id[i] for j in range(p_idx.shape[0])])
                        pseudo_pct = torch.tensor([1 for j in range(p_idx.shape[0])])
                        
                        # [旁批] ★ Chain 的测试要跑 3 轮 forward，且后一轮依赖前一轮的硬答案：
                        #   第 1 轮 subtask='p' → 得到父的 yes/no
                        #   第 2 轮 subtask='c' + p_ans → 得到子的 yes/no
                        #   第 3 轮 subtask='a' + p_ans + c_ans → 得到最终分数
                        #   所以同一份候选下 Chain 的测试耗时是 Hidden 的 3 倍（训练也是 3 次 forward + 3 次 backward）。
                        _, p_ans = model.scorer(p_idx, q_idx, c_idx, pseudo_pct, pseudo_pct, pseudo_pct, pe=pe, ce=ce, subtask='p')
                        _, c_ans = model.scorer(p_idx, q_idx, c_idx, pseudo_pct, pseudo_pct, pseudo_pct, pe=pe, ce=ce, p_ans=p_ans, subtask='c')
                        
                        scores, _ = model.scorer(p_idx, q_idx, c_idx, pseudo_pct, pseudo_pct, pseudo_pct, pe=pe, ce=ce, p_ans=p_ans, c_ans=c_ans, subtask='a')
                        scores = scores.squeeze(1)
                    batched_energy_scores.append(scores)
                batched_energy_scores_cat = torch.cat(batched_energy_scores)

                # Total
                batched_energy_scores_cat, labels = rearrange(batched_energy_scores_cat, candidate_positions,
                                                              node2pos[query])
                ranks = self.pre_metric(batched_energy_scores_cat, labels)
                all_ranks.extend(ranks)

                if query in leaf_queries:
                    leaf_ranks.extend(ranks)
                else:
                    nonleaf_ranks.extend(ranks)

            print(all_ranks)
            total_metrics = [metric(all_ranks) for metric in self.metrics]
            leaf_metrics = [metric(leaf_ranks) for metric in self.metrics]
            nonleaf_metrics = [metric(nonleaf_ranks) for metric in self.metrics]

            print(f'leaf_metrics:{leaf_metrics}')
            print(f'nonleaf_metrics:{nonleaf_metrics}')
        return total_metrics, leaf_metrics, nonleaf_metrics