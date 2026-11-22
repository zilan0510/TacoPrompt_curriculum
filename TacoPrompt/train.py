# [导读] train.py —— TacoPrompt 的训练入口（只做「组装 + 启动」，不含算法）
#   六个步骤，全部围绕 config_files/<dataset>/config.test.{hidden,chain}.json：
#     ① ConfigParser 读 json，并把命令行 CustomArgs 覆盖进去
#     ② config.initialize('train_data_loader', ...) → UnifiedDataLoader
#        （内部还会顺手建 MAGDataset 读图 + RawDataset 建正例集/负采样池）
#     ③ config.initialize('arch', ...) → TacoPromptHidden / TacoPromptChain
#        ★ 第 3 个参数传的是 dataset 对象本身，不是词表 —— 模型要靠它查 description
#     ④ pre_metric = obtain_ranks(mode=1)，因为 loss=bce_loss（分数越大越好）
#     ⑤ config.initialize_trainer('arch', ...) → TrainerHidden / TrainerChain
#     ⑥ trainer.train()：跑 epochs 轮，结束后自动 load best_state 做一次全量测试
#   与 TaxoDiff 的区别：TacoPrompt 没有扩散模型，没有 pretrain / 采样两个阶段，一个脚本跑完。
#   读法建议：先看下面 main() 里这 6 行 initialize，再去 trainer/trainer.py 看 _train_epoch 和 _test。

import argparse
import collections
import os
import time
import warnings
from functools import partial
from math import ceil

import numpy as np
import torch
import transformers

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import trainer as trainer_arch
import wandb
from parse_config import ConfigParser
import random


def main(config):
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    warnings.filterwarnings('ignore')
    logger = config.get_logger('train')
    # [旁批] wandb 只用来记录 loss/指标曲线，不影响实验本身。下面几次 config.update 把超参写进 wandb。
    wandb.init(project='QEN', name=config['exp_name'])
    wandb.config.update({arg: config['trainer'][arg] for arg in
                         ['lp', 'lc', 'epochs', 'test_batch_size', 'early_stop', 'accumulation_iters']})
    wandb.config.update(config['arch']['args'])
    wandb.config.update(config['train_data_loader']['args'])
    wandb.config.update(config['optimizer']['args'])
    wandb.config.update({'optimizer': config['optimizer']['type']})

    # setup data_loader instances
    # [旁批] 一行代码背后发生的事（按顺序）：
    #   ① MAGDataset(raw=True)：读 .terms/.taxo/.desc，划分 train/valid/test，存成 .pickle.bin
    #   ② RawDataset：建 core_subgraph、node2pos、all_edges（负采样池）、valid/test_node2pos
    #   这些都是一次性的，第二次跑会读 .pickle.bin / subgraphs.pickle 加速。
    print('initializing dataloader...')
    train_data_loader = config.initialize('train_data_loader', module_data, config['data_path'])
    logger.info(train_data_loader)

    # build model architecture, then print to console
    # [旁批] ★ 关键：initizlize 的第 3 个位置参数是 train_data_loader.dataset（RawDataset 对象）。
    #   模型构造时要拿它做三件事：
    #     1) taxon2id / id2taxon → 把 Taxon 对象翻译成整数编号
    #     2) train_taxon2allemb_id → 训练子图里的「嵌入行号」（pit/qit/cit 就是这玩意）
    #     3) 所有节点的 description 列表 → 拼 prompt 的文本
    print('initializing model...')
    model = config.initialize('arch', module_arch, train_data_loader.dataset)
    # logger.info(model)

    # get function handles of loss and metrics
    print('initializing loss and metrics...')
    loss = getattr(module_loss, config['loss'])
    if config['loss'].startswith("FocalLoss"):
        loss = loss()
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # [旁批] 这行决定评测时「分数大的排前面」还是「分数小的排前面」：
    #   bce_loss / info_nce_loss 训练出的分数是「越大越像正例」→ mode=1（按 -score 降序排名）
    #   其它 loss（例如 margin loss）分数可能越小越好 → mode=0（按 score 升序）
    #   本项目 config 的 loss 是 bce_loss，所以走 mode=1。选错这个排序方向，全部指标会反过来。
    if config['loss'].startswith("info_nce") or config['loss'].startswith("bce_loss"):
        pre_metric = partial(module_metric.obtain_ranks, mode=1)  # info_nce_loss
    else:
        pre_metric = partial(module_metric.obtain_ranks, mode=0)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    print('initializing optimizer...')
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)
    if config['lr_scheduler']['library'] == 'torch':
        lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    elif 'warmup' in config['lr_scheduler']['type']:
        # [旁批] 这里假设「每个优化步吃掉 accumulation_iters 个 batch」，用它估算总步数给 LMScheduler 排余弦曲线。
        #   [旁批?] 但 trainer 里 self.accumulation_iters 被硬编码成 2，而 config 里写的是 8 →
        #   真实优化步数是这里的 4 倍，warmup 与余弦退火的曲线和实际步数对不上（不影响能否跑通，影响收敛形状）。
        total_steps = int(
            ceil(len(train_data_loader) / config['trainer']['accumulation_iters']) * config['trainer']['epochs'])
        params = {
            'optimizer': optimizer,
            'num_training_steps': total_steps,
            'num_warmup_steps': 0.1 * total_steps
        }
        lr_scheduler = getattr(transformers, config['lr_scheduler']['type'])(**params)
    else:
        lr_scheduler = None

    start = time.time()
    print('initializing trainer...')
    # [旁批] initialize_trainer 读的是 config['arch']['trainer']，不是 ['arch']['type']：
    #   type=TacoPromptHidden（模型类） / trainer=TrainerHidden（训练循环类），两个是分开配的。
    #   所以 chain 版本要同时改这两处：TacoPromptChain + TrainerChain。
    Trainer = config.initialize_trainer('arch', trainer_arch)
    trainer = Trainer(model, loss, metrics, pre_metric, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      lr_scheduler=lr_scheduler)
    print('training starts.')
    evaluations = trainer.train()
    end = time.time()
    logger.info(f"Finish training in {end - start} seconds")
    return evaluations


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--suffix', default="", type=str, help='suffix indicating this run (default: None)')
    args.add_argument('-n', '--n_trials', default=1, type=int, help='number of trials (default: 1)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--exp'], type=str, target=('exp_name',)),
        # Data loader (self-supervision generation)
        CustomArgs(['--train_data'], type=str, target=('train_data_loader', 'args', 'data_path')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('train_data_loader', 'args', 'batch_size')),
        CustomArgs(['--ns', '--negative_size'], type=int, target=('train_data_loader', 'args', 'negative_size')),
        CustomArgs(['--ef', '--expand_factor'], type=int, target=('train_data_loader', 'args', 'expand_factor')),
        CustomArgs(['--crt', '--cache_refresh_time'], type=int,
                   target=('train_data_loader', 'args', 'cache_refresh_time')),
        CustomArgs(['--nw', '--num_workers'], type=int, target=('train_data_loader', 'args', 'num_workers')),
        CustomArgs(['--sm', '--sampling_mode'], type=int, target=('train_data_loader', 'args', 'sampling_mode')),
        # Trainer & Optimizer
        CustomArgs(['--mode'], type=str, target=('mode',)),
        CustomArgs(['--loss'], type=str, target=('loss',)),
        CustomArgs(['--ep', '--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--es', '--early_stop'], type=int, target=('trainer', 'early_stop')),
        CustomArgs(['--tbs', '--test_batch_size'], type=int, target=('trainer', 'test_batch_size')),
        CustomArgs(['--v', '--verbose_level'], type=int, target=('trainer', 'verbosity')),
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--wd', '--weight_decay'], type=float, target=('optimizer', 'args', 'weight_decay')),
        CustomArgs(['--l1'], type=float, target=('trainer', 'l1')),
        CustomArgs(['--l2'], type=float, target=('trainer', 'l2')),
        CustomArgs(['--l3'], type=float, target=('trainer', 'l3')),
        # Model architecture
        CustomArgs(['--pm', '--propagation_method'], type=str, target=('arch', 'args', 'propagation_method')),
        CustomArgs(['--rm', '--readout_method'], type=str, target=('arch', 'args', 'readout_method')),
        CustomArgs(['--mm', '--matching_method'], type=str, target=('arch', 'args', 'matching_method')),
        CustomArgs(['--k'], type=int, target=('arch', 'args', 'k')),
        CustomArgs(['--in_dim'], type=int, target=('arch', 'args', 'in_dim')),
        CustomArgs(['--hidden_dim'], type=int, target=('arch', 'args', 'hidden_dim')),
        CustomArgs(['--out_dim'], type=int, target=('arch', 'args', 'out_dim')),
        CustomArgs(['--pos_dim'], type=int, target=('arch', 'args', 'pos_dim')),
        CustomArgs(['--num_heads'], type=int, target=('arch', 'args', 'heads', 0)),
        CustomArgs(['--feat_drop'], type=float, target=('arch', 'args', 'feat_drop')),
        CustomArgs(['--attn_drop'], type=float, target=('arch', 'args', 'attn_drop')),
        CustomArgs(['--hidden_drop'], type=float, target=('arch', 'args', 'hidden_drop')),
        CustomArgs(['--out_drop'], type=float, target=('arch', 'args', 'out_drop')),
        # TC added
        CustomArgs(['--lp'], type=float, target=('trainer', 'lp')),
        CustomArgs(['--lc'], type=float, target=('trainer', 'lc')),
        CustomArgs(['--lb'], type=float, target=('trainer', 'lb')),
        CustomArgs(['--lw'], type=float, target=('trainer', 'lw')),
        # 结构化父节点负采样（只开总开关；层列表/课程表写 config 的 hard_schedule）
        CustomArgs(['--hard_ratio'], type=float, target=('train_data_loader', 'args', 'hard_ratio')),
    ]
    config = ConfigParser(args, options)
    args = args.parse_args()
    n_trials = args.n_trials

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    setup_seed(47)

    if n_trials > 0:
        config.get_logger('train').info(f'number of trials: {n_trials}')
        metrics = config['metrics']
        save_file = config.log_dir / 'evaluations.txt'
        fin = open(save_file, 'w')
        # [旁批?] 这里少写了一个 '\n'，所以表头和第一组数值会粘在同一行。
        #   想直接用 pandas 读的话，先手动补换行，或者按 '\t' 切分后自己对齐列名。
        fin.write('\t'.join(metrics))

        evaluations = []
        exp_name = config['exp_name']

        for i in range(n_trials):
            config.config['exp_name'] = exp_name + '_main'
            config.set_save_dir(i + 1)
            res = main(config)
            evaluations.append(res)
            # [旁批] res 只包含 Total 那一套指标（BaseTrainer.test() 的返回值只有它）。
            #   论文表格里的 Leaf / Non-leaf 两栏不在这里，只在 log/info.log 的 test_leaf_* / test_nonleaf_* 里。
            fin.write('\t'.join([f'{i:.3f}' for i in res]))
            wandb.finish()

        evaluations = np.array(evaluations)
        means = evaluations.mean(axis=0)
        stds = evaluations.std(axis=0)
        final_output = '  '.join([f'& {i:.3f} +- {j:.3f}' for i, j in zip(means, stds)])
        fin.write(final_output)
        config.get_logger('train').info(final_output)
    else:
        main(config)
