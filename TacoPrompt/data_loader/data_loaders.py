# [导读] data_loaders.py —— 把 RawDataset 包成 PyTorch DataLoader（只有 62 行，很轻）
#   它只做三件事：
#     ① 建 MAGDataset（读图 + 切分）+ RawDataset（建正例集 / 负采样池）
#     ② collate_fn：把「几个 query 各自的 1 正 + N 负行」拼成一个大 batch 的 tensor
#     ③ 暴露 n_samples 给 trainer 打日志
#   ★ 17 个字段的顺序在这里锁定，与 dataset.__getitem__ 的 return、model.forward 的入参三者必须保持一致。
#   ★ 与 TaxoDiff 的差别：TacoPrompt 这里仍然把 5 列文本（q/p/c/b/w）送进 tokenizer；
#     但 model.forward(qs, ps, ...) 根本没有读它们 —— 这几行 tokenize 是纯浪费。
#   （TaxoDiff 那边已把它改成空 dict，结果完全一样。）

import torch
from torch.utils.data import DataLoader
from functools import partial

from .dataset import MAGDataset, RawDataset
from itertools import chain


class UnifiedDataLoader(DataLoader):
    def __init__(self, data_path, taxonomy_name, sampling_mode=1, batch_size=10, negative_size=20, max_pos_size=100,
                 expand_factor=50, shuffle=True, num_workers=8, cache_refresh_time=64,
                 tokenizer='/codes/b/offline_models/bert_model', test_topk=-1,
                 hard_ratio=0.0, hard_schedule=None):
        self.sampling_mode = sampling_mode
        self.batch_size = batch_size
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.shuffle = shuffle
        self.cache_refresh_time = cache_refresh_time

        # raw_graph_dataset = MAGDataset(name=taxonomy_name, path=data_path, raw=False, existing_partition=True)
        # [旁批] ★ 这两个布尔值决定了「数据集怎么来」：
        #   raw=True           → 每次都从 .terms/.taxo/.desc 重新读图（有 .pickle.bin 也会重读，然后覆写）
        #   existing_partition=False → 不看 .terms.train/.validation/.test 文件，而是用
        #                              random.seed(47) + shuffle 重新抽 10% / 10%（上限各 1000）作验证/测试
        #   想用自己切好的划分，就把 existing_partition 改成 True 并准备三个 .terms.* 文件。
        raw_graph_dataset = MAGDataset(name=taxonomy_name, path=data_path, raw=True, existing_partition=False)
        self.dataset = RawDataset(raw_graph_dataset, sampling_mode=sampling_mode,
                                  negative_size=negative_size, max_pos_size=max_pos_size, expand_factor=expand_factor,
                                  cache_refresh_time=cache_refresh_time, test_topk=test_topk, tokenizer=tokenizer,
                                  hard_ratio=hard_ratio, hard_schedule=hard_schedule)
        self.tokenizer = self.dataset.tokenizer
        self.num_workers = num_workers
        # [旁批] collate_fn 是自定义的（见下），所以 batch 的形状是「1 个 query 的 1+N 行」乘 batch_size。
        #   num_workers>0 时每个 worker 会各自复制一份 dataset（含 4.3 万候选池），非常吃内存 → 建议设 0。
        super(UnifiedDataLoader, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                collate_fn=self.collate_fn, num_workers=self.num_workers,
                                                pin_memory=True)
        self.n_samples = len(self.dataset)  # total number of samples that will be emitted by this data loader

    def collate_fn(self, samples):
        # input: q, p, c, b, w, (p,c) tag, p tag, c tag, b tag, w tag, p exists, c exists, b exists, w exists
        # qs, ps, cs, bs, ws, pct, pt, ct, bt, wt, pe, ce, be, we = map(list, zip(*chain(*samples)))
        #
        # # the _t suffix stands for tensor
        # qs_t, ps_t, cs_t, bs_t, ws_t = map(
        #     partial(self.tokenizer, return_tensors='pt', padding=True, truncation=True, max_length=64),
        #     [qs, ps, cs, bs, ws])
        # pct_t, pt_t, ct_t, bt_t, wt_t, pe_t, ce_t, be_t, we_t = map(torch.tensor, [pct, pt, ct, bt, wt, pe, ce, be, we])
        # return qs_t, ps_t, cs_t, bs_t, ws_t, pct_t, pt_t, ct_t, bt_t, wt_t, pe_t, ce_t, be_t, we_t

        # 加入了p, q, c的train idx
        # [旁批] samples 是 List[Tuple[16 行]]（一个 batch 有 batch_size 个 query，每 query 16 行）。
        #   chain(*samples) 先把「query 的列表」展开成一条长长的行列表，zip(*...) 再按列转置，
        #   得到 17 个长度 = batch_size*16 的列表。所以后面每个 tensor 的第 0 维都是「总行数」。
        qs, ps, cs, bs, ws, pct, pt, ct, bt, wt, pe, ce, be, we, pi, qi, ci = map(list, zip(*chain(*samples)))

        # the _t suffix stands for tensor
        # [旁批?] 这 5 列文本会被 tokenize，但 model.forward 完全不用（只读 pit/qit/cit 和三个标签）→
        #   可以整段删掉换成 qs_t, ps_t, cs_t, bs_t, ws_t = {}, {}, {}, {}, {}
        #   因为 trainer 里 `{k: v.to(device) for k, v in qs.items()}` 对空 dict 也是空 dict，forward 会忽略。
        #   每个 query 16 行、每行 5 段描述，删掉能省相当一部分 CPU 时间（TaxoDiff 已这么改）。
        qs_t, ps_t, cs_t, bs_t, ws_t = map(
            partial(self.tokenizer, return_tensors='pt', padding=True, truncation=True, max_length=64),
            [qs, ps, cs, bs, ws])
        pct_t, pt_t, ct_t, bt_t, wt_t, pe_t, ce_t, be_t, we_t, pit, qit, cit = \
            map(torch.tensor, [pct, pt, ct, bt, wt, pe, ce, be, we, pi, qi, ci])
        return qs_t, ps_t, cs_t, bs_t, ws_t, pct_t, pt_t, ct_t, bt_t, wt_t, pe_t, ce_t, be_t, we_t, pit, qit, cit

    def __str__(self):
        return "\n\t".join([
            f"sampling_mode: {self.sampling_mode}",
            f"batch_size: {self.batch_size}",
            f"negative_size: {self.negative_size}",
            f"expand_factor: {self.expand_factor}",
            f"cache_refresh_time: {self.cache_refresh_time}",
        ])
