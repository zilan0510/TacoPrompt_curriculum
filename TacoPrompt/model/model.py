# [导读] model.py —— 所有模型定义在这里（570 行），但只关心最后两个类
#   前五个类都是 「baseline」，跑 TacoPrompt 时完全用不到，可以整段跳过：
#     MatchModel / ExpanMatchModel  → TaxoExpan / TMN / Arborist 那类「结构嵌入 + 匹配头」的老方法
#     QENModel                      → 双塔 PolyEncoder + 辅助分类头
#     TEMPCompletion                → 用 BERT [CLS] 向量 + MLP 打分（「+ [SEP] 描述」 的拼接式）
#   真正要读的两个：
#     TacoPromptHidden → 主模型（config.test.hidden.json 用这个）
#        一个 BERT + 一个 prompt: "All: [M] Parent: [M] Chlid: [M]. <Parent>: {p} <Child>: {c} [SEP] <Query>: {q}"
#        三个 [MASK] 各自对应一个二分类子任务（整体 / 父 / 子）
#     TacoPromptChain  → chain 变体（config.test.chain.json）
#        分三段跑：先问 Parent，把答案写回 prompt；再问 Child；最后问 All
#   两者共享的套路：score = logit(yes) - logit(no)（不是概率，是 log-odds）
# 与 TaxoDiff 的关系：TaxoDiff 的 model.py 就是在 TacoPromptHidden 基础上多加了扩散模型和 e_q 表征。
#   所以把 TacoPromptHidden 读透，再看 TaxoDiff 的 model.py 就是「原版 + 一个分支」。

import time

import torch
import torch.nn as nn

from transformers import AutoModel, DistilBertModel, BertForMaskedLM, DistilBertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM, ElectraForMaskedLM, AutoTokenizer
from base import BaseModel
from .model_zoo import *
import scipy.sparse as sp
from gensim.models import KeyedVectors
import numpy as np


class AbstractPathModel(nn.Module):
    def __init__(self):
        super(AbstractPathModel, self).__init__()

    def init(self, **options):
        self.hidden_size = options['out_dim']
        in_dim = options['in_dim']
        out_dim = options['out_dim']
        self.p_lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_size, batch_first=True)
        self.c_lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_size, batch_first=True)
        self.p_control = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU())
        self.c_control = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU())

    def init_hidden(self, batch_size, device):
        hidden = (torch.randn(1, batch_size, self.hidden_size).to(device),
                  torch.randn(1, batch_size, self.hidden_size).to(device))
        return hidden

    def encode_parent_path(self, p, lens):
        batch_size, seq_len = p.size()
        hidden = self.init_hidden(batch_size, self.device)
        p = self.embedding(p)
        c = self.p_control(p[:, 0, :]).view(batch_size, 1, -1)
        X = torch.nn.utils.rnn.pack_padded_sequence(p, lens, batch_first=True, enforce_sorted=False)
        X, hidden = self.p_lstm(X, hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = (c * X).max(dim=1)[0]
        return X

    def encode_child_path(self, p, lens):
        batch_size, seq_len = p.size()
        hidden = self.init_hidden(batch_size, self.device)
        p = self.embedding(p)
        c = self.c_control(p[:, 0, :]).view(batch_size, 1, -1)
        X = torch.nn.utils.rnn.pack_padded_sequence(p, lens, batch_first=True, enforce_sorted=False)
        X, hidden = self.c_lstm(X, hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = (c * X).max(dim=1)[0]
        return X

    def forward_path_encoders(self, pu, pv, lens):
        pu = pu.to(self.device)
        pv = pv.to(self.device)
        lens = lens.to(self.device)
        hpu = self.encode_parent_path(pu, lens[:, 0])
        hpv = self.encode_child_path(pv, lens[:, 1])
        return hpu, hpv


class AbstractGraphModel(nn.Module):
    def __init__(self):
        super(AbstractGraphModel, self).__init__()

    def init(self, **options):
        propagation_method = options['propagation_method']
        readout_method = options['readout_method']
        options = options
        if propagation_method == "GCN":
            self.parent_graph_propagate = GCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                activation=F.leaky_relu, in_dropout=options["feat_drop"], hidden_dropout=options["hidden_drop"],
                output_dropout=options["out_drop"])
            self.child_graph_propagate = GCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                activation=F.leaky_relu, in_dropout=options["feat_drop"], hidden_dropout=options["hidden_drop"],
                output_dropout=options["out_drop"])
        elif propagation_method == "PGCN":
            self.parent_graph_propagate = PGCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], activation=F.leaky_relu, in_dropout=options["feat_drop"],
                hidden_dropout=options["hidden_drop"], output_dropout=options["out_drop"])
            self.child_graph_propagate = PGCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], activation=F.leaky_relu, in_dropout=options["feat_drop"],
                hidden_dropout=options["hidden_drop"], output_dropout=options["out_drop"])
        elif propagation_method == "GAT":
            self.parent_graph_propagate = GAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                heads=options["heads"], activation=F.leaky_relu, feat_drop=options["feat_drop"],
                attn_drop=options["attn_drop"])
            self.child_graph_propagate = GAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                heads=options["heads"], activation=F.leaky_relu, feat_drop=options["feat_drop"],
                attn_drop=options["attn_drop"])
        elif propagation_method == "PGAT":
            self.parent_graph_propagate = PGAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], heads=options["heads"], activation=F.leaky_relu,
                feat_drop=options["feat_drop"], attn_drop=options["attn_drop"])
            self.child_graph_propagate = PGAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], heads=options["heads"], activation=F.leaky_relu,
                feat_drop=options["feat_drop"], attn_drop=options["attn_drop"])
        else:
            assert f"Unacceptable Graph Propagation Method: {self.propagation_method}"

        if readout_method == "MR":
            self.p_readout = MeanReadout()
            self.c_readout = MeanReadout()
        elif readout_method == "WMR":
            self.p_readout = WeightedMeanReadout()
            self.c_readout = WeightedMeanReadout()
        else:
            assert f"Unacceptable Readout Method: {self.readout_method}"

    def encode_parent_graph(self, g):
        h = self.embedding(g.ndata['_id'].to(self.device))
        pos = g.ndata['pos'].to(self.device)
        g.ndata['h'] = self.parent_graph_propagate(g, h)
        h = self.p_readout(g, pos)
        return h

    def encode_child_graph(self, g):
        h = self.embedding(g.ndata['_id'].to(self.device))
        pos = g.ndata['pos'].to(self.device)
        g.ndata['h'] = self.child_graph_propagate(g, h)
        h = self.c_readout(g, pos)
        return h

    def forward_graph_encoders(self, gu, gv):
        hgu = self.encode_parent_graph(gu)
        hgv = self.encode_child_graph(gv)
        return hgu, hgv


class MatchModel(BaseModel, AbstractPathModel, AbstractGraphModel):
    # [旁批] baseline：TaxoExpan / TMN / Arborist 系列。它直接拼「节点 + 图邻居 + 路径」嵌入，
    #   再用一个匹配头（MLP/NTN/...）打分。TacoPrompt 不用，整段可跳过。
    def __init__(self, mode, **options):
        super(MatchModel, self).__init__()
        self.options = options
        self.mode = mode

        l_dim = 0
        if 'r' in self.mode:
            l_dim += options["in_dim"]
        if 'g' in self.mode:
            l_dim += options["out_dim"]
            AbstractGraphModel.init(self, **options)
        if 'p' in self.mode:
            l_dim += options["out_dim"]
            AbstractPathModel.init(self, **options)
        self.l_dim = l_dim
        self.r_dim = options["in_dim"]

        if options['matching_method'] == "MLP":
            self.match = MLP(self.l_dim, self.r_dim, 100, options["k"])
        elif options['matching_method'] == "SLP":
            self.match = SLP(self.l_dim, self.r_dim, 100)
        elif options['matching_method'] == "DST":
            self.match = DST(self.l_dim, self.r_dim)
        elif options['matching_method'] == "LBM":
            self.match = LBM(self.l_dim, self.r_dim)
        elif options['matching_method'] == "BIM":
            self.match = BIM(self.l_dim, self.r_dim)
        elif options['matching_method'] == "Arborist":
            self.match = Arborist(self.l_dim, self.r_dim, options["k"])
        elif options['matching_method'] == "NTN":
            self.match = NTN(self.l_dim, self.r_dim, options["k"])
        elif options['matching_method'] == "CNTN":
            self.match = CNTN(self.l_dim, self.r_dim, options["k"])
        elif options['matching_method'] == "TMN":
            self.match = TMN(self.l_dim, self.r_dim, options["k"])
        else:
            assert f"Unacceptable Matching Method: {options['matching_method']}"

    def forward_encoders(self, u=None, v=None, gu=None, gv=None, pu=None, pv=None, lens=None):
        ur, vr = [], []
        if 'r' in self.mode:
            hu = self.embedding(u.to(self.device))
            hv = self.embedding(v.to(self.device))
            ur.append(hu)
            vr.append(hv)
        if 'g' in self.mode:
            gu = dgl.batch(gu)
            gv = dgl.batch(gv)
            hgu, hgv = self.forward_graph_encoders(gu, gv)
            ur.append(hgu)
            vr.append(hgv)
        if 'p' in self.mode:
            hpu, hpv = self.forward_path_encoders(pu, pv, lens)
            ur.append(hpu)
            vr.append(hpv)
        ur = torch.cat(ur, -1)
        vr = torch.cat(vr, -1)
        return ur, vr

    def forward(self, q, *inputs):
        qf = self.embedding(q.to(self.device))
        ur, vr = self.forward_encoders(*inputs)
        scores = self.match(ur, vr, qf)
        return scores


class ExpanMatchModel(BaseModel, AbstractPathModel, AbstractGraphModel):
    # [旁批] baseline：TaxoExpan 的原始形式（只编码父侧，一次只评一个候选）。TacoPrompt 不用，可跳过。
    def __init__(self, mode, **options):
        super(ExpanMatchModel, self).__init__()
        self.options = options
        self.mode = mode

        l_dim = 0
        if 'r' in self.mode:
            l_dim += options["in_dim"]
        if 'g' in self.mode:
            l_dim += options["out_dim"]
            AbstractGraphModel.init(self, **options)
        if 'p' in self.mode:
            l_dim += options["out_dim"]
            AbstractPathModel.init(self, **options)
        self.l_dim = l_dim
        self.r_dim = options["in_dim"]

        if options['matching_method'] == "NTN":
            self.match = RawNTN(self.l_dim, self.r_dim, options["k"])
        if options['matching_method'] == "BIM":
            self.match = RawBIM(self.l_dim, self.r_dim)
        if options['matching_method'] == "MLP":
            self.match = RawMLP(self.l_dim, self.r_dim, 100, options["k"])
        elif options['matching_method'] == "ARB":
            self.match = RawArborist(self.l_dim, self.r_dim, options["k"])
        else:
            assert f"Unacceptable Matching Method: {options['matching_method']}"

    def forward_encoders(self, u=None, gu=None, pu=None, lens=None):
        ur = []
        if 'r' in self.mode:
            hu = self.embedding(u.to(self.device))
            ur.append(hu)
        if 'g' in self.mode:
            gu = dgl.batch(gu)
            hgu = self.encode_parent_graph(gu)
            ur.append(hgu)
        if 'p' in self.mode:
            pu = pu.to(self.device)
            lens = lens.to(self.device)
            hpu = self.encode_parent_path(pu, lens)
            ur.append(hpu)
        ur = torch.cat(ur, -1)
        return ur

    def forward(self, q, us, graphs, paths, lens):
        qf = self.embedding(q.to(self.device))
        ur = self.forward_encoders(us, graphs, paths, lens)
        scores = self.match(ur, qf)
        return scores


class QENModel(BaseModel):
    # [旁批] baseline：双塔（PolyEncoder）编码 query / parent / child / best / worst，
    #   再用 PolyScorer + 两个辅助 head 打分。TacoPrompt 不用，可跳过。
    def __init__(self, **options):
        super(BaseModel, self).__init__()
        self.options = options
        self.dropout = nn.Dropout(p=options['dropout'])
        self.poly_encoder = PolyEncoder(**options)
        self.poly_scorer = PolyScorer(**options)
        self.parent_scorer = nn.Sequential(
            nn.Linear(options['code_dim'], options['scorer_hidden_dim']),
            nn.ReLU(),
            self.dropout,
            nn.Linear(options['scorer_hidden_dim'], 1)
        )
        self.sibling_scorer = nn.Sequential(
            nn.Linear(options['code_dim'], options['scorer_hidden_dim']),
            nn.ReLU(),
            self.dropout,
            nn.Linear(options['scorer_hidden_dim'], 1)
        )
        # self.parent_scorer = nn.Linear(options['code_dim'], 1)
        # self.sibling_scorer = nn.Linear(options['code_dim'], 1)

    def forward(self, qs, ps, cs, bs, ws, pe, ce, be, we, debug=False):
        q_code, p_code, c_code, b_code, w_code = map(self.poly_encoder, [qs, ps, cs, bs, ws])
        # p_code, c_code, b_code, w_code = map(self.poly_encoder, [ps, cs, bs, ws])
        score, pq_parental, qc_parental, qb_sibling, qw_sibling = self.poly_scorer(q_code, p_code, c_code, b_code,
                                                                                   w_code, pe, ce, be, we)

        # auxiliary scores
        p_parent_score = self.parent_scorer(pq_parental)
        c_child_score = self.parent_scorer(qc_parental)
        b_sibling_score = self.sibling_scorer(qb_sibling)
        w_sibling_score = self.sibling_scorer(qw_sibling)

        return score, p_parent_score, c_child_score, b_sibling_score, w_sibling_score


class TEMPCompletion(BaseModel):
    # [旁批] baseline：TEMP。把「从 root 到 parent 的路径 + q + c」拼成一句话，取 BERT 的 [CLS] 向量，
    #   过一层 MLP 出 1 个分数。它需要的 parent2paths 就是 dataset.py 里预先并缓存的那个。
    #   与 TacoPrompt 的核心差别：TEMP 用 [CLS] 做回归，TacoPrompt 用 [MASK] 位置的 yes/no logits。
    def __init__(self, raw_dataset, **options):
        super(BaseModel, self).__init__()
        self.raw_dataset = raw_dataset
        self.n_anchors = len(self.raw_dataset.train_allemb_id2taxon)
        self.taxon2id = self.raw_dataset.taxon2id
        self.id2taxon = self.raw_dataset.id2taxon
        self.train_allemb_id2taxon = self.raw_dataset.train_allemb_id2taxon
        self.train_taxon2allemb_id = self.raw_dataset.train_taxon2allemb_id
        self.pseudo_leaf_allemb_id = self.taxon2id[raw_dataset.pseudo_leaf_node]
        self.tokenizer = AutoTokenizer.from_pretrained(options['plm'])
        self.plm = AutoModel.from_pretrained(options['plm'])
        self.mlp = nn.Sequential(nn.Linear(768, 256), nn.Dropout(0.2), nn.Linear(256, 1))

        self.options = options
        self.desc = [node.description for node_id, node in self.id2taxon.items()]
        self.query_name = [node.norm_name for node_id, node in self.id2taxon.items()]
        self.parent2path = self.raw_dataset.parent2paths

        self._init_idx_map()

    def _init_idx_map(self):
        self.all_embed_id2all_node_id = {}
        for id, taxon in self.train_allemb_id2taxon.items():
            self.all_embed_id2all_node_id[id] = self.taxon2id[taxon]
        self.all_node_id_2all_embed_id = {v: k for k, v in self.all_embed_id2all_node_id.items()}

    def convert_id(self, id):
        return self.all_embed_id2all_node_id[id]

    def lm_forward(self, text):
        repr = self.plm(**text)
        repr = repr['last_hidden_state'][:, 0, :].squeeze(1)
        return repr

    def get_text_emb(self, pit, qid, cit, batch=32, test=False):
        if test:
            batch = 4096
        text_prompt = "{} [SEP] {}"
        text = [text_prompt.format(self.parent2path[self.train_allemb_id2taxon[pit[i].item()]] + ',{},{}'.format(
            self.query_name[qid[i]],self.train_allemb_id2taxon[cit[i].item()].norm_name), 
            self.desc[qid[i]]) for i in range(pit.shape[0])]

        text_emb = torch.zeros(pit.shape[0], 768)
        total_times = math.ceil(pit.shape[0] / batch)
        for i in range(total_times - 1):
            start = i * batch
            text_ = [text[start + j] for j in range(batch)]
            text_pt = self.tokenizer(text_, return_tensors='pt', padding=True, truncation=True, max_length=256)
            text_pt = {k: v.to("cuda") for k, v in text_pt.items()}
            text_emb[start: start + batch] = self.lm_forward(text_pt)
        start = (total_times - 1) * batch
        end_step = pit.shape[0] - start
        text_ = [text[start + j] for j in range(end_step)]
        text_pt = self.tokenizer(text_, return_tensors='pt', padding=True, truncation=True, max_length=256)
        text_pt = {k: v.to("cuda") for k, v in text_pt.items()}
        text_emb[start: start + end_step] = self.lm_forward(text_pt)
        return text_emb.to('cuda')

    def scorer(self, pit, qid, cit, test=False):
        text_emb = self.get_text_emb(pit, qid, cit, test=test)
        score = self.mlp(text_emb)
        return score

    def forward(self, qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit):
        qid = [self.convert_id(qit[i].item()) for i in range(qit.shape[0])]
        return self.scorer(pit, qid, cit)


class TacoPromptHidden(BaseModel):
    # [旁批] ★ 主模型。
    #   名字里的 Hidden 指的是：Parent / Child 两个子任务用「可学习的隐藏表示」参与建模
    #   （prompt 里 Parent/Child 两个位置放的是 [MASK]，而不是已经算好的 yes/no）。
    #   而 TacoPromptChain 是先把 Parent 算成硬结果（yes/no）再写进 prompt，再算 Child，最后算 All。
    def __init__(self, raw_dataset, **options):
        super(BaseModel, self).__init__()
        self.raw_dataset = raw_dataset
        self.n_anchors = len(self.raw_dataset.train_allemb_id2taxon)
        self.taxon2id = self.raw_dataset.taxon2id
        self.id2taxon = self.raw_dataset.id2taxon
        self.train_allemb_id2taxon = self.raw_dataset.train_allemb_id2taxon
        self.train_taxon2allemb_id = self.raw_dataset.train_taxon2allemb_id
        self.pseudo_leaf_allemb_id = self.taxon2id[raw_dataset.pseudo_leaf_node]

        # [旁批] 整个模型 = 一个 BERT + 一个 MLM 头，没有别的参数。全部知识都在 prompt 的措辞里。
        self.model = BertForMaskedLM.from_pretrained(options['plm'])
        self.tokenizer = AutoTokenizer.from_pretrained(options['plm'])
        # [旁批] yes / no 在 bert-base-uncased 里都是单 token（yes=2748, no=2053），
        #   所以可以直接拿两个 logits 相减当二分类分数（log-odds），不需要再套 sigmoid。
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('yes'))[0]
        self.no_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('no'))[0]
        self.mask_token_id = self.tokenizer.mask_token_id
        self.max_seq_length = 256

        self.options = options
        # [旁批] desc 是列表，下标是 node_id。读它之前一定要先 convert_id(嵌入行号) 换成 node_id，
        #   否则会取到别的概念的描述 —— 这是排查「prompt 拼错概念」的第一现场。
        self.desc = [node.description for node_id, node in self.id2taxon.items()]
        self.device = "cuda"

        self._init_idx_map()

    def _init_idx_map(self):
        # all_emb_id <-> all_node_id
        # [旁批] 两套编号的翻译表（dataset 里那条批注说的「第三套编号」就是指 allemb_id）：
        #   正向：嵌入行号 → 全图 node_id（convert_id 用它）
        #   倒排：node_id → 嵌入行号（测试时把真值 Taxon 转成索引会用到）
        self.all_embed_id2all_node_id = {}
        for id, taxon in self.train_allemb_id2taxon.items():
            self.all_embed_id2all_node_id[id] = self.taxon2id[taxon]
        self.all_node_id_2all_embed_id = {v: k for k, v in self.all_embed_id2all_node_id.items()}

    def convert_id(self, id):
        return self.all_embed_id2all_node_id[id]

    def get_lm_inputs(self, pit, qid, cit, label, pl, cl, tc=False):
        # [旁批] ★ 这个函数就是论文 §4.2 的 "Hidden" prompt 模板，拼两条句子：
        #   推理用：All: [MASK] Parent: [MASK] Chlid: [MASK]. <Parent>: {p_desc} <Child>: {c_desc} [SEP] <Query>: {q_desc}
        #   标签用：All: yes/no Parent: yes/no Child: yes/no. <Parent>: ... <Child>: ... [SEP] <Query>: ...
        #   两份句子逐 token 对齐（都是 max_length=256, padding='max_length'），
        #   然后用 input_ids == mask_token_id 把「三个 [MASK] 位置的标签」挑出来作为 label_ids。
        #   这就是 MLM 训 TacoPrompt 的全部秘密：不新加分类头，而是把分类问题变成「填空」。
        max_seq_length = self.max_seq_length
        tokenizer = self.tokenizer
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token_id = tokenizer.pad_token_id
        mask_token = tokenizer.mask_token
        mask_token_id = tokenizer.mask_token_id

        sentences_prompt = []
        sentences_ground_truth = []

        for i in range(label.shape[0]):
            # [旁批] ★★ 三个 [MASK] 对应的就是三个子任务：
            #   第 1 个（All）    → 这个位置整体对不对（pct）
            #   第 2 个（Parent） → 父对不对（pt）
            #   第 3 个（Child）  → 子对不对（ct）
            #   [旁批?] 注意上面用的是 "Chlid"（拼错），下面 ground truth 用的是 "Child"。
            #     后果：Child → 1 个 token；Chlid → 2 个 token（ch + ##lid），模板长 1 个 token。
            #     不影响打分（靠 mask 值定位而不是文本匹配），但改代码时别把它 "修正" 成 Child，
            #     否则与已发布实验不可比。
            a_prompt = "All: {} Parent: {} Chlid: {}. <Parent>: {} <Child>: {}".format(
                mask_token, mask_token, mask_token, self.desc[self.convert_id(pit[i].item())], self.desc[self.convert_id(cit[i].item())])
            b_prompt = "<Query>: {}".format(self.desc[qid[i]])
            sentence_prompt = a_prompt + sep_token + b_prompt
            sentences_prompt.append(sentence_prompt)

            label_token = 'yes' if label[i].item() else 'no'
            pl_token = 'yes' if pl[i].item() else 'no'
            cl_token = 'yes' if cl[i].item() else 'no'

            a_prompt = "All: {} Parent: {} Child: {}. <Parent>: {} <Child>: {}".format(
                label_token, pl_token, cl_token, self.desc[self.convert_id(pit[i].item())], self.desc[self.convert_id(cit[i].item())])
            sentence_prompt_gt = a_prompt + sep_token + b_prompt
            sentences_ground_truth.append(sentence_prompt_gt)

        sentences_prompt_pt = tokenizer(sentences_prompt, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        sentences_prompt_gt_pt = tokenizer(sentences_ground_truth, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = sentences_prompt_pt.input_ids
        attention_mask = sentences_prompt_pt.attention_mask
        labels = sentences_prompt_gt_pt.input_ids
        label_ids = torch.where(input_ids == mask_token_id, labels, -100)

        return input_ids, label_ids, attention_mask

    def scorer(self, pit, qid, cit, pct, pl, cl):
        # [旁批] ★ 打分函数。注意它其实算了两份输入：
        #   第一份 sentences_prompt（带 [MASK]）送去 forward → 得到 logits
        #   第二份 sentences_ground_truth（带 yes/no）只用来生成 label_ids → 只在训练时用
        #   测试时虽然也白算了一份 gt 句子的 tokenize，但不影响结果（可优化点）。
        batch_input_ids, batch_labels, batch_attention_mask = self.get_lm_inputs(pit, qid, cit, pct, pl, cl)
        batch_lm_inputs = {'input_ids': batch_input_ids,
                           'attention_mask': batch_attention_mask}
        batch_lm_inputs = {k: v.to(self.device) for k, v in batch_lm_inputs.items()}
        # [旁批?] ★ 这里是全流程最大的算力浪费：BertForMaskedLM.forward 会对全部 256 个位置
        #   跑 MLM 头（得到 (B,256,30522) 的 logits），而我们只偷 3 个 [MASK] 位置。
        #   浪费的不仅是 ~22% 的 FLOPs，更是显存：batch=775 时这个中间张量 fp16 就有 12 GB。
        #   优化办法：先跑 model.model.bert(...) 拿 hidden，再只用 3 行跑 model.model.cls(rows)。
        outputs = self.model(**batch_lm_inputs)
        score = outputs[0][batch_lm_inputs['input_ids'] == self.mask_token_id]   # (B*3, V)
        # [旁批] 分数 = logit(yes) - logit(no)，即 log-odds。
        #   因为训练用 BCEWithLogits，这个量与 loss 的输入完全同量纲，不需要再 sigmoid。
        pred_scores = score[:, self.yes_token_id] - score[:, self.no_token_id]
        # [旁批] view(B, 3)：第 0 列是 All、第 1 列是 Parent、第 2 列是 Child。
        #   测试时只取 score[:, 0]（All 位）；训练时三列都要（对应 L_a / L_p / L_c）。
        pred_scores = pred_scores.view(pit.shape[0], -1)
        return pred_scores

    def forward(self, qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct):
        # [旁批] forward 其实只用了 cit / pct / pt / ct 四个参数，其余全部忽略。
        #   qs/ps/cs/bs/ws（五列 tokenize 过的文本）在这里被完全丢弃 —— 它们的存在只是为了
        #   兼容早期代码。这也是为什么 collate_fn 里那五行 tokenizer 可以安全地改成空 dict。
        #   [旁批] qit 是「嵌入行号」，要先 convert_id 换成 node_id，因为 scorer 里用 self.desc[qid[i]] 索引。
        qid = [self.convert_id(qit[i].item()) for i in range(qit.shape[0])]
        return self.scorer(pit, qid, cit, pct, pt, ct)
    

class TacoPromptChain(BaseModel):
    # [旁批] ★ chain 变体：把一个多答案的 prompt 拆成三段串行问，后一模用前一模的「硬结果」。
    #   第 1 模 subtask='p'："Parent: [MASK]. <Parent>: {p}" [SEP] <Query>: {q}         → 父对不对
    #   第 2 模 subtask='c'："Child: [MASK] Parent: {父的答案}. <Child>: {c}" ...       → 子对不对
    #   第 3 模 subtask='a'："All: [MASK] Parent: {父答案} Child: {子答案}. ..."         → 整体对不对
    #   关键区别：Hidden 是把三个子问题写在同一句里同时算（soft、可学习），
    #   Chain 是把上一段的 yes/no 当文本拼进下一段（hard、离散）。所以 Chain 每步要 forward 3 次，
    #   _test 里就是一个 query 跑 3 轮 scorer（见 trainer/trainer.py 的 TrainerChain._test）。
    def __init__(self, raw_dataset, **options):
        super(BaseModel, self).__init__()
        self.raw_dataset = raw_dataset
        self.n_anchors = len(self.raw_dataset.train_allemb_id2taxon)
        self.taxon2id = self.raw_dataset.taxon2id
        self.id2taxon = self.raw_dataset.id2taxon
        self.train_allemb_id2taxon = self.raw_dataset.train_allemb_id2taxon
        self.train_taxon2allemb_id = self.raw_dataset.train_taxon2allemb_id
        self.pseudo_leaf_allemb_id = self.taxon2id[raw_dataset.pseudo_leaf_node]

        self.model = BertForMaskedLM.from_pretrained(options['plm'])
        self.tokenizer = AutoTokenizer.from_pretrained(options['plm'])
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('yes'))[0]
        self.no_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('no'))[0]
        self.mask_token_id = self.tokenizer.mask_token_id
        self.max_seq_length = 256

        self.options = options
        self.desc = [node.description for node_id, node in self.id2taxon.items()]
        self.device = "cuda"

        self._init_idx_map()

    def _init_idx_map(self):
        # all_emb_id <-> all_node_id
        self.all_embed_id2all_node_id = {}
        for id, taxon in self.train_allemb_id2taxon.items():
            self.all_embed_id2all_node_id[id] = self.taxon2id[taxon]
        self.all_node_id_2all_embed_id = {v: k for k, v in self.all_embed_id2all_node_id.items()}

    def convert_id(self, id):
        return self.all_embed_id2all_node_id[id]

    def get_lm_inputs(self, pit, qid, cit, label, pl, cl, pe, ce, p_ans=None, c_ans=None, subtask='a', tc=False):
        # [旁批] 三种 subtask 的 prompt 模板完全不同，这是 chain 的核心：
        #   'p' → 只问父，没有子相关字段
        #   'c' → 问子，并把【父的答案】以文本形式拼在前面（信息从上一模传来）
        #   'a' → 问整体，父子两个答案都作为文本拼进来
        #   同一个函数先拼「推理用（带 [MASK]）」再拼「标签用（带 yes/no）」，两段逻辑几乎重复。
        max_seq_length = self.max_seq_length
        tokenizer = self.tokenizer
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token_id = tokenizer.pad_token_id
        mask_token = tokenizer.mask_token
        mask_token_id = tokenizer.mask_token_id

        sentences_prompt = []
        sentences_ground_truth = []

        for i in range(label.shape[0]):
            if subtask == 'a':
                p_answer = 'yes' if p_ans[i] else 'no'
                c_answer = 'yes' if c_ans[i] else 'no'

                a_prompt = "All: {} Parent: {}  Child: {}. <Parent>: {} <Child>: {}".format(
                    mask_token, p_answer, c_answer, self.desc[self.convert_id(pit[i].item())], self.desc[self.convert_id(cit[i].item())])
            
            elif subtask == 'p':
                a_prompt = "Parent: {}. <Parent>: {}".format(
                    mask_token, self.desc[self.convert_id(pit[i].item())])
                
            elif subtask == 'c':
                p_answer = 'yes' if p_ans[i] else 'no'
                a_prompt = "Child: {} Parent: {}. <Child>: {}".format(
                    mask_token, p_answer, self.desc[self.convert_id(cit[i].item())])
            b_prompt = "<Query>: {}".format(self.desc[qid[i]])
            sentence_prompt = a_prompt + sep_token + b_prompt
            sentences_prompt.append(sentence_prompt)


            label_token = 'yes' if label[i].item() else 'no'
            pl_token = 'yes' if pl[i].item() else 'no'
            cl_token = 'yes' if cl[i].item() else 'no'
            
            if subtask == 'a':
                p_answer = 'yes' if p_ans[i] else 'no'
                c_answer = 'yes' if c_ans[i] else 'no'

                a_prompt = "All: {} Parent: {}  Child: {}. <Parent>: {} <Child>: {}".format(
                    label_token, p_answer, c_answer, self.desc[self.convert_id(pit[i].item())], self.desc[self.convert_id(cit[i].item())])
                
            elif subtask == 'p':
                a_prompt = "Parent: {}. <Parent>: {}".format(
                    pl_token, self.desc[self.convert_id(pit[i].item())])
                
            elif subtask == 'c':
                p_answer = 'yes' if p_ans[i] else 'no'
                a_prompt = "Child: {} Parent: {}. <Child>: {}".format(
                    cl_token, p_answer, self.desc[self.convert_id(cit[i].item())])
            sentence_prompt_gt = a_prompt + sep_token + b_prompt

            sentences_ground_truth.append(sentence_prompt_gt)

        sentences_prompt_pt = tokenizer(sentences_prompt, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        sentences_prompt_gt_pt = tokenizer(sentences_ground_truth, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = sentences_prompt_pt.input_ids
        attention_mask = sentences_prompt_pt.attention_mask
        labels = sentences_prompt_gt_pt.input_ids
        label_ids = torch.where(input_ids == mask_token_id, labels, -100)

        return input_ids, label_ids, attention_mask

    def scorer(self, pit, qid, cit, pct, pl, cl, pe, ce, p_ans=None, c_ans=None, subtask='a'):
        # [旁批] ★ Chain 的 scorer 返回的是【两个值】：分数和硬答案。
        #   硬答案 ans 会被 TrainerChain._test 拿去当下一模的 p_ans / c_ans（串行推理的关键）。
        batch_input_ids, batch_labels, batch_attention_mask = self.get_lm_inputs(pit, qid, cit, pct, pl, cl, pe, ce, p_ans=p_ans, c_ans=c_ans, subtask=subtask)
        # batch_lm_inputs = {'input_ids': batch_input_ids,
        #                    'attention_mask': batch_attention_mask,
        #                    'labels': batch_labels}
        batch_lm_inputs = {'input_ids': batch_input_ids,
                           'attention_mask': batch_attention_mask}
        batch_lm_inputs = {k: v.to(self.device) for k, v in batch_lm_inputs.items()}
        outputs = self.model(**batch_lm_inputs)
        score = outputs[0][batch_lm_inputs['input_ids'] == self.mask_token_id]
        pred_scores = score[:, self.yes_token_id] - score[:, self.no_token_id]
        pred_scores = pred_scores.view(pit.shape[0], -1)
        
        # results
        # [旁批] 硬答案 = logit(yes) > logit(no)（等价于 score > 0），没有阈值可调。
        #   这个 0-1 决策会被直接拼进下一模的 prompt 文本，所以错一步后面全错 —— chain 的弱点。
        ans = score[:, self.yes_token_id] > score[:, self.no_token_id]
        ans = ans.view(pit.shape[0], -1)
        return pred_scores, ans

    def forward(self, qs, ps, cs, bs, ws, pe, ce, be, we, pit, qit, cit, pct, pt, ct, p_ans=None, c_ans=None, subtask='a'):
        qid = [self.convert_id(qit[i].item()) for i in range(qit.shape[0])]
        return self.scorer(pit, qid, cit, pct, pt, ct, pe, ce, p_ans=p_ans, c_ans=c_ans, subtask=subtask)