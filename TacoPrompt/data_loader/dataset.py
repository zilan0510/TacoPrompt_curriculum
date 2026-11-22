# [导读] dataset.py —— 数据 + 负采样的全部实现（TacoPrompt 论文 §4.4 的自监督样本生成在这里）
#   三类东西：
#     1) Taxon          图的节点对象（比价/哈希都按 display_name，去重和 dict key 靠它）
#     2) MAGDataset     读 .terms/.taxo/.desc 建 networkx 有向图，并切分 train/valid/test
#     3) RawDataset     继承 torch 的 Dataset：预建正例集合 node2pos、负采样池 all_edges
#        └─ 四个负采样函数 _get_*_negatives：在 __getitem__ 里被调用，决定「1 个 query 配 1 正 + N 负」
#   关键函数备忘（不写行号，方便用 Ctrl+F）：
#     RawDataset.__init__          建 core_subgraph / node2pos / all_edges（一次性，训前跑一次）
#     RawDataset.__getitem__       返回一个 query 的 16 行，每行 17 字段 ← 数据流起点
#     _get_negative_positions      分发器（按 sampling_mode 选下面两个之一）
#     _get_at_most_k_negatives     sampling_mode=0：至多 N 个
#     _get_exactly_k_negatives     sampling_mode=1（本项目）：恰好 N 个
#     _my_get_exactly_k_negatives  死代码，从未被调用（可跳过）
#   与 TaxoDiff 的差别：TacoPrompt 把**每个节点**都直连了 pseudo_leaf（TaxoDiff 只连真叶子），
#   所以 TacoPrompt 里伪叶的 in_degree = 节点数，而「真叶子」要靠 out_degree==1 判定。
#   读法：先看 __getitem__ 那张「17 字段 → forward 入参」对照，再回头看 __init__ 里 node2pos/all_edges 怎么来的。

import os
import pickle
import random
import time
from collections import deque
from itertools import chain, product

import networkx as nx
import spacy
from networkx.algorithms import descendants
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

MAX_TEST_SIZE = 1000
MAX_VALIDATION_SIZE = 1000

# nlp = spacy.load('en_core_web_md')

#LIU
import torch
import numpy as np
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops
# def get_edges(edge_index_list):
#     undirected_edge_list = []
#     edge_index, _ = remove_self_loops(
#         torch.from_numpy(np.array(edge_index_list)).transpose(1, 0))  # remove self-loop
#     undirected_edge_list.append(to_undirected(edge_index))  # convert to undirected/bi-directed edge_index
#     return undirected_edge_list[0]

def get_edges(edge_index_list):
    undirected_edge_list = []
    edge_index_list = [[item[0], item[1]] for item in edge_index_list if item[0] != item[1]]
    for item in edge_index_list:
        if [item[1], item[0]] not in edge_index_list:
            edge_index_list.append([item[1], item[0]])
    edge_index = torch.from_numpy(np.array(edge_index_list)).transpose(1, 0)
    # edge_index, _ = remove_self_loops(
    #     torch.from_numpy(np.array(edge_index_list)).transpose(1, 0))  # remove self-loop
    undirected_edge_list.append(edge_index) # convert to undirected/bi-directed edge_index
    return undirected_edge_list[0]


class Taxon(object):
    def __init__(self, tx_id, rank=-1, norm_name="none", display_name="None", main_type="", level="-100", p_count=0,
                 c_count=0, create_date="None", use_wordnet=True, single_word=False):
        self.tx_id = tx_id
        self.rank = int(rank)
        self.norm_name = norm_name
        self.display_name = display_name
        self.main_type = main_type
        self.level = int(level)
        self.p_count = int(p_count)
        self.c_count = int(c_count)
        self.create_date = create_date
        self.use_wordnet = use_wordnet
        self.single_word = single_word
        self.description = ''
        # self.nlp_description = nlp('')

    def set_description(self, description):
        self.description = description
        # self.nlp_description = nlp(description)

    def __str__(self):
        return "Taxon {} (name: {}, level: {})".format(self.tx_id, self.norm_name, self.level)

    def __lt__(self, other):
        if self.display_name < other.display_name:
            return True
        else:
            return False

    def __eq__(self, other):
        # [旁批] ★ 只比 display_name。后果：
        #   「同名不同义」的两个节点会被当成同一个（SemEval-Verb 里同名词很多，有轻微噪音）
        #   但好处是图里同名节点会自动合并，避免重复候选位置。
        if self.display_name == other.display_name:
            return True
        else:
            return False

    def __hash__(self):
        # [旁批] 与 __eq__ 保持一致（都是 display_name），所以能当 set 元素 / dict key。
        #   all_edges 是 set of (Taxon,Taxon)，node2edge 是 set of (Taxon,Taxon)，都靠这个方法去重。
        return hash(self.display_name)


class MAGDataset(object):
    def __init__(self, name, path, embed_suffix="", raw=True, existing_partition=True, partition_pattern='internal'):
        """ Raw dataset class for MAG dataset

        Parameters
        ----------
        name : str
            taxonomy name
        path : str
            path to dataset, if raw=True, this is the directory path to dataset, if raw=False, this is the pickle path
        embed_suffix : str
            suffix of embedding file name, by default ""
        raw : bool, optional
            load raw dataset from txt (True) files or load pickled dataset (False), by default True
        existing_partition : bool, optional_my
            whether to use the existing the train/validation/test partitions or randomly sample new ones, by default False
        """
        self.name = name  # taxonomy name
        self.embed_suffix = embed_suffix
        self.existing_partition = existing_partition
        self.partition_pattern = partition_pattern
        self.vocab = []  # from node_id to human-readable concept string
        self.train_node_ids = []  # a list of train node_ids
        self.validation_node_ids = []  # a list of validation node_ids
        self.test_node_ids = []  # a list of test node_ids
        self.data_path = path

        if raw:
            self._load_dataset_raw(path)
        else:
            self._load_dataset_pickled(path)

    def _load_dataset_pickled(self, pickle_path):
        print('loading pickled dataset')
        with open(pickle_path, "rb") as fin:
            data = pickle.load(fin)
        self.name = data["name"]
        self.taxonomy = data['taxonomy']
        self.node_id2taxon = data['id2taxon']
        self.taxon2node_id = data['taxon2id']
        self.vocab = data["vocab"]
        self.train_node_ids = data["train_node_ids"]
        self.validation_node_ids = data["validation_node_ids"]
        self.test_node_ids = data["test_node_ids"]

        # path = os.path.split(pickle_path)[0]
        #
        # with open(os.path.join(path, f'{self.name}.terms.train'), 'w') as f:
        #     for id in self.train_node_ids:
        #         f.write(str(id) + '\n')
        # with open(os.path.join(path, f'{self.name}.terms.validation'), 'w') as f:
        #     for id in self.validation_node_ids:
        #         f.write(str(id) + '\n')
        # with open(os.path.join(path, f'{self.name}.terms.test'), 'w') as f:
        #     for id in self.test_node_ids:
        #         f.write(str(id) + '\n')
        # exit(0)
        print('dataset loaded')

    def _load_dataset_raw(self, dir_path):
        path = os.path.split(dir_path)[0]
        node_file_name = os.path.join(path, f"{self.name}.terms")
        edge_file_name = os.path.join(path, f"{self.name}.taxo")
        desc_file_name = os.path.join(path, f"{self.name}.desc")
        if self.embed_suffix == "":
            output_pickle_file_name = os.path.join(path, f"{self.name}.pickle.bin")
        else:
            output_pickle_file_name = os.path.join(path, f"{self.name}.{self.embed_suffix}.pickle.bin")

        tx_id2taxon = {}
        self.taxonomy = nx.DiGraph()

        # load nodes
        with open(desc_file_name, "r", encoding='utf-8') as fdesc:
            with open(node_file_name, "r", encoding='utf-8') as fin:
                for line, desc in tqdm(zip(fin, fdesc), desc="Loading terms"):
                    line = line.strip()
                    desc = desc.strip()
                    if line:
                        segs = line.split("\t")
                        segs_desc = desc.split("\t")
                        assert len(segs) == 2, f"Wrong number of segmentations {line}"
                        try:
                            assert segs[1] == segs_desc[0]
                            desc = segs_desc[1]
                        except AssertionError:
                            # assert len(segs_desc) == 1
                            desc = segs_desc[0]
                        taxon = Taxon(tx_id=segs[0], norm_name=segs[1], display_name=segs[1])
                        taxon.set_description(desc)
                        tx_id2taxon[segs[0]] = taxon
                        self.taxonomy.add_node(taxon)
        # load edges
        with open(edge_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading relations"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    parent_taxon = tx_id2taxon[segs[0]]
                    child_taxon = tx_id2taxon[segs[1]]
                    self.taxonomy.add_edge(parent_taxon, child_taxon)

        # generate vocab
        # [旁批] 这里出现了两套编号，务必分清（后面 model 里还会出现第三套）：
        #   tx_id      → .terms 文件里的原始编号（可能是任意整数）
        #   node_id    → 0..N-1 的连续编号（train_taxon2allemb_id 用的就是另一套，见 RawDataset）
        #   vocab[i]   → "concept@@@nodeid"，仅用于打印/调试
        # tx_id is the old taxon_id read from {self.name}.terms file, node_id is the new taxon_id from 0 to len(vocab)
        self.tx_id2node_id = {node.tx_id: idx for idx, node in enumerate(self.taxonomy.nodes())}
        self.node_id2tx_id = {v: k for k, v in self.tx_id2node_id.items()}
        self.node_id2taxon = {k: tx_id2taxon[v] for k, v in self.node_id2tx_id.items()}
        self.taxon2node_id = {v: k for k, v in self.node_id2taxon.items()}
        self.vocab = [tx_id2taxon[self.node_id2tx_id[node_id]].norm_name + "@@@" + str(node_id) for node_id in
                      self.node_id2tx_id]

        if self.existing_partition:
            # Use the pickled partitions
            # with open(os.path.join(path, 'split_node_ids.pickle'), 'rb') as f:
            #     data = pickle.load(f)
            # self.validation_node_ids = data["validation_node_ids"]
            # self.test_node_ids = data["test_node_ids"]
            # self.train_node_ids = data["train_node_ids"]

            # Or use the train/val/test files
            dir_path = os.path.dirname(dir_path)
            train_node_file_name = os.path.join(dir_path, f"{self.name}.terms.train")
            validation_node_file_name = os.path.join(dir_path, f"{self.name}.terms.validation")
            test_file_name = os.path.join(dir_path, f"{self.name}.terms.test")

            raw_train_node_list = self._load_node_list(train_node_file_name)
            raw_validation_node_list = self._load_node_list(validation_node_file_name)
            raw_test_node_list = self._load_node_list(test_file_name)

            self.train_node_ids = [int(n) for n in raw_train_node_list]
            self.validation_node_ids = [int(n) for n in raw_validation_node_list]
            self.test_node_ids = [int(n) for n in raw_test_node_list]

        else:
            print("Partition graph ...")
            # [旁批] ★ 两种切分策略，config 里默认 partition_pattern='internal'（上面的 else 分支）：
            #   'leaf'     → 只从叶子节点里抽验证/测试（leaf completion 设定）
            #   'internal' → 排除根，从其余所有节点里抽（本项目用这个）
            #   random.seed(47) 写死 → 第一次生成的 .pickle.bin 就是固定的划分，之后都读 pickle。
            #   验证集/测试集上限 MAX_VALIDATION_SIZE / MAX_TEST_SIZE = 1000，即 WordNet-Verb 上各抽 1000。
            if self.partition_pattern == 'leaf':
                sampled_node_ids = []
                for node in self.taxonomy.nodes():
                    if self.taxonomy.out_degree(node) == 0:
                        sampled_node_ids.append(self.tx_id2node_id[node.tx_id])
                random.seed(47)
                random.shuffle(sampled_node_ids)
            elif self.partition_pattern == 'internal':
                root_node = [node for node in self.taxonomy.nodes() if self.taxonomy.in_degree(node) == 0]
                sampled_node_ids = [self.tx_id2node_id[node.tx_id] for node in self.taxonomy.nodes() if
                                    node not in root_node]
                random.seed(47)
                random.shuffle(sampled_node_ids)
            else:
                raise ValueError('Unknown partition method!')

            validation_size = min(int(len(sampled_node_ids) * 0.1), MAX_VALIDATION_SIZE)
            test_size = min(int(len(sampled_node_ids) * 0.1), MAX_TEST_SIZE)
            self.validation_node_ids = sampled_node_ids[:validation_size]
            self.test_node_ids = sampled_node_ids[validation_size:(validation_size + test_size)]
            self.train_node_ids = [node_id for node_id in self.node_id2tx_id if
                                   node_id not in self.validation_node_ids and node_id not in self.test_node_ids]
            print("Finish partitioning graph ...")

        # save to pickle for faster loading next time
        print("start saving pickle data")
        with open(output_pickle_file_name, 'wb') as fout:
            data = {
                "name": self.name,
                "taxonomy": self.taxonomy,
                "id2taxon": self.node_id2taxon,
                "taxon2id": self.taxon2node_id,
                "vocab": self.vocab,
                "train_node_ids": self.train_node_ids,
                "validation_node_ids": self.validation_node_ids,
                "test_node_ids": self.test_node_ids,
            }
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_file_name}")

    def _load_node_list(self, file_path):
        node_list = []
        with open(file_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    node_list.append(line)
        return node_list


class RawDataset(Dataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100,
                 expand_factor=64, cache_refresh_time=128, test_topk=-1, tokenizer=None,
                 hard_ratio=0.0, hard_schedule=None):
        start = time.time()
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.cache_refresh_time = cache_refresh_time
        self.test_topk = test_topk
        # 结构化父节点负采样的状态（先置默认值；train 分支里会再填真实内容）
        self.hard_ratio = float(hard_ratio) if hard_ratio else 0.0
        self.hard_schedule = list(hard_schedule) if hard_schedule else []
        self.cur_epoch = 1
        self.cur_hard_ratio = 0.0
        self.cur_layers = []
        self._layer_cache = {}
        self._all_pool_set = set()
        self.hard_stats = self._new_hard_stats()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.taxon2id = graph_dataset.taxon2node_id
        self.id2taxon = graph_dataset.node_id2taxon

        full_graph = graph_dataset.taxonomy
        # xu: MAG-CS, 去自环
        # [旁批] 去自环很关键：有自环的话 nx.descendants 会把节点当成自己的后代，
        #   all_edges 里就会出现 (x, x)，而位置 (p,q,c) 要求 p != c。
        full_graph.remove_edges_from(nx.selfloop_edges(full_graph))

        # self.full_graph = full_graph
        train_nodes = [self.id2taxon[node_id] for node_id in graph_dataset.train_node_ids]
        roots = [node for node in full_graph.nodes() if full_graph.in_degree(node) == 0]
        # [旁批] pseudo_root：虚拟根，连向所有入度为 0 的节点。
        #   作用：保证「每个节点都有父」，否则顶层节点的正例位置（父×子）会是空集，无法算 rank。
        #   代价：会多出一大批 (pseudo_root, c) 的候选位置（占总池子约 23%），且它的 description 是空串。
        self.pseudo_root_node = Taxon(tx_id='', norm_name='', display_name='pseudo root')
        full_graph.add_node(self.pseudo_root_node)
        for node in roots:
            full_graph.add_edge(self.pseudo_root_node, node)
        train_nodes.append(self.pseudo_root_node)
        self.full_graph = full_graph

        if mode == 'train':
            # add pseudo leaf node to core graph
            datapath = os.path.split(graph_dataset.data_path)[0]
            graph_pickle_path = os.path.join(datapath, 'subgraphs.pickle')
            graph_pickled = False
            if os.path.isfile(graph_pickle_path):
                graph_pickled = True
                with open(graph_pickle_path, 'rb') as f:
                    graphs = pickle.load(f)

            print('adding pseudo leaf')
            if graph_pickled:
                self.core_subgraph = graphs['core_subgraph']
                self.pseudo_leaf_node = graphs['pseudo_leaf_node']
            else:
                self.core_subgraph = self._get_holdout_subgraph(train_nodes)
                print('holding out subgraph')
                self.pseudo_leaf_node = Taxon(tx_id='', norm_name='', display_name='pseudo leaf')
                self.core_subgraph.add_node(self.pseudo_leaf_node)
                # [旁批] ★ 与 TaxoDiff 的重要差异：这里把**每一个节点**都直连到 pseudo_leaf，
                #   而 TaxoDiff 只连 out_degree==0 的真叶子。两个后果：
                #     1) pseudo_leaf 的 in_degree = 节点总数，它是全图唯一的 sink
                #     2) 每个节点的 children 都包含 pseudo_leaf，所以后面必须用
                #        `if len(children) > 1: children.remove(pseudo_leaf)` 把它剔掉，
                #        否则所有节点都会被当成「能挂克节点」，正例位置全部爆炸。
                #   好处：任意 query 都能表达成「插入为叶子」（父=真父，子=pseudo_leaf），连非叶子节点也有这个选项。
                for node in list(self.core_subgraph.nodes()):
                    self.core_subgraph.add_edge(node, self.pseudo_leaf_node)

            # liu: adj mat
            # edges = list(self.core_subgraph.edges())
            # tmp_id2taxon = {idx: taxon for idx, taxon in enumerate(self.core_subgraph.nodes())}
            # tmp_taxon2id = {v: k for k, v in tmp_id2taxon.items()}
            # train_edges_list = [[tmp_taxon2id[item[0]], tmp_taxon2id[item[1]]] for item in edges if item[0] != self.pseudo_leaf_node and item[1] != self.pseudo_leaf_node]
            # edge_list = get_edges(train_edges_list)
            # self.adj = edge_list.long()
            # print(torch.max(self.adj[1]))
            ######################################################################################

            self.taxon2id[self.pseudo_leaf_node] = len(full_graph.nodes)
            self.taxon2id[self.pseudo_root_node] = len(full_graph.nodes) - 1
            self.id2taxon[len(full_graph.nodes)] = self.pseudo_leaf_node
            self.id2taxon[len(full_graph.nodes) - 1] = self.pseudo_root_node
            # [旁批?] 注意：pseudo_root 已经在 full_graph 里了（上面 add_node 过），所以 len(full_graph.nodes)-1
            #   很可能和某个真实节点的 node_id 撞车。实测没出问题（因为 taxon2id 是反向字典，覆盖后自洽），
            #   但如果要改 dataset 结构，这一行值得重新检查。
            # [旁批] 因为所有节点都连了 pseudo_leaf，所以「真叶子」的判据变成 out_degree == 1（只有那条虚拟边）。
            self.leaf_nodes = [node for node in self.core_subgraph.nodes() if self.core_subgraph.out_degree(node) == 1]

            # add interested node list and subgraph
            # remove supersource nodes (i.e., nodes without in-degree 0)
            # self.node_list = [n for n in train_nodes if n not in roots]
            self.node_list = [n for n in train_nodes if n != self.pseudo_root_node]
            # getitem的idx找到对应的taxon，使用在all_embed中找对应的结构emb
            # [旁批] ★ 第三套编号登场，三套要分清（这是本项目最容易搞错的地方）：
            #   node_id    → 全图 0..N-1（graph_dataset.taxon2node_id，taxon2id 就是它）
            #   allemb_id  → core_subgraph 里的行号（下面这两个字典，pit/qit/cit 存的就是它）
            #   idx        → node_list 的下标（DataLoader 传给 __getitem__ 的 idx）
            #   转换关系：allemb_id --train_allemb_id2taxon--> Taxon --taxon2id--> node_id
            #   模型里的 convert_id(id) 就是走这条路径。两套编号不相等，直接混用会取到错的概念描述。
            self.train_allemb_id2taxon = {idx: taxon for idx, taxon in enumerate(self.core_subgraph.nodes())}
            self.train_taxon2allemb_id = {v: k for k, v in self.train_allemb_id2taxon.items()}

            # For TEMP: 加入path
            # [旁批] parent2paths 只被 TEMPCompletion（baseline）用，TacoPromptHidden / Chain 都不读。
            #   内容：每个节点从 pseudo_root 出发的一条最短路径（节点的 norm_name 串成字符串）。
            #   不管跑不跑 TEMP，这个文件都会被生成并缓存成 parent2paths.pickle。
            print("adding parent2path...")
            datapath = os.path.split(graph_dataset.data_path)[0]
            path_pickle_path = os.path.join(datapath, 'parent2paths.pickle')

            if os.path.isfile(path_pickle_path):
                with open(path_pickle_path, 'rb') as f:
                    paths = pickle.load(f)
                self.parent2paths = paths['parent2paths']
            else:
                print("generating parent2path...")
                self.parent2paths = {}
                for node in self.core_subgraph.nodes():
                    try:
                        self.parent2paths[node] = list(nx.all_shortest_paths(self.core_subgraph, self.pseudo_root_node, node))[0]
                    except:
                        print("None Path!")
                        self.parent2paths[node] = [node]
                for node, path in self.parent2paths.items():
                    self.parent2paths[node] = ",".join([n.norm_name for n in path])
                with open(path_pickle_path, 'wb') as f:
                    pickle.dump({
                        'parent2paths': self.parent2paths
                    }, 
                    f, protocol=pickle.HIGHEST_PROTOCOL)

            # build node2pos, node2edge
            print('building node2pos, node2edge')
            # [旁批] 四个字典构成「一个 query 的局部结构」，也是负采样的过滤依据：
            #   node2parents / node2children → 判定 yp / yc（负例也分「父对子错」「父错子对」）
            #   node2edge  → 与 query 直接相邻的边（负采样必须排除，否则会采到真边）
            #   node2pos   → 正例位置集 = 父集合 × 子集合（笛卡尔积，所以可能不止一个）
            self.node2pos, self.node2edge = {}, {}
            self.node2parents, self.node2children = {}, {}
            for node in self.node_list:
                parents = set(self.core_subgraph.predecessors(node))
                children = set(self.core_subgraph.successors(node))
                # [旁批] 因为所有节点都连了 pseudo_leaf，所以多个子时必须把伪叶剔掉；
                #   只剩一个子且它是伪叶时则保留 —— 这就是「插入为叶子」的表示。
                if len(children) > 1:
                    children = [i for i in children if i != self.pseudo_leaf_node]
                node_pos_edges = [(pre, suc) for pre in parents for suc in children if pre != suc]
                if len(node_pos_edges) == 0:
                    node_pos_edges = [(pre, suc) for pre in parents for suc in children]

                self.node2edge[node] = set(self.core_subgraph.in_edges(node)).union(
                    set(self.core_subgraph.out_edges(node)))
                self.node2pos[node] = node_pos_edges
                self.node2parents[node] = parents
                self.node2children[node] = children

            print('building valid and test node list')
            # [旁批] valid/test 的节点已经从 core_subgraph 里删掉了，所以要单独建两件事：
            #   valid_holdout_subgraph / test_holdout_subgraph  → 包含「train + 这批待测节点」的图（有真实父子关系）
            #   valid_node2pos / test_node2pos                  → 每个待测 query 的正确插入位置
            # 注意：holdout 图里待测节点的子仍然指向 pseudo_leaf（下面那个 out_degree==0 循环），
            #   但这里只对「出度为 0」的节点加边，而不是全部 —— 与 core_subgraph 的做法不一致。
            self.valid_node_list = [self.id2taxon[node_id] for node_id in graph_dataset.validation_node_ids]
            if graph_pickled:
                self.valid_holdout_subgraph = graphs['valid_subgraph']
            else:
                self.valid_holdout_subgraph = self._get_holdout_subgraph(train_nodes + self.valid_node_list)
                self.valid_holdout_subgraph.add_node(self.pseudo_leaf_node)
                for node in [node for node in self.valid_holdout_subgraph.nodes() if
                             self.valid_holdout_subgraph.out_degree(node) == 0]:
                    self.valid_holdout_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.valid_id2taxon = {idx: taxon for idx, taxon in enumerate(self.valid_holdout_subgraph.nodes())}
            self.valid_taxon2id = {v: k for k, v in self.valid_id2taxon.items()}
            self.valid_node2pos = self._find_insert_position(self.valid_node_list, self.valid_holdout_subgraph)

            self.test_node_list = [self.id2taxon[node_id] for node_id in graph_dataset.test_node_ids]
            if graph_pickled:
                self.test_holdout_subgraph = graphs['test_subgraph']
            else:
                self.test_holdout_subgraph = self._get_holdout_subgraph(train_nodes + self.test_node_list)
                self.test_holdout_subgraph.add_node(self.pseudo_leaf_node)
                for node in [node for node in self.test_holdout_subgraph.nodes() if
                             self.test_holdout_subgraph.out_degree(node) == 0]:
                    self.test_holdout_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.test_id2taxon = {idx: taxon for idx, taxon in enumerate(self.test_holdout_subgraph.nodes())}
            self.test_taxon2id = {v: k for k, v in self.test_id2taxon.items()}
            self.test_node2pos = self._find_insert_position(self.test_node_list, self.test_holdout_subgraph)

            if not graph_pickled:
                with open(graph_pickle_path, 'wb') as f:
                    pickle.dump({
                        'pseudo_leaf_node': self.pseudo_leaf_node,
                        'core_subgraph': self.core_subgraph,
                        'valid_subgraph': self.valid_holdout_subgraph,
                        'test_subgraph': self.test_holdout_subgraph
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)

            # used for sampling negative positions during train/validation stage
            # [旁批] ★ 负采样候选池：core_subgraph 里所有「祖先, 后代」对（不是所有边）。
            #   SemEval-Verb 实测：11938 个节点 / 23904 条边 → 51159 个候选位置，其中直接边只占 46.7%。
            #   random.shuffle 只洗一次，之后靠指针顺序切片 —— 相当于「每轮把 5 万张牌洗一次，然后顺序切」。
            self.pointer = 0
            self.all_edges = list(self._get_candidate_positions(self.core_subgraph))
            random.shuffle(self.all_edges)
            # [旁批] node2pos_node：每个 query 的「父集合 / 子集合」，并把相邻边也加进去。
            #   只被 _my_get_exactly_k_negatives 使用，而那个函数从未被调用 → 这里算完基本是只写不读。
            #   （属于早期试验的遗留物，可跳过细读。）
            self.node2pos_node = {}
            tot = 0
            for node, eles in self.node2pos.items():
                self.node2pos_node[node] = [set(), set()]
                # xu:正例只有一边时只加入前面
                if len(eles) == 1 and eles[0][1] is self.pseudo_leaf_node:
                    self.node2pos_node[node][0].add(eles[0][0])
                    tot += 1
                    continue
                for ele in eles:
                    self.node2pos_node[node][0].add(ele[0])
                    self.node2pos_node[node][1].add(ele[1])
            print(tot, len(self.node2pos))
            for node, eles in self.node2edge.items():
                for ele in eles:
                    self.node2pos_node[node][0].add(ele[0])
                    self.node2pos_node[node][1].add(ele[1])

            # ===== 结构化父节点负采样（课程学习）=====
            # 依据：错误分析里「父错子对」的 1~9 类（第 10 类 = 无公共祖先 / pseudo root，
            #       没有可定位的结构特征，不主动采样）。
            # hard_ratio <= 0 且 hard_schedule 为空时，整条路径不消耗任何额外 RNG，
            # 行为与改造前逐位一致（可用于确认 baseline 未被改动）。
            self.hard_ratio = float(hard_ratio) if hard_ratio else 0.0
            self.hard_schedule = list(hard_schedule) if hard_schedule else []
            self.cur_epoch = 1
            self.cur_hard_ratio = 0.0
            self.cur_layers = []
            self._layer_cache = {}                    # p0 -> {layer: [候选父, ...]}（按 p0 缓存，跨 epoch 复用）
            self._all_pool_set = set(self.all_edges)  # 合法性校验用
            self.hard_stats = self._new_hard_stats()
            if self.hard_ratio > 0 or self.hard_schedule:
                self.set_epoch(1)

        end = time.time()
        print(f"Finish loading dataset ({end - start} seconds)")

    # ------------------------------------------------------------------ #
    #           结构化父节点负采样（课程学习）工具函数                      #
    # ------------------------------------------------------------------ #
    # 各层权重 = 两份错误分析里「叶子 query 父错子对」各层占比的平均值
    LAYER_WEIGHTS = {1: 5.27, 2: 8.31, 3: 5.19, 4: 2.13, 5: 2.49, 6: 0.71, 7: 0.46, 8: 0.52}

    @staticmethod
    def _new_hard_stats():
        return {
            'hard': 0,
            'rand': 0,
            'shortfall': 0,                               # 某层池子不足配额、被迫放回补齐的次数
            'empty_layer': {i: 0 for i in range(1, 9)},   # 某层池子为空、配额作废的次数
            'layer_hist': {i: 0 for i in range(1, 9)},    # 实际采到的各层数量
        }

    def set_epoch(self, epoch):
        """由 trainer 在每个 epoch 开头调用，推进课程阶段（由远及近：L6~L8 → L4 → L2 → L1）。"""
        self.cur_epoch = epoch
        self.cur_hard_ratio = 0.0
        self.cur_layers = []
        for end_epoch, ratio, layers in self.hard_schedule:
            if epoch <= end_epoch:
                self.cur_hard_ratio = float(ratio)
                self.cur_layers = list(layers)
                break
        else:
            if self.hard_schedule:                       # 超出最后一段：沿用最后一段
                _, ratio, layers = self.hard_schedule[-1]
                self.cur_hard_ratio = float(ratio)
                self.cur_layers = list(layers)

        # 兜底：没配 schedule 但给了 hard_ratio → 全程固定比例 + 全 8 层
        if not self.hard_schedule and self.hard_ratio > 0:
            self.cur_hard_ratio = self.hard_ratio
            self.cur_layers = list(range(1, 9))

        if not self.cur_layers:                          # warmup：层列表为空 → 强制关掉 hard
            self.cur_hard_ratio = 0.0

        self.hard_stats = self._new_hard_stats()         # 统计按 epoch 归零，便于核对配额
        print(f'[hard-neg] epoch {epoch}: ratio={self.cur_hard_ratio}, layers={self.cur_layers}')

    def _layers_of(self, p0):
        """以正确父 p0 为基准，按 core 子图上的 BFS 距离给出各层候选父（结果按 p0 缓存）。

        层定义与错误分析的 10 类一一对应（距离取最短距离）：
          L1 上 1 跳（直接父）、L2 下 1 跳（直接子）、L3 兄弟、L4 上 2 跳、L5 下 2 跳、
          L6 上 3 跳、L7 下 3 跳、L8 上/下 4 跳及以上。
        「与 p0 无公共祖先」的节点不进入任何一层（对应第 10 类，不主动采样）。
        优先级：祖先/后代 > 兄弟（与错误分析的判定顺序一致）。
        """
        if p0 in self._layer_cache:
            return self._layer_cache[p0]

        up = {p0: 0}                                     # 祖先 -> 距离
        frontier = [p0]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for n in frontier:
                for pp in self.node2parents.get(n, ()):
                    if pp not in up:
                        up[pp] = d
                        nxt.append(pp)
            frontier = nxt

        down = {p0: 0}                                   # 后代 -> 距离
        frontier = [p0]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for n in frontier:
                for ch in self.node2children.get(n, ()):
                    if ch not in down:
                        down[ch] = d
                        nxt.append(ch)
            frontier = nxt

        layers = {i: [] for i in range(1, 9)}
        for x, dist in up.items():
            if dist == 0:
                continue
            layers[{1: 1, 2: 4, 3: 6}.get(dist, 8)].append(x)
        for x, dist in down.items():
            if dist == 0:
                continue
            layers[{1: 2, 2: 5, 3: 7}.get(dist, 8)].append(x)

        # L3 兄弟：所有父的孩子并集，去掉 p0 与已归入祖先/后代层的节点
        sib = set()
        for pp in self.node2parents.get(p0, ()):
            sib.update(self.node2children.get(pp, ()))
        sib.discard(p0)
        sib.difference_update(set(up) | set(down))
        layers[3] = list(sib)

        self._layer_cache[p0] = layers
        return layers

    def _classify_parent(self, p0, p_prime):
        """返回 p_prime 相对正确父 p0 的难度层（1..8）；p_prime == p0 返回 0；
        与 p0 无公共祖先（第 10 类）返回 None。仅用于诊断打印。"""
        if p_prime is p0:
            return 0
        for layer, bucket in self._layers_of(p0).items():
            if any(p_prime is x for x in bucket):
                return layer
        return None

    def _filter_hard(self, bucket, p0, c, pos_set, query_node=None):
        """逐个候选父做合法性过滤：
        0) p' 不能是 query 自身（测试时候选池里根本没有 q）
        1) p' != p0；2) 不能是 pseudo_root / pseudo_leaf（对应「第 10 类」，不主动采）；
        3) p' 必须在 train_taxon2allemb_id（拼 prompt 要查描述）；
        4) (p', c) 必须真的在候选池 all_edges 里；5) (p', c) 不能撞上该 query 的其它正例位置。
        注：pseudo 节点用 == 判等（Taxon.__eq__ 按 display_name），
        因为 core_subgraph 可能来自 pickle，对象不是同一实例。
        """
        out = []
        for p in bucket:
            if p is p0 or p == p0:
                continue
            if query_node is not None and (p is query_node or p == query_node):
                continue
            if p == self.pseudo_root_node or p == self.pseudo_leaf_node:
                continue
            if p not in self.train_taxon2allemb_id:
                continue
            if (p, c) in pos_set:
                continue
            if (p, c) not in self._all_pool_set:
                continue
            out.append(p)
        return out

    def _allocate_quotas(self, n_hard, layers):
        """在【当前开放层】内按 LAYER_WEIGHTS 重新归一化，用最大余数法拆配额，总和恰为 n_hard。"""
        ws = {l: self.LAYER_WEIGHTS[l] for l in layers if l in self.LAYER_WEIGHTS}
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
        """先按层配额取 hard 负例，再补齐随机负例，保证恰好 negative_size 个位置。"""
        rho = self.cur_hard_ratio
        if rho <= 0 or not self.cur_layers or not getattr(self, 'node2parents', None):
            return self._get_negative_positions(query_node, negative_size)

        n_hard = int(round(negative_size * rho))
        if n_hard <= 0:
            return self._get_negative_positions(query_node, negative_size)

        p0, c = random.choice(positive_positions)        # 与本次正例对齐，避免正负位置互相矛盾
        layers = self._layers_of(p0)
        pos_set = set(self.node2pos[query_node])
        quota = self._allocate_quotas(n_hard, self.cur_layers)

        hard_negs = []
        for layer in self.cur_layers:                    # 严格按层遍历，不跨界补
            need = quota.get(layer, 0)
            if need <= 0:
                continue
            bucket = self._filter_hard(layers.get(layer, ()), p0, c, pos_set, query_node)
            if not bucket:
                self.hard_stats['empty_layer'][layer] += 1
                continue                                 # 配额作废，交给随机部分
            if len(bucket) >= need:
                picked = random.sample(bucket, need)
            else:
                picked = random.choices(bucket, k=need)  # 池子不足：放回补齐
                self.hard_stats['shortfall'] += 1
            hard_negs.extend((p, c) for p in picked)
            self.hard_stats['layer_hist'][layer] += len(picked)

        hard_negs = list(dict.fromkeys(hard_negs))       # 放回采样可能重复，去重
        self.hard_stats['hard'] += len(hard_negs)

        n_rand = negative_size - len(hard_negs)
        rand_negs = self._get_exactly_k_negatives(query_node, n_rand, ignore=hard_negs) if n_rand > 0 else []
        self.hard_stats['rand'] += len(rand_negs)

        return hard_negs + rand_negs

    def __str__(self):
        return f"{self.__class__.__name__} mode:{self.mode}"

    def __len__(self):
        # [旁批] 「项数」= 训练 query 数（11937），不是样本行数。DataLoader 用它算 len(loader)。
        return len(self.node_list)

    def __getitem__(self, idx):
        """ Generate an data instance based on train/validation/test mode.

        If self.sampling_mode == 0:
            This list may contain more than one triplets with label = 1
        If self.sampling_mode == 1:
            This list contain one and ONLY one triplet with label = 1, others have label = 0
        """
        res = []
        q = self.node_list[idx]

        # generate positive triplet(s)
        # [旁批] sampling_mode=1（本项目）：正例只随机取 1 条；=0：正例取全部（超过 max_pos_size 时随机抽）。
        #   「随机取 1 条」意味着一个 query 的多个真实位置会在不同 epoch 被轮流命中，
        #   但同一个 epoch 内模型只看得到它的 1 个正确位置 —— 这是 H@k 和 R@k 会产生差异的根源之一。
        if self.sampling_mode == 0:
            positive_positions = self.node2pos[q]
            if len(positive_positions) > self.max_pos_size and self.mode == 'train':
                positive_positions = random.sample(positive_positions, k=self.max_pos_size)
        else:  # self.sampling_mode > 0
            positive_positions = [random.choice(self.node2pos[q])]

        # select negative triplet(s)
        # [旁批?] 这行有 bug：res 此刻还是空列表，所以 len(res) 恒为 0，
        #   只有当 negative_size == -1 时才会得到 0 个负样本。config 里是 15，走不到那个分支。
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        # [旁批] ★ 负采样的唯一调用点：一个 query、一个 epoch 只调这一次。
        #   改造后走 _get_mixed_negatives：先按层配额取结构化 hard 负例，再用原来的随机池补齐。
        #   hard_ratio=0（且无 hard_schedule）时它内部直接转回原来的纯随机采样，
        #   且不消耗任何额外 RNG → 与改造前逐位一致。
        negative_positions = self._get_mixed_negatives(q, negative_size, positive_positions)

        # # Tags: position, parent, child, best, worst
        # for p, c in positive_positions:
        #     best, worst = self.get_best_worst_siblings(p, q, g=self.core_subgraph)
        #     res.append([q.description, p.description, c.description, best.description, worst.description, 1, 1, 1, 1, 1,
        #                 bool(p.description),
        #                 bool(c.description),
        #                 bool(best.description),
        #                 bool(worst.description)])
        # for p, c in negative_positions:
        #     best, worst = self.get_best_worst_siblings(p, q, g=self.core_subgraph)
        #     res.append([q.description, p.description, c.description, best.description, worst.description, 0,
        #                 int(p in self.node2parents[q]),
        #                 int(c in self.node2children[q]),
        #                 int(not set(self.core_subgraph.predecessors(best)).isdisjoint(
        #                     set(self.core_subgraph.predecessors(q)))),
        #                 int(not set(self.core_subgraph.predecessors(worst)).isdisjoint(
        #                     set(self.core_subgraph.predecessors(q)))),
        #                 bool(p.description),
        #                 bool(c.description),
        #                 bool(best.description),
        #                 bool(worst.description)])
        # Tags: position, parent, child, best, worst
        # 加入了p, q, c的idx
        # [旁批] ★★ 一行 17 个字段，顺序 = collate_fn 解包顺序 = model.forward 的入参。逐列含义：
        #   0 q.description → qs | 1 p.description → ps | 2 c.description → cs | 3→bs | 4→ws
        #   5 pct（位置整体对不对）| 6 pt（父对不对）| 7 ct（子对不对）| 8 bt | 9 wt
        #   10 pe（p 的 description 非空）| 11 ce | 12 be | 13 we
        #   14 pit | 15 qit | 16 cit （三个「嵌入行号」，即 train_taxon2allemb_id 的值）
        #   模型真正消费的只有：5/6/7（三态标签）、10/11（两个掩码）、14/15/16（三个行号）。
        #   qs/ps/cs/bs/ws 虽然被 tokenize 了，但 forward 完全不用（白浪费 CPU）。
        # [旁批] best/worst 被写死成 pseudo_leaf（description 为空串），所以 be/we 恒为 False、
        #   bt/wt 基本恒为 1。这两列是 TacoPrompt「最好/最差兄弟」的遗留位，对 TacoPrompt 本身无效。
        for p, c in positive_positions:
            # best, worst = self.get_best_worst_siblings(p, q, g=self.core_subgraph)
            best, worst = self.pseudo_leaf_node, self.pseudo_leaf_node
            res.append([q.description, p.description, c.description, best.description, worst.description, 1, 1, 1, 1, 1,
                        bool(p.description),
                        bool(c.description),
                        bool(best.description),
                        bool(worst.description),
                        int(self.train_taxon2allemb_id[p]),
                        int(self.train_taxon2allemb_id[q]),
                        int(self.train_taxon2allemb_id[c])])
        for p, c in negative_positions:
            best, worst = self.pseudo_leaf_node, self.pseudo_leaf_node
            # [旁批] ★ 三态标签（TacoPrompt 相对 TaxoExpan 的关键设计）：
            #   pct=0 —— 位置一定要错的
            #   pt = int(p in node2parents[q])  —— 父是对的吗？
            #   ct = int(c in node2children[q]) —— 子是对的吗？
            #   于是负例自动分成了「父对子错」「父错子对」「全错」三类，分别用三个 MASK 位学。
            #   [旁批?] pe/ce 是 bool(description)，意思是「这个节点有没有描述文本」，
            #   不是「节点存不存在」。pseudo_root / pseudo_leaf 的描述是空串，所以这两类行会被掩掉。
            res.append([q.description, p.description, c.description, best.description, worst.description, 0,
                        int(p in self.node2parents[q]),
                        int(c in self.node2children[q]),
                        int(not set(self.core_subgraph.predecessors(best)).isdisjoint(
                            set(self.core_subgraph.predecessors(q)))),
                        int(not set(self.core_subgraph.predecessors(worst)).isdisjoint(
                            set(self.core_subgraph.predecessors(q)))),
                        bool(p.description),
                        bool(c.description),
                        bool(best.description),
                        bool(worst.description),
                        int(self.train_taxon2allemb_id[p]),
                        int(self.train_taxon2allemb_id[q]),
                        int(self.train_taxon2allemb_id[c])])
            # if res[-1][8] == 1 and bool(best.description):
            #     print(q.description, best.description)
            #     print(list(set(self.core_subgraph.predecessors(q)).intersection(set(self.core_subgraph.predecessors(best))))[0].display_name)

        return tuple(res)

    def get_best_worst_siblings(self, parent, query, g=None, train=True):
        probability = 0.9
        best_score, worst_score = -2, 2
        best, worst = self.pseudo_leaf_node, self.pseudo_leaf_node
        sibling_list = list(g.successors(parent))
        # if parent in g.predecessors(query):
        #     sibling_list += list(g.successors(query))
        no_pseudo = False
        if len(sibling_list) > 1:
            no_pseudo = True
        for n in sibling_list:
            if n == query or (n.description == '' and no_pseudo):
                continue
            semantic_similarity = query.nlp_description.similarity(n.nlp_description)
            if semantic_similarity > best_score:
                best = n
                best_score = semantic_similarity
            if semantic_similarity < worst_score:
                worst = n
                worst_score = semantic_similarity
        r = random.choices([True, False], [probability, 1 - probability])[0]
        if train:
            if r:
                best = random.sample(sibling_list, k=1)[0]
            r = random.choices([True, False], [probability, 1 - probability])[0]
            if r:
                worst = random.sample(sibling_list, k=1)[0]
        return best, worst

    def _get_holdout_subgraph(self, nodes):
        # [旁批] 用途：拿走 nodes 以外的所有节点，并在被拿走的节点位置「跳过它」直接连边。
        #   例：A→B→C，B 被拿走 → 子图里补一条 A→C（jump edge）。
        #   这保证了「待测节点的父/子关系不会因为它不在子图里而丢失」。
        #   后半段 remove jump edges：如果 A→C 同时已经能通过 A→B'→C 到达，就删掉冗余的 A→C，
        #   避免同一个层级关系在候选池里出现两次。
        node_to_remove = [n for n in self.full_graph.nodes if n not in nodes]
        subgraph = self.full_graph.subgraph([node for node in nodes]).copy()
        for node in node_to_remove:
            parents = set()
            children = set()
            ps = deque(self.full_graph.predecessors(node))
            cs = deque(self.full_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(self.full_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(self.full_graph.successors(c))
            for p, c in product(parents, children):
                subgraph.add_edge(p, c)
        # remove jump edges
        node2descendants = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
        for node in subgraph.nodes():
            if subgraph.out_degree(node) > 1:
                successors1 = set(subgraph.successors(node))
                successors2 = set(chain.from_iterable([node2descendants[n] for n in successors1]))
                checkset = successors1.intersection(successors2)
                if checkset:
                    for s in checkset:
                        if subgraph.in_degree(s) > 1:
                            subgraph.remove_edge(node, s)
        return subgraph

    def _get_candidate_positions(self, graph):
        # [旁批] ★ 候选位置池 = 所有「严格祖先, 后代」对，不是所有边。
        #   descendants 不含自身，所以永远不会出现 (x, x)。
        #   这里也隐含了「p 不可能是 pseudo_leaf」（它是 sink，没有后代），
        #   但 p 可能是 pseudo_root（它有无穷多后代，占总池子约 23%）。
        node2descendants = {n: set(descendants(graph, n)) for n in graph.nodes}
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()]))
        return candidates

    def _find_insert_position(self, node_ids, holdout_graph, ignore=[]):
        # [旁批] 「跳层」就发生在这里：从节点的真实父/子出发 BFS，遇到不在 core_subgraph 的就继续往上/往下钻，
        #   直到找到第一个「还活着」的节点。所以真值可能跳过中间层（例：真子是 test 节点时，真值会下沉到它的子）。
        #   实测 SemEval-Verb：父方向跳层 0/1000，子方向 6/1000，所以影响很小。
        #   [旁批?] 注意 subgraph 是写死的 self.core_subgraph，不是参数传进来的 —— 验证和测试共用同一可见集。
        node2pos = {}
        subgraph = self.core_subgraph
        for node in node_ids:
            if node in ignore:
                continue
            parents = set()
            children = set()
            ps = deque(holdout_graph.predecessors(node))
            cs = deque(holdout_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(holdout_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(holdout_graph.successors(c))
            if not children:
                children.add(self.pseudo_leaf_node)
            position = [(p, c) for p in parents for c in children if p != c]
            node2pos[node] = position
        return node2pos

    def _get_negative_positions(self, query_node, negative_size):
        # [旁批] ★ 负采样分发器。四件事速查：
        #   ① 数量：由 sampling_mode 决定（0=至多 N，1=恰好 N）
        #   ② 随机/结构化：两个分支都是「共享候选池 + 全局游标」的结构化取法
        #   ③ 时机：每个 epoch、每个 query 各调一次；池子只在 __init__ 建一次
        #   ④ 改造落点：在这里加分流（按 epoch 号 / hard_ratio），并让 trainer 在 epoch 开头刷新游标
        if self.sampling_mode == 0:
            return self._get_at_most_k_negatives(query_node, negative_size)
        elif self.sampling_mode == 1:
            return self._get_exactly_k_negatives(query_node, negative_size)

    def _get_at_most_k_negatives(self, query_node, negative_size):
        """ Generate AT MOST negative_size samples for the query node
        """
        # [旁批] sampling_mode=0 的分支（本项目不用）。四件事：
        #   ① 数量：至多 N 个 —— 切片里被过滤掉的不补，所以经常明显少于 N
        #   ② 随机/结构化：结构化（全局 pointer 顺序切片）；对单个 query 来说不是独立随机采样
        #   ③ 时机：每个 epoch、每个 query 一次
        #   ④ 改造：改负采样时可以完全不动这个分支，只改 exactly 那一支
        if self.pointer == 0:
            random.shuffle(self.all_edges)

        while True:
            negatives = [ele for ele in self.all_edges[self.pointer: self.pointer + negative_size] if
                         ele not in self.node2pos[query_node] and ele not in self.node2edge[query_node]]
            if len(negatives) > 0:
                break
            self.pointer += negative_size
            if self.pointer >= len(self.all_edges):
                self.pointer = 0

        return negatives

    def _get_exactly_k_negatives(self, query_node, negative_size, ignore=[]):
        """ Generate EXACTLY negative_size samples for the query node
        """
        # [旁批] ★★ 本项目实际使用的分支（config: sampling_mode=1）。四件事：
        #   ① 数量：恰好 N 个（config 是 15）—— 不够就继续往后切片补，多了截断
        #   ② 随机/结构化：结构化。all_edges 只在指针回到 0 时洗一次，之后靠全局 pointer 顺序切段，
        #      所以同一批次相邻 query 拿到的负例在池子里是相邻的（不是每个 query 独立随机采样）
        #   ③ 时机：每个 epoch、每个 query 一次；pointer 跨 query 累加，所以一个 epoch 会消耗 N_query×N 个槽位
        #   ④ 改造落点：可以在这里按 hard_ratio 把「随机负例」换成「结构化难负例」，
        #      基础模板：固定真父向下取子 / 固定真子向上取父 / 兄弟替换（后者必须显式校验合法性）
        #   两个过滤条件：
        #      ele not in node2pos[q]  → 不能是 q 的正确位置（父×子）
        #      ele not in node2edge[q] → 不能正好是 q 的相邻边（(p,q) 或 (q,c)）
        #      [旁批?] 但 node2edge 只含直接边，所以 (远祖(q), q) 这类候选（把 q 自己当 child）
        #      不会被过滤，会以负例形式出现 —— 这是这一版的小漏洞。
        if self.pointer == 0:
            random.shuffle(self.all_edges)

        negatives = []
        while len(negatives) != negative_size:
            n_lack = negative_size - len(negatives)
            negatives.extend([ele for ele in self.all_edges[self.pointer: self.pointer + n_lack] if
                              ele not in self.node2pos[query_node] and ele not in self.node2edge[
                                  query_node] and ele not in ignore])
            self.pointer += n_lack
            if self.pointer >= len(self.all_edges):
                self.pointer = 0
                random.shuffle(self.all_edges)
        if len(negatives) > negative_size:
            negatives = negatives[:negative_size]

        return negatives

    def _my_get_exactly_k_negatives(self, query_node, negative_size, ignore=[]):
        """ Generate EXACTLY negative_size samples for the query node
        """
        # [旁批] 死代码，没有任何地方调用（搜 "_my_get_exactly_k_negatives" 只有定义）。
        #   它与上面那个版本的区别：要求「两端都不成立」（p 不在父集 且 c 不在子集），更严；
        #   且对 pseudo_leaf 开了例外。依赖 node2pos_node。可以整段跳过。
        if self.pointer == 0:
            random.shuffle(self.all_edges)

        negatives = []
        while len(negatives) != negative_size:
            n_lack = negative_size - len(negatives)
            negatives.extend([ele for ele in self.all_edges[self.pointer: self.pointer + n_lack] if
                              ele[0] not in self.node2pos_node[query_node][0] and (ele[1] not in self.node2pos_node[
                                  query_node][1] or ele[1] is self.pseudo_leaf_node) and ele not in ignore])
            self.pointer += n_lack
            if self.pointer >= len(self.all_edges):
                self.pointer = 0
                random.shuffle(self.all_edges)
        if len(negatives) > negative_size:
            negatives = negatives[:negative_size]

        return negatives


