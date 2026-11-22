"""
校验 fast_scores（只对 MASK 位跑 MLM head）与项目原版 scorer（BertForMaskedLM 全位置）
在**同一设备同一精度**下是否数学等价。

本脚本可纯 CPU 跑（此时两边都是 fp32，可排除 fp16 形状效应）：
    python analysis/check_equiv.py -c config_files/mesh/config.test.hidden.json \
        -m data/saved/mesh/TacoPromptHidden/1108_153108/models/trial1/model_best.pth -n 32

GPU 上两个路径的差异完全来自 fp16 matmul 的形状/累加顺序，本脚本会同时报告
"差值的数量级是否与 score 自身的量级相称"。
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.cuda.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "analysis"))

import data_loader.data_loaders as module_data  # noqa: E402
import model.model as module_arch  # noqa: E402
from dump_hidden_rank import fast_scores  # noqa: E402


def build(cfg_path, model_path, device):
    with open(cfg_path) as f:
        cfg = json.load(f)
    dl = dict(cfg["train_data_loader"]["args"])
    name = dl.pop("taxonomy_name")
    loader = getattr(module_data, cfg["train_data_loader"]["type"])(
        data_path=cfg["data_path"], taxonomy_name=name, **dl)
    ds = loader.dataset
    model = getattr(module_arch, cfg["arch"]["type"])(ds, **cfg["arch"]["args"])
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.to(device).eval()
    return cfg, ds, model


def full_scores(model, p_idx, q_idx, c_idx):
    """逐行照抄 model/model.py: TacoPromptHidden.scorer 的计算，但允许指定设备。"""
    dev = next(model.parameters()).device
    input_ids, _, attn = model.get_lm_inputs(
        p_idx, q_idx, c_idx,
        torch.ones(p_idx.shape[0], dtype=torch.long),
        torch.ones(p_idx.shape[0], dtype=torch.long),
        torch.ones(p_idx.shape[0], dtype=torch.long))
    with torch.no_grad(), autocast(enabled=(dev.type == "cuda")):
        outputs = model.model(input_ids=input_ids.to(dev), attention_mask=attn.to(dev))
        score = outputs[0][input_ids.to(dev) == model.mask_token_id]
        pred = score[:, model.yes_token_id] - score[:, model.no_token_id]
        pred = pred.view(p_idx.shape[0], -1)
    return pred[:, 0].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("-n", "--n_cand", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}  (autocast {'on' if device.type == 'cuda' else 'off'})")
    cfg, ds, model = build(args.config, args.model_path, device)
    model.device = str(device)

    cands = ds.all_edges[:args.n_cand]
    query = ds.test_node_list[0]
    q_allemb = model.taxon2id[query]
    p_idx = torch.tensor([model.train_taxon2allemb_id[p] for p, _ in cands])
    c_idx = torch.tensor([model.train_taxon2allemb_id[c] for _, c in cands])
    q_idx = torch.full((len(cands),), q_allemb, dtype=torch.long)

    a = full_scores(model, p_idx, q_idx, c_idx).cpu()
    b = fast_scores(model, p_idx, q_idx, c_idx).cpu()
    d = (a - b).abs()
    print(f"n_cand={len(cands)}  |score| max={a.abs().max():.4f}  "
          f"score std={a.std():.4f}")
    print(f"max|full-fast|={d.max():.3e}  mean|diff|={d.mean():.3e}")
    print("first 6 (full, fast):")
    for i in range(min(6, len(a))):
        print(f"  {a[i]:+.6f}  {b[i]:+.6f}  d={d[i]:.3e}")
    # 排序是否一致
    ra = torch.argsort(-a)
    rb = torch.argsort(-b)
    print(f"top-{min(10, len(a))} 顺序是否完全一致: {torch.equal(ra[:10], rb[:10])}")


if __name__ == "__main__":
    main()
