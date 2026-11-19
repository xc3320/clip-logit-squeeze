#!/usr/bin/env python3
"""RAA 训练主循环：base-to-novel，多种子，显存记录。

依赖 torch / open_clip_torch；缺失时明确停止，不生成任何虚假指标。

CLI 用法（单方法）:
  python scripts/train.py --config configs/raa.yaml --method raa --seed 0 --dataset dtd
  python scripts/train.py --config configs/raa.yaml --method raa_rev --seed 0 --dataset dtd

批量用法（由 train_batch.py / run_full.py 调用，模型只加载一次）:
  from train import run_one

method 列表:
  clip0, coop, kgcoop,
  raa（可靠类强锚定/acc）, raa_rev（不可靠类强锚定/acc）,
  raa_margin（可靠类强锚定/连续边距）, raa_margin_rev（不可靠类强锚定/连续边距）,
  raa_anneal（acc 自适应强度从 1 衰减到 0，末轮退化为 KgCoOp）,
  raa_margin_anneal（连续边距版本）

这些变体只改损失里的逐类权重来源/方向/退火，不改 prompt 结构与骨干。
"""
import argparse
import os
import sys
import time
import json

try:
    import torch
    import torch.nn.functional as F
    import open_clip
    import yaml
except ImportError as e:
    print("[停止] 缺少训练依赖（torch / open_clip_torch / pyyaml）。")
    print("       请先按 docs/环境搭建.md 安装，再运行本脚本。")
    print(f"       原始错误: {e}")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_raa as M  # noqa: E402

# 变体开关：全部保持 mean 聚合与 λ 均值为 center，避免改聚合口径造成假差异。
# rel_mode=None 表示沿用配置里的 rel_mode；anneal='decay' 表示自适应偏差逐轮衰减到 0。
VARIANTS = {
    "raa":                {"rel_mode": None,     "reverse": False, "anneal": "none"},
    "raa_rev":            {"rel_mode": None,     "reverse": True,  "anneal": "none"},
    "raa_margin":         {"rel_mode": "margin", "reverse": False, "anneal": "none"},
    "raa_margin_rev":     {"rel_mode": "margin", "reverse": True,  "anneal": "none"},
    "raa_anneal":         {"rel_mode": None,     "reverse": False, "anneal": "decay"},
    "raa_margin_anneal":  {"rel_mode": "margin", "reverse": False, "anneal": "decay"},
}
RAA_METHODS = set(VARIANTS)
ALL_METHODS = ["clip0", "coop", "kgcoop"] + list(VARIANTS)


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_split(dataset, seed):
    fp = os.path.join(ROOT, "data", "splits", f"{dataset}_seed{seed}.csv")
    if not os.path.exists(fp):
        print(f"[停止] 找不到划分文件 {fp}。请先运行 build_split.py 并放入真实数据。")
        sys.exit(1)
    import pandas as pd
    return pd.read_csv(fp)


def run_one(method, dataset, seed, cfg, model, preprocess, device, dtype,
            tag="", kg_lam=None, spread=None, rel_mode=None, margin_scale=None,
            anneal=None, epochs=None, lr=None):
    """训练一个 (方法, 数据集, 种子) 组合；模型由调用方加载，可重复复用。

    旧结果未显式固定 prompt 初始化随机种子；新实验用 seed 固定，保证可复现。
    用 tag 写独立文件，不覆盖旧结果，避免两套协议混在同一张表里。
    """
    suffix = f"_{tag}" if tag else ""
    epochs = cfg["epochs"] if epochs is None else epochs
    lr = cfg["lr"] if lr is None else lr
    kg_lam = cfg["kg_lam"] if kg_lam is None else kg_lam
    spread = cfg.get("raa_spread", 6.0) if spread is None else spread
    margin_scale = cfg.get("margin_scale", 10.0) if margin_scale is None else margin_scale
    variant = VARIANTS.get(method, {"rel_mode": None, "reverse": False, "anneal": "none"})
    rel_mode = rel_mode or variant["rel_mode"] or cfg.get("rel_mode", "acc")
    anneal = anneal or variant["anneal"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = open_clip.get_tokenizer(cfg["backbone"])
    M.set_tokenizer(tokenizer)
    M.tokenize = lambda prompts: tokenizer(prompts).to(device)

    df = read_split(dataset, seed)
    train_df = df[df["split"] == "base_train"]
    base_classes = sorted(train_df["class_name"].unique())
    class_to_idx = {c: i for i, c in enumerate(base_classes)}
    print(f"[{method}|{dataset}|seed{seed}] base类={len(base_classes)} "
          f"训练样本={len(train_df)} 骨干={cfg['backbone']}", flush=True)

    if method == "clip0":
        return {"method": method, "dataset": dataset, "seed": seed, "tag": tag,
                "skipped": True}

    # 冻结图像特征（一次性前向，避免每轮重复，节省显存与时间）
    from PIL import Image
    feats, labels = [], []
    for _, r in train_df.iterrows():
        img = Image.open(r["image_path"]).convert("RGB")
        with torch.no_grad():
            f = model.encode_image(preprocess(img).unsqueeze(0).to(device))
            f = F.normalize(f, dim=-1)
        feats.append(f)
        labels.append(class_to_idx[r["class_name"]])
    image_features = torch.cat(feats).to(dtype)          # [N, D]
    labels_t = torch.tensor(labels, device=device)
    anchor = M.clip_anchor_features(model, base_classes, dtype, device)  # W_clip [C, D]

    prompt = M.CoOpPrompt(model, base_classes, n_ctx=cfg["n_ctx"]).to(device).to(dtype)
    prompt.ctx.requires_grad_(True)
    optim = torch.optim.SGD([prompt.ctx], lr=lr, momentum=0.9)

    rel_stats = None
    if method in RAA_METHODS:
        rel, lam_i = M.estimate_reliability(
            image_features.float(), anchor.float(), labels_t,
            center=kg_lam, spread=spread, mode=rel_mode,
            reverse=variant["reverse"], margin_scale=margin_scale
        )
        rel_stats = {
            "rel_mode": rel_mode, "reverse": variant["reverse"], "anneal": anneal,
            "kg_lam": kg_lam, "spread": spread, "margin_scale": margin_scale,
            "rel_min": round(rel.min().item(), 4), "rel_max": round(rel.max().item(), 4),
            "rel_std": round(rel.std().item(), 4),
            "lam_min": round(lam_i.min().item(), 4), "lam_max": round(lam_i.max().item(), 4),
            "lam_mean": round(lam_i.mean().item(), 4),
        }
        direction = "反向(不可靠类强锚定)" if variant["reverse"] else "正向(可靠类强锚定)"
        print(f"[{method}] 可靠性信号={rel_mode} 方向={direction} anneal={anneal}")
        print(f"[RAA] 可靠性 rel_i: min={rel.min().item():.3f} max={rel.max().item():.3f} "
              f"std={rel.std().item():.3f}")
        print(f"[RAA] 逐类锚定权重 λ_i: min={lam_i.min().item():.3f} "
              f"max={lam_i.max().item():.3f} mean={lam_i.mean().item():.3f} "
              f"(mean应≈kg_lam={kg_lam}, 保证与KgCoOp公平对照)")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(epochs):
        optim.zero_grad()
        w_coop = prompt.forward_text_features()          # [C, D]
        logits = image_features @ w_coop.T * model.logit_scale.exp()
        loss_ce = F.cross_entropy(logits.float(), labels_t)
        if method == "coop":
            loss = loss_ce
        elif method == "kgcoop":
            loss = loss_ce + kg_lam * M.kgcoop_loss(w_coop, anchor, 1.0)
        else:  # 各类 RAA 变体
            if anneal == "decay":
                a = 1.0 if epochs == 1 else 1.0 - ep / (epochs - 1)
            elif anneal == "grow":
                a = 0.0 if epochs == 1 else ep / (epochs - 1)
            else:
                a = 1.0
            lam_ep = kg_lam + (lam_i - kg_lam) * a
            loss = loss_ce + M.raa_loss(w_coop, anchor, lam_ep.to(dtype))
        loss.backward()
        optim.step()
        if (ep + 1) % cfg.get("log_every", 50) == 0:
            print(f"  ep{ep+1:4d} loss={loss.item():.4f} ce={loss_ce.item():.4f}")

    peak = torch.cuda.max_memory_allocated() / 1024**2
    elapsed = time.time() - t0
    outdir = os.path.join(ROOT, "results", dataset)
    os.makedirs(outdir, exist_ok=True)
    torch.save(prompt.ctx.detach().cpu(),
               os.path.join(outdir, f"ctx_{method}{suffix}_seed{seed}.pt"))
    stats = {
        "method": method, "dataset": dataset, "seed": seed, "tag": tag,
        "epochs": epochs, "lr": lr, "kg_lam": kg_lam,
        "peak_vram_mb": round(peak, 1),
        "seconds": round(elapsed, 1), "vram_ok": peak <= cfg.get("vram_limit_mb", 12288),
    }
    if rel_stats is not None:
        stats["reliability"] = rel_stats
    with open(os.path.join(outdir, f"train_{method}{suffix}_seed{seed}.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[完成] {stats}", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "raa.yaml"))
    ap.add_argument("--method", choices=ALL_METHODS, default="raa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="dtd")
    ap.add_argument("--kg-lam", type=float, default=None)
    ap.add_argument("--spread", type=float, default=None)
    ap.add_argument("--rel-mode", choices=["acc", "conf", "margin"], default=None)
    ap.add_argument("--margin-scale", type=float, default=None)
    ap.add_argument("--anneal", choices=["none", "decay", "grow"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tag", default="", help="结果文件后缀；留空兼容旧结果")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if not torch.cuda.is_available():
        print("[停止] CUDA 不可用。本方案面向 RTX 3060 单卡，请检查环境。")
        sys.exit(1)
    device = "cuda"
    dtype = torch.float32 if cfg.get("fp16") is False else torch.float16
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["backbone"], pretrained=cfg["pretrained"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    run_one(args.method, args.dataset, args.seed, cfg, model, preprocess, device, dtype,
            tag=args.tag, kg_lam=args.kg_lam, spread=args.spread, rel_mode=args.rel_mode,
            margin_scale=args.margin_scale, anneal=args.anneal,
            epochs=args.epochs, lr=args.lr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
