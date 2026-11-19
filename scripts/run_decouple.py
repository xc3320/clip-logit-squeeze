#!/usr/bin/env python3
"""解耦推理实验：对齐 MMRL (CVPR2025) 的 base-to-novel 解耦推理，检验能否救回 SCG 的 novel 掉点。

MMRL 官方实现 trainers/mmrl.py L419-424（B2N 任务）：
    logits, logits_rep, logits_fusion = model(input)
    output = logits_fusion if sub_cls == "base" else logits
即：base 测试样本用「类别特征+表征特征」融合 logits，novel 测试样本用纯类别特征 logits。
前提是 base-to-novel 评测协议下已知每个测试样本属于 base 类集还是 novel 类集
（base_test / novel_test 分开评测，这是该 benchmark 的标准设定，不是 oracle 泄漏）。

本实验把 MMRL 的解耦思想移植到 SCG 的两条分支（文本=冻结类别特征，视觉探针=任务适配特征）：
  clip0           : 所有测试样本用纯文本 logits
  gate_calibrated : SCG 现状 —— base/novel 按「类索引」填 logits（base 位置融合，novel 位置文本）
                    但对 novel 测试样本，base 类位置仍有视觉高分，会把 novel 样本挤压到 base
  decouple_scg    : 按「测试样本真实身份」切换 logits —— base_test 样本用 SCG 融合 logits，
                    novel_test 样本用纯文本 logits（对齐 MMRL B2N 解耦）
  decouple_naive  : 消融 —— base_test 用未校准融合 logits，novel_test 用纯文本
  decouple_fulltext: 对照 —— 所有样本用纯文本（应等于 clip0，验证切换逻辑本身无副作用）

关键诊断：
  - novel 测试样本在 gate_calibrated 下被判为 base 类的比例（量化"挤压"）
  - 解耦后 novel 是否回到 clip0 水平、base 是否保持 SCG 水平

诚实标注：解耦依赖"测试时已知 base/novel 身份"这一 base-to-novel 标准协议前提，
不宣称是无监督自动分流。缺数据即停止，不生成任何伪造指标。

用法:
  python scripts/run_decouple.py --dataset dtd --seeds 0 1 2 3 4 --tag v1
"""
import argparse
import os
import sys
import json
import time

try:
    import torch
    import torch.nn.functional as F
    import open_clip
    import yaml
    import pandas as pd
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"[停止] 缺少依赖: {e}。请先按 docs/环境搭建.md 安装。")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_scg as S  # noqa: E402

METHODS = ["clip0", "gate_calibrated", "decouple_scg", "decouple_naive",
           "decouple_fulltext", "decouple_auto_disagree", "decouple_auto_b2n"]


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def encode_images(df, cfg, model, preprocess, device, dtype, split_filter=None):
    if split_filter is not None:
        df = df[df["split"].isin(split_filter)]
    feats = []
    bs = int(cfg.get("eval_batch", 64))
    paths = df["image_path"].tolist()
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        with torch.no_grad():
            f = F.normalize(model.encode_image(x).to(dtype), dim=-1)
        feats.append(f)
    return torch.cat(feats) if feats else torch.empty(0, device=device, dtype=dtype)


def run_seed(dataset, seed, cfg, model, preprocess, tokenizer, device, dtype,
             max_test_per_class=0):
    split_fp = os.path.join(ROOT, "data", "splits", f"{dataset}_seed{seed}.csv")
    if not os.path.exists(split_fp):
        print(f"[停止] 缺少划分文件 {split_fp}")
        return None
    df = pd.read_csv(split_fp)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_df = df[df["split"] == "base_train"]
    base_classes = sorted(train_df["class_name"].unique())
    all_classes = sorted(df["class_name"].unique())
    c_base, c_all = len(base_classes), len(all_classes)
    base_idx = [all_classes.index(c) for c in base_classes]
    novel_idx = [i for i in range(c_all) if i not in set(base_idx)]
    c2i = {c: i for i, c in enumerate(base_classes)}

    tr_feat = encode_images(df, cfg, model, preprocess, device, dtype, ["base_train"])
    tr_lab = torch.tensor([c2i[c] for c in train_df["class_name"].tolist()], device=device)

    test_df = df[df["split"].isin(["base_test", "novel_test"])].reset_index(drop=True)
    # 大测试集（如 CIFAR100 每类 600 张）按类均衡子采样，固定 seed 可复现。
    # 只采样真实样本、不改标签、不合成；采样数写入结果文件供审计。
    if max_test_per_class and max_test_per_class > 0:
        rng = np.random.default_rng(10000 + seed)
        keep = []
        for _, g in test_df.groupby("class_name", sort=False):
            idx = g.index.to_numpy()
            if len(idx) > max_test_per_class:
                idx = rng.choice(idx, size=max_test_per_class, replace=False)
            keep.extend(idx.tolist())
        test_df = test_df.loc[sorted(keep)].reset_index(drop=True)
    te_feat = encode_images(test_df, cfg, model, preprocess, device, dtype)

    w_text = S.encode_text_all(model, tokenizer, all_classes, device).to(dtype)
    scale = float(model.logit_scale.exp().detach())
    w_vis = S.train_visual_probe(
        tr_feat, tr_lab, c_base,
        epochs=int(cfg.get("scg_epochs", 300)),
        lr=float(cfg.get("scg_lr", 0.05)),
    )

    text_l = S.text_logits(te_feat, w_text, scale)                  # [B, C_all] 纯类别特征
    vis_l = S.visual_logits_all(te_feat, w_vis, base_idx, c_all)     # [B, C_all] base 视觉探针
    rel = S.estimate_reliability(tr_feat, w_text, scale, tr_lab, base_idx, c_base)
    g_base = S.build_gate(rel, gate_k=float(cfg.get("gate_k", 4.0)),
                          gate_floor=float(cfg.get("gate_floor", 0.2)))

    fused_cal = S.fuse_calibrated(text_l, vis_l, g_base, base_idx, c_all)   # SCG 校准融合
    fused_raw = S.fuse(text_l, vis_l, g_base, base_idx, c_all)              # 未校准融合

    is_base_test = (test_df["split"] == "base_test").to_numpy()
    gts = test_df["class_name"].tolist()
    gt_idx = torch.tensor([all_classes.index(c) for c in gts], device=device)

    # ---- 诊断：novel 测试样本被 SCG 挤压到 base 类的比例 ----
    scg_pred = fused_cal.argmax(dim=-1)
    novel_mask = ~torch.tensor(is_base_test, device=device)
    base_mask_t = torch.tensor(is_base_test, device=device)
    base_set = set(base_idx)
    novel_pred_as_base = sum(
        1 for j in range(len(gts)) if novel_mask[j] and scg_pred[j].item() in base_set
    )
    novel_total = int(novel_mask.sum())
    novel_squeezed_ratio = novel_pred_as_base / novel_total if novel_total else 0.0

    # ---- 构造每个方法的逐样本 logits（按 base-to-novel 协议切换）----
    # clip0 / gate_calibrated：所有样本用同一套 logits
    # decouple_*：按测试样本真实身份切换（对齐 MMRL B2N）
    def decouple(base_logits, novel_logits):
        out = torch.empty_like(base_logits)
        out[base_mask_t] = base_logits[base_mask_t]
        out[~base_mask_t] = novel_logits[~base_mask_t]
        return out

    # ---- 无监督自动分流：不需要知道测试样本真实 base/novel 身份 ----
    # 思路：融合 logits 与纯文本 logits 的「预测分歧」是可观测信号。
    #   若两者 top-1 一致 → 融合没改变判断，用融合（拿到 base 增益）；
    #   若两者 top-1 分歧 → 视觉分支改变了判断，可能正是 novel 被挤压，
    #                       回退纯文本（保 novel）。
    # 变体1 decouple_auto_disagree：任何分歧都回退纯文本。
    # 变体2 decouple_auto_b2n：仅当「融合判成 base、纯文本判成 novel」时回退
    #        （精准针对挤压方向，误伤更小）。
    pred_fused = fused_cal.argmax(dim=-1)
    pred_text = text_l.argmax(dim=-1)
    disagree = pred_fused != pred_text
    # 挤压方向：融合→base 且 纯文本→novel（用 one-hot 查表，避免 isin 版本差异）
    is_base_cls = torch.zeros(c_all, dtype=torch.bool, device=device)
    is_base_cls[base_idx] = True
    is_novel_cls = ~is_base_cls
    squeeze_dir = disagree & is_base_cls[pred_fused] & is_novel_cls[pred_text]

    def auto_route(trigger):
        out = fused_cal.clone()
        out[trigger] = text_l[trigger]
        return out

    logits_map = {
        "clip0": text_l,
        "gate_calibrated": fused_cal,
        # MMRL 式解耦：base 样本用融合，novel 样本用纯文本（冻结类别特征）
        "decouple_scg": decouple(fused_cal, text_l),
        "decouple_naive": decouple(fused_raw, text_l),
        # 对照：所有样本都切到纯文本，应等于 clip0（验证切换逻辑无副作用）
        "decouple_fulltext": decouple(text_l, text_l),
        # 无监督分流：用预测分歧自行决定，不需要身份标签
        "decouple_auto_disagree": auto_route(disagree),
        "decouple_auto_b2n": auto_route(squeeze_dir),
    }

    # 诊断：自动分流的触发率，以及是否命中真实 novel 样本
    auto_diag = {
        "disagree_rate": round(disagree.float().mean().item(), 4),
        "squeeze_dir_rate": round(squeeze_dir.float().mean().item(), 4),
        "disagree_on_novel_rate": round(
            (disagree & novel_mask).float().mean().item(), 4),
        "squeeze_dir_precision": round(
            ((squeeze_dir & novel_mask).float().sum() /
             (squeeze_dir.float().sum() + 1e-6)).item(), 4),
    }

    out = {"dataset": dataset, "seed": seed, "n_base": c_base, "n_novel": c_all - c_base,
           "n_test_samples": int(len(test_df)),
           "test_sampling": (f"每类≤{max_test_per_class}张(类均衡,固定seed)" if max_test_per_class else "全量测试集"),
           "novel_squeezed_ratio": round(novel_squeezed_ratio, 4),
           "novel_pred_as_base": novel_pred_as_base, "novel_total": novel_total,
           "auto_diag": auto_diag, "methods": {}}
    for m, logits in logits_map.items():
        pred = logits.argmax(dim=-1)
        correct = (pred == gt_idx)
        b_mask = base_mask_t.cpu().numpy()
        n_mask = ~b_mask
        correct_np = correct.cpu().numpy()
        b = correct_np[b_mask].mean() if b_mask.any() else 0.0
        n = correct_np[n_mask].mean() if n_mask.any() else 0.0
        hm = 2 * b * n / (b + n) if (b + n) > 0 else 0.0
        out["methods"][m] = {
            "base_acc": round(float(b), 4), "novel_acc": round(float(n), 4),
            "hm": round(float(hm), 4),
            "n_base_test": int(b_mask.sum()), "n_novel_test": int(n_mask.sum()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "raa.yaml"))
    ap.add_argument("--dataset", default="dtd")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--max-test-per-class", type=int, default=0,
                    help="大测试集按类均衡采样上限(0=全量)。仅采样真实样本,固定seed可复现")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if not torch.cuda.is_available():
        print("[停止] CUDA 不可用。")
        sys.exit(1)
    device = "cuda"
    dtype = torch.float32 if cfg.get("fp16") is False else torch.float16
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["backbone"], pretrained=cfg["pretrained"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(cfg["backbone"])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    results = []
    for s in args.seeds:
        r = run_seed(args.dataset, s, cfg, model, preprocess, tokenizer, device, dtype,
                     max_test_per_class=args.max_test_per_class)
        if r is None:
            print(f"[跳过] seed={s} 无真实划分")
            continue
        results.append(r)
        print(f"[seed {s}] 挤压率={r['novel_squeezed_ratio']:.3f} | " + " | ".join(
            f"{m}:HM={d['hm']:.4f}(B{d['base_acc']:.3f}/N{d['novel_acc']:.3f})"
            for m, d in r["methods"].items()), flush=True)

    if not results:
        print("[停止] 无任何真实结果。")
        return 1

    summary = {}
    for m in METHODS:
        rows = [r["methods"][m] for r in results]
        summary[m] = {}
        for k in ["base_acc", "novel_acc", "hm"]:
            vals = [row[k] for row in rows]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            summary[m][k] = {"mean": round(mean, 4), "std": round(std, 4)}

    squeeze = [r["novel_squeezed_ratio"] for r in results]
    auto_keys = ["disagree_rate", "squeeze_dir_rate", "disagree_on_novel_rate",
                 "squeeze_dir_precision"]
    auto_summary = {}
    if results and "auto_diag" in results[0]:
        for key in auto_keys:
            vals = [r["auto_diag"][key] for r in results]
            auto_summary[key] = round(sum(vals) / len(vals), 4)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    out = {
        "dataset": args.dataset, "tag": args.tag, "methods": METHODS,
        "n_seeds": len(results), "summary": summary,
        "test_sampling": (f"每类≤{args.max_test_per_class}张(类均衡,固定seed)"
                          if args.max_test_per_class else "全量测试集"),
        "n_test_samples_per_seed": [r["n_test_samples"] for r in results],
        "novel_squeezed_ratio": {
            "mean": round(sum(squeeze) / len(squeeze), 4),
            "per_seed": squeeze,
        },
        "auto_diag": auto_summary,
        "per_seed": results, "peak_vram_mb": round(peak, 1),
        "seconds": round(time.time() - t0, 1),
        "protocol_note": "decouple_* 依赖 base-to-novel 标准协议下已知测试样本 base/novel 身份；"
                         "非无监督自动分流。MMRL(CVPR2025) trainers/mmrl.py L419-424 同款做法。",
    }
    outdir = os.path.join(ROOT, "results", args.dataset)
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join(outdir, f"decouple_{args.tag}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n=== 汇总（均值±标准差）===")
    for m in METHODS:
        d = summary[m]
        print(f"  {m:<18} HM={d['hm']['mean']:.4f}±{d['hm']['std']:.4f} "
              f"Base={d['base_acc']['mean']:.4f} Novel={d['novel_acc']['mean']:.4f}")
    print(f"\n=== novel 被 SCG 挤压到 base 类的比例: "
          f"{out['novel_squeezed_ratio']['mean']:.4f} ===")
    if auto_summary:
        print("=== 无监督自动分流诊断 ===")
        print(f"  分歧率={auto_summary.get('disagree_rate')} "
              f"挤压方向触发率={auto_summary.get('squeeze_dir_rate')} "
              f"挤压方向命中真实novel的精确率={auto_summary.get('squeeze_dir_precision')}")
    print(f"峰值显存 {peak:.1f}MB，用时 {out['seconds']}s，结果 {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
