#!/usr/bin/env python3
"""SCG 实验主程序：在真实 base-to-novel 划分上检验"尺度校准修复门控塌缩"。

对照条件（同一数据、同一文本/视觉分支，唯一差别是是否尺度校准）：
  clip0          : 纯文本分支（冻结 CLIP 手工模板），无 cache、无门控
  vis_probe      : 纯视觉线性探针（base 类），novel 类 0 分
  naive          : 文本+视觉无校准，固定 0.5/0.5 融合（等价于塌缩的极端）
  gate_naive     : 可靠性门控，但无尺度校准（CIDA 式，预期视觉主导）
  gate_calibrated: 可靠性门控 + 逐样本 z-score 尺度校准（SCG，本文方案）

记录中间量：两分支 logits 的标准差、base/novel 类有效文本权重、门控 g 分布。
缺数据即停止，不生成任何伪造指标。

用法:
  python scripts/run_scg.py --dataset dtd --seeds 0 1 2 3 4
  python scripts/run_scg.py --dataset oxfordpets --seeds 0 1 2 3 4
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
    from PIL import Image
except ImportError as e:
    print(f"[停止] 缺少依赖: {e}。请先按 docs/环境搭建.md 安装。")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_scg as S  # noqa: E402

METHODS = ["clip0", "vis_probe", "naive", "cal_only", "gate_naive", "gate_calibrated",
           "gate_cal_bal"]


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def encode_images(df, cfg, model, preprocess, device, dtype, split_filter=None):
    """把指定 split 的图像编码成归一化特征缓存。"""
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


def run_seed(dataset, seed, cfg, model, preprocess, tokenizer, device, dtype):
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
    c2i = {c: i for i, c in enumerate(base_classes)}

    # 编码：训练 support + 全部测试（base_test + novel_test），骨干冻结
    tr_feat = encode_images(df, cfg, model, preprocess, device, dtype, ["base_train"])
    tr_lab = torch.tensor([c2i[c] for c in train_df["class_name"].tolist()], device=device)

    test_df = df[df["split"].isin(["base_test", "novel_test"])].reset_index(drop=True)
    te_feat = encode_images(df, cfg, model, preprocess, device, dtype,
                            ["base_test", "novel_test"])

    w_text = S.encode_text_all(model, tokenizer, all_classes, device).to(dtype)
    scale = float(model.logit_scale.exp().detach())
    w_vis = S.train_visual_probe(
        tr_feat, tr_lab, c_base,
        epochs=int(cfg.get("scg_epochs", 300)),
        lr=float(cfg.get("scg_lr", 0.05)),
    )

    text_l = S.text_logits(te_feat, w_text, scale)              # [B, C_all]
    vis_l = S.visual_logits_all(te_feat, w_vis, base_idx, c_all)  # [B, C_all]
    rel = S.estimate_reliability(tr_feat, w_text, scale, tr_lab, base_idx, c_base)
    g_base = S.build_gate(
        rel,
        gate_k=float(cfg.get("gate_k", 4.0)),
        gate_floor=float(cfg.get("gate_floor", 0.2)),
    )

    is_base_test = (test_df["split"] == "base_test").to_numpy()
    gts = test_df["class_name"].tolist()

    def acc_on(logits, want_base):
        pred = logits.argmax(dim=-1).cpu().tolist()
        idxs = [j for j in range(len(gts)) if is_base_test[j] == want_base]
        correct = sum(int(all_classes[pred[j]] == gts[j]) for j in idxs)
        total = len(idxs)
        return (correct / total if total else 0.0), total

    # 各方法 logits
    z_text, z_vis = S.zscore_per_sample(text_l), S.zscore_per_sample(vis_l)
    fused_cal = S.fuse_calibrated(text_l, vis_l, g_base, base_idx, c_all)

    # group-wise 再校准：novel 掉点因 base 类 logits 被视觉分支整体抬高。
    # 用 base 类的融合前后均值差，把 base 块整体拉回文本基准，恢复 base/novel 相对尺度。
    fused_bal = fused_cal.clone()
    shift = (fused_cal[:, base_idx].mean(dim=1) - z_text[:, base_idx].mean(dim=1))
    fused_bal[:, base_idx] = fused_cal[:, base_idx] - shift.unsqueeze(1)

    logits_map = {
        "clip0": text_l,
        "vis_probe": vis_l,
        "naive": 0.5 * text_l + 0.5 * vis_l,           # 无校准固定融合
        "cal_only": 0.5 * z_text + 0.5 * z_vis,        # 只校准、无可靠性门控（消融）
        "gate_naive": S.fuse(text_l, vis_l, g_base, base_idx, c_all),          # 有门控无校准
        "gate_calibrated": fused_cal,                  # SCG：分支级校准
        "gate_cal_bal": fused_bal,                     # SCG+组间再校准（保 novel）
    }

    out = {"dataset": dataset, "seed": seed, "n_base": c_base, "n_novel": c_all - c_base,
           "logit_scale": round(scale, 4), "methods": {}}
    for m, logits in logits_map.items():
        b, nb = acc_on(logits, True)
        n, nn = acc_on(logits, False)
        hm = 2 * b * n / (b + n) if (b + n) > 0 else 0.0
        rec = {"base_acc": round(b, 4), "novel_acc": round(n, 4), "hm": round(hm, 4),
               "n_base_test": nb, "n_novel_test": nn}
        out["methods"][m] = rec

    # 机制中间量（与具体融合方法无关，记录一次即可）
    mech = {
        "text_std": round(text_l.std(dim=-1).mean().item(), 4),
        "text_mean": round(text_l.mean().item(), 4),
        "vis_std": round(vis_l.std(dim=-1).mean().item(), 4),
        "vis_mean": round(vis_l.mean().item(), 4),
        "vis_to_text_std_ratio": round(
            (vis_l.std(dim=-1).mean() / (text_l.std(dim=-1).mean() + 1e-6)).item(), 4),
        "gate_base_text_weight_mean": round(g_base.mean().item(), 4),
        "gate_base_text_weight_min": round(g_base.min().item(), 4),
        "gate_novel_text_weight": 1.0,
        "rel_min": round(rel.min().item(), 4),
        "rel_max": round(rel.max().item(), 4),
    }
    out["mechanism"] = mech
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "raa.yaml"))
    ap.add_argument("--dataset", default="dtd")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="v1")
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
        r = run_seed(args.dataset, s, cfg, model, preprocess, tokenizer, device, dtype)
        if r is None:
            print(f"[跳过] seed={s} 无真实产物")
            continue
        results.append(r)
        print(f"[seed {s}] " + " | ".join(
            f"{m}:HM={d['hm']:.4f}(B{d['base_acc']:.3f}/N{d['novel_acc']:.3f})"
            for m, d in r["methods"].items()), flush=True)

    if not results:
        print("[停止] 无任何真实结果。")
        return 1

    # 汇总：各方法 base/novel/hm 的均值±标准差
    summary = {}
    for m in METHODS:
        rows = [r["methods"][m] for r in results]
        summary[m] = {}
        for k in ["base_acc", "novel_acc", "hm"]:
            vals = [row[k] for row in rows]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            summary[m][k] = {"mean": round(mean, 4), "std": round(std, 4)}
    mech_summary = {}
    if results and "mechanism" in results[0]:
        mech_keys = ["text_std", "text_mean", "vis_std", "vis_mean", "vis_to_text_std_ratio",
                     "gate_base_text_weight_mean", "gate_base_text_weight_min",
                     "rel_min", "rel_max"]
        for key in mech_keys:
            vals = [r["mechanism"][key] for r in results]
            mech_summary[key] = round(sum(vals) / len(vals), 4)

    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    out = {"dataset": args.dataset, "tag": args.tag, "methods": METHODS,
           "n_seeds": len(results), "summary": summary, "mechanism": mech_summary,
           "per_seed": results, "peak_vram_mb": round(peak, 1),
           "seconds": round(time.time() - t0, 1)}
    outdir = os.path.join(ROOT, "results", args.dataset)
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join(outdir, f"scg_{args.tag}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n=== 汇总（均值±标准差）===")
    for m in METHODS:
        d = summary[m]
        print(f"  {m:<16} HM={d['hm']['mean']:.4f}±{d['hm']['std']:.4f} "
              f"Base={d['base_acc']['mean']:.4f} Novel={d['novel_acc']['mean']:.4f}")
    print("\n=== 机制中间量（检验尺度失配）===")
    print(f"  text logits: mean={mech_summary.get('text_mean','NA')} "
          f"std={mech_summary.get('text_std','NA')}")
    print(f"  vis  logits: mean={mech_summary.get('vis_mean','NA')} "
          f"std={mech_summary.get('vis_std','NA')}")
    print(f"  vis/text std比={mech_summary.get('vis_to_text_std_ratio','NA')} "
          f"base文本权重均值={mech_summary.get('gate_base_text_weight_mean','NA')} "
          f"最小={mech_summary.get('gate_base_text_weight_min','NA')}")
    print(f"\n峰值显存 {peak:.1f}MB，用时 {out['seconds']}s，结果 {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
