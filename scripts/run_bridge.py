#!/usr/bin/env python3
"""Text→Visual Bridge 实验：用 base 类文本/视觉原型配对学映射，给 novel 造伪视觉原型，
修复 SCG 中 novel 视觉分支填 0 导致的结构性掉点。

对照方法（同一数据、同一划分）：
  clip0             : 纯文本（冻结 CLIP 手工模板）
  gate_calibrated   : 原 SCG（视觉=base 线性探针，novel 填 0，z-score 校准+门控）—— 配对基线
  bridge_proto      : 纯桥接视觉原型（base 真实 + novel 伪），无文本
  bridge_avg_raw    : 文本/桥接视觉 固定 0.5 融合，无 z-score（两边都是 cosine，尺度天然一致）
  bridge_gate_raw   : 文本/桥接视觉 可靠性门控，无 z-score
  bridge_gate_cal   : 文本/桥接视觉 可靠性门控 + z-score 校准

novel 门控：不再强制 g=1（纯文本），而用 base 平均可靠性推出的全局先验，
使 novel 也能吃到视觉证据。

诊断（可证伪核心）：留一类交叉验证的 text→visual 映射余弦。若低，判桥接失败。
缺数据即停止，不生成任何伪造指标。

用法:
  python scripts/run_bridge.py --dataset dtd --seeds 0
  python scripts/run_bridge.py --dataset dtd --seeds 0 1 2 3 4 --tag v1
  python scripts/run_bridge.py --dataset oxfordpets --seeds 0 1 2 3 4 --tag v1
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
import models_scg as S      # noqa: E402
import models_bridge as B   # noqa: E402

METHODS = ["clip0", "gate_calibrated", "bridge_proto", "bridge_avg_raw",
           "bridge_gate_raw", "bridge_gate_cal",
           "bridge_proto_center", "bridge_avg_center", "bridge_gate_center"]


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
    novel_idx = [i for i in range(c_all) if i not in set(base_idx)]
    c2i = {c: i for i, c in enumerate(base_classes)}

    tr_feat = encode_images(df, cfg, model, preprocess, device, dtype, ["base_train"])
    tr_lab = torch.tensor([c2i[c] for c in train_df["class_name"].tolist()], device=device)
    test_df = df[df["split"].isin(["base_test", "novel_test"])].reset_index(drop=True)
    te_feat = encode_images(df, cfg, model, preprocess, device, dtype,
                            ["base_test", "novel_test"])

    w_text = S.encode_text_all(model, tokenizer, all_classes, device).to(dtype)
    scale = float(model.logit_scale.exp().detach())
    text_base = w_text[base_idx]                                  # [C_base, D]

    # ---- base 真实视觉原型 + 桥接诊断（可证伪核心）----
    vis_base = B.class_vis_prototypes(tr_feat, tr_lab, c_base).to(dtype)
    ridge = float(cfg.get("bridge_ridge", 1e3))
    diag = B.bridge_diagnostics(text_base, vis_base, ridge=ridge)
    # 生成全部类的视觉原型：base 真实 + novel 桥接伪
    vis_proto, W = B.build_pseudo_visual(
        w_text, text_base, vis_base, base_idx, novel_idx, c_all, ridge=ridge)
    # 中心化变体：novel 伪原型减去 base 伪原型均值方向，恢复区分度
    vis_proto_c, _ = B.build_pseudo_visual(
        w_text, text_base, vis_base, base_idx, novel_idx, c_all, ridge=ridge,
        center_pseudo=True, W=W.clone())
    # 诊断 novel 伪原型区分度（越低越好）
    div_raw = B.pseudo_diversity(text_base, vis_base, w_text[novel_idx], ridge=ridge)
    div_cen = B.pseudo_diversity(text_base, vis_base, w_text[novel_idx], ridge=ridge,
                                 center_pseudo=True)

    # 两条分支的 logits（都是 scale * cos，天然同量纲）
    text_l = S.text_logits(te_feat, w_text, scale)                # [B, C_all]
    vis_l = (te_feat @ vis_proto.T) * scale                       # [B, C_all] 桥接视觉
    vis_l_c = (te_feat @ vis_proto_c.T) * scale                   # 中心化桥接视觉

    # ---- 原 SCG 基线：base 线性探针，novel 填 0 ----
    w_vis = S.train_visual_probe(
        tr_feat, tr_lab, c_base,
        epochs=int(cfg.get("scg_epochs", 300)),
        lr=float(cfg.get("scg_lr", 0.05)),
    )
    vis_l_probe = S.visual_logits_all(te_feat, w_vis, base_idx, c_all)

    # ---- 可靠性门控 ----
    rel = S.estimate_reliability(tr_feat, w_text, scale, tr_lab, base_idx, c_base)
    gate_k = float(cfg.get("gate_k", 4.0))
    gate_floor = float(cfg.get("gate_floor", 0.2))
    g_base = S.build_gate(rel, gate_k=gate_k, gate_floor=gate_floor)  # [C_base]
    # novel 用 base 平均可靠性推出的全局先验（不再强制纯文本）
    g_novel = float((gate_floor + (1 - gate_floor) *
                     torch.sigmoid(torch.tensor(gate_k * (rel.mean().item() - 0.5)))).item())

    def gate_fuse(tl, vl, gv):
        """门控融合：g 是文本权重。gv [C_all] 逐类。"""
        return gv.unsqueeze(0) * tl + (1 - gv.unsqueeze(0)) * vl

    g_all = torch.full((c_all,), g_novel, device=text_l.device, dtype=text_l.dtype)
    g_all[base_idx] = g_base.to(dtype=text_l.dtype)

    z_text, z_vis = S.zscore_per_sample(text_l), S.zscore_per_sample(vis_l)
    z_vis_c = S.zscore_per_sample(vis_l_c)

    logits_map = {
        "clip0": text_l,
        # 原 SCG：novel 视觉=0，novel 门控强制纯文本
        "gate_calibrated": S.fuse_calibrated(text_l, vis_l_probe, g_base, base_idx, c_all),
        "bridge_proto": vis_l,
        "bridge_avg_raw": 0.5 * text_l + 0.5 * vis_l,
        "bridge_gate_raw": gate_fuse(text_l, vis_l, g_all),
        "bridge_gate_cal": gate_fuse(z_text, z_vis, g_all),
        # 中心化变体
        "bridge_proto_center": vis_l_c,
        "bridge_avg_center": 0.5 * z_text + 0.5 * z_vis_c,
        "bridge_gate_center": gate_fuse(z_text, z_vis_c, g_all),
    }

    is_base_test = (test_df["split"] == "base_test").to_numpy()
    gts = test_df["class_name"].tolist()

    def acc_on(logits, want_base):
        pred = logits.argmax(dim=-1).cpu().tolist()
        idxs = [j for j in range(len(gts)) if is_base_test[j] == want_base]
        correct = sum(int(all_classes[pred[j]] == gts[j]) for j in idxs)
        total = len(idxs)
        return (correct / total if total else 0.0), total

    out = {"dataset": dataset, "seed": seed, "n_base": c_base, "n_novel": c_all - c_base,
           "bridge_diag": diag, "pseudo_diversity": {"raw": div_raw, "center": div_cen},
           "g_novel": round(g_novel, 4),
           "g_base_mean": round(g_base.mean().item(), 4), "methods": {}}
    for m, logits in logits_map.items():
        b, nb = acc_on(logits, True)
        n, nn = acc_on(logits, False)
        hm = 2 * b * n / (b + n) if (b + n) > 0 else 0.0
        out["methods"][m] = {"base_acc": round(b, 4), "novel_acc": round(n, 4),
                             "hm": round(hm, 4), "n_base_test": nb, "n_novel_test": nn}
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
        d = r["bridge_diag"]
        print(f"[seed {s}] bridge留一余弦={d['cos_leave_one_class_out']} "
              f"in-sample={d['cos_insample']} raw={d['cos_raw_no_mapping']} "
              f"泛化差={d['gap_in_minus_loo']} g_novel={r['g_novel']}", flush=True)
        print("  " + " | ".join(
            f"{m}:HM={v['hm']:.4f}(B{v['base_acc']:.3f}/N{v['novel_acc']:.3f})"
            for m, v in r["methods"].items()), flush=True)

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

    diag_summary = {}
    for key in ["cos_insample", "cos_leave_one_class_out", "cos_raw_no_mapping",
                "gap_in_minus_loo"]:
        vals = [r["bridge_diag"][key] for r in results]
        diag_summary[key] = round(sum(vals) / len(vals), 4)
    div_summary = {}
    if results and "pseudo_diversity" in results[0]:
        for variant in ["raw", "center"]:
            key = "novel_pseudo_offdiag"
            vals = [r["pseudo_diversity"][variant].get(key) for r in results
                    if key in r["pseudo_diversity"][variant]]
            if vals:
                div_summary[f"{variant}_{key}"] = round(sum(vals) / len(vals), 4)

    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    out = {"dataset": args.dataset, "tag": args.tag, "method": "Text2VisualBridge",
           "methods": METHODS, "n_seeds": len(results), "summary": summary,
           "bridge_diag": diag_summary, "pseudo_diversity": div_summary,
           "per_seed": results, "peak_vram_mb": round(peak, 1),
           "seconds": round(time.time() - t0, 1)}
    outdir = os.path.join(ROOT, "results", args.dataset)
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join(outdir, f"bridge_{args.tag}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n=== 汇总（均值±标准差）===")
    for m in METHODS:
        d = summary[m]
        print(f"  {m:<18} HM={d['hm']['mean']:.4f}±{d['hm']['std']:.4f} "
              f"Base={d['base_acc']['mean']:.4f} Novel={d['novel_acc']['mean']:.4f}")
    print("\n=== 桥接诊断（可证伪核心）===")
    print(f"  留一类余弦={diag_summary['cos_leave_one_class_out']} "
          f"in-sample={diag_summary['cos_insample']} "
          f"无映射raw={diag_summary['cos_raw_no_mapping']} "
          f"记忆差(in-loo)={diag_summary['gap_in_minus_loo']}")
    print(f"  novel伪原型类间余弦(越低越好): raw={div_summary.get('raw_novel_pseudo_offdiag','NA')} "
          f"center={div_summary.get('center_novel_pseudo_offdiag','NA')}")
    print(f"\n峰值显存 {peak:.1f}MB，用时 {out['seconds']}s，结果 {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
