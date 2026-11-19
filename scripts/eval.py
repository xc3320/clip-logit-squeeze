#!/usr/bin/env python3
"""评估：Base Acc / Novel Acc / HM，并汇总多种子的均值±标准差。

依赖 torch / open_clip_torch；缺失时明确停止，不生成任何虚假指标。

用法:
  python scripts/eval.py --config configs/raa.yaml --method raa --dataset dtd --seeds 0 1 2 3 4
"""
import argparse
import os
import sys
import json
import itertools

try:
    import torch
    import torch.nn.functional as F
    import open_clip
    import yaml
except ImportError as e:
    print("[停止] 缺少训练依赖（torch / open_clip_torch / pyyaml）。")
    print(f"       原始错误: {e}")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_raa as M  # noqa: E402


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_image_cache(dataset, seed, cfg, model, preprocess, device, dtype):
    """一次性编码某个划分的全部图像，供多个方法复用。

    骨干冻结，图像特征与方法无关；缓存后多方法评估从"重复读图编码"变成矩阵乘法，
    显存额外占用约 10-20MB，远低于 12GB 上限。
    """
    import pandas as pd
    from PIL import Image

    split_fp = os.path.join(ROOT, "data", "splits", f"{dataset}_seed{seed}.csv")
    if not os.path.exists(split_fp):
        return None
    df = pd.read_csv(split_fp)
    paths = df["image_path"].tolist()
    bs = int(cfg.get("eval_batch", 64)) * 2
    feats = []
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        with torch.no_grad():
            f = F.normalize(model.encode_image(x).to(dtype), dim=-1)
        feats.append(f)
    return {
        "df": df,
        "feats": torch.cat(feats).to(device),
        "base_classes": sorted(df[df["split"] == "base_train"]["class_name"].unique()),
        "all_classes": sorted(df["class_name"].unique()),
    }


def evaluate_one(method, dataset, seed, cfg, model, preprocess, device, dtype,
                 suffix="", cache=None):
    split_fp = os.path.join(ROOT, "data", "splits", f"{dataset}_seed{seed}.csv")
    ctx_fp = os.path.join(ROOT, "results", dataset, f"ctx_{method}{suffix}_seed{seed}.pt")
    if not os.path.exists(split_fp):
        return None
    if method != "clip0" and not os.path.exists(ctx_fp):
        return None  # 缺少真实训练产物，跳过，不伪造

    if cache is None:
        import pandas as pd
        df = pd.read_csv(split_fp)
        base_classes = sorted(df[df["split"] == "base_train"]["class_name"].unique())
        all_classes = sorted(df["class_name"].unique())
    else:
        df = cache["df"]
        base_classes = cache["base_classes"]
        all_classes = cache["all_classes"]
    tokenizer = open_clip.get_tokenizer(cfg["backbone"])

    # 文本特征：clip0 用手工模板；其它用训练得到的 prompt ctx
    with torch.no_grad():
        if method == "clip0":
            txt = tokenizer([f"a photo of a {c.replace('_',' ')}." for c in all_classes]).to(device)
            text_feat = M.encode_text_features(model, txt)  # 官方编码路径（含位置编码+attn_mask）
        else:
            prompt = M.CoOpPrompt(model, all_classes, n_ctx=cfg["n_ctx"]).to(device).to(dtype)
            ctx = torch.load(ctx_fp, map_location=device).to(dtype)
            prompt.ctx.data = ctx
            M.set_tokenizer(tokenizer)
            M.tokenize = lambda p: tokenizer(p).to(device)
            text_feat = prompt.forward_text_features()

    def acc_on(split_name, class_filter):
        """批量前向评估。CLIP ViT 用 LayerNorm 且处于 eval 模式，
        批量与逐张前向数值等价（仅浮点累加顺序有微小差异）。
        已用 DTD clip0 逐张版结果对照验证一致性。"""
        sub = df[df["split"] == split_name]
        sub = sub[sub["class_name"].isin(class_filter)]
        if sub.empty:
            return 0.0, 0

        if cache is not None:
            idx = sub.index.tolist()
            gts = sub["class_name"].tolist()
            with torch.no_grad():
                logits = (cache["feats"][idx] @ text_feat.T) * model.logit_scale.exp()
                preds = logits.argmax(dim=-1).cpu().tolist()
            correct = sum(int(all_classes[int(pi)] == gts[j]) for j, pi in enumerate(preds))
            return correct / len(gts), len(gts)

        from PIL import Image  # 仅在无缓存的兼容路径中需要
        paths = sub["image_path"].tolist()
        gts = sub["class_name"].tolist()
        bs = int(cfg.get("eval_batch", 64))
        correct = 0
        for i in range(0, len(paths), bs):
            chunk = paths[i:i + bs]
            x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            with torch.no_grad():
                f = F.normalize(model.encode_image(x).to(dtype), dim=-1)  # [B, D]
                logits = (f @ text_feat.T) * model.logit_scale.exp()      # [B, C]
                preds = logits.argmax(dim=-1).cpu().tolist()
            for j, pi in enumerate(preds):
                correct += int(all_classes[int(pi)] == gts[i + j])
        return correct / len(paths), len(paths)

    novel_classes = [c for c in all_classes if c not in base_classes]
    base_cls = [c for c in all_classes if c in base_classes]
    b, nb = acc_on("base_test", base_cls)
    n, nn = acc_on("novel_test", novel_classes)
    hm = 2 * b * n / (b + n) if (b + n) > 0 else 0.0
    return {"base_acc": round(b, 4), "novel_acc": round(n, 4), "hm": round(hm, 4),
            "n_base": nb, "n_novel": nn}


def main():
    all_methods = ["clip0", "coop", "kgcoop", "raa", "raa_rev", "raa_margin",
                   "raa_margin_rev", "raa_anneal", "raa_margin_anneal"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "raa.yaml"))
    ap.add_argument("--method", choices=all_methods, default=None, help="兼容单方法调用")
    ap.add_argument("--methods", nargs="+", choices=all_methods, default=None,
                    help="一次加载模型评估多个方法，显著减少模型重复加载时间")
    ap.add_argument("--dataset", default="dtd")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="", help="结果文件后缀；须与训练时的 --tag 一致")
    args = ap.parse_args()

    methods = args.methods or ([args.method] if args.method else ["raa"])
    tag = args.tag.strip()
    suffix = f"_{tag}" if tag else ""
    cfg = load_cfg(args.config)
    if not torch.cuda.is_available():
        print("[停止] CUDA 不可用。")
        sys.exit(1)
    device, dtype = "cuda", (torch.float16 if cfg.get("fp16", True) else torch.float32)
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["backbone"], pretrained=cfg["pretrained"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 图像特征与提示方法无关：每种 seed 只编码一次，所有方法复用。
    caches = {}
    for s in args.seeds:
        c = build_image_cache(args.dataset, s, cfg, model, preprocess, device, dtype)
        if c is None:
            print(f"[跳过缓存] seed={s} 缺少真实划分文件。")
            continue
        caches[s] = c
    if not caches:
        print("[停止] 没有任何可缓存的真实划分，拒绝生成指标。")
        return 1
    print(f"[缓存] 已为 {len(caches)} 个种子预编码图像特征，供 {len(methods)} 个方法复用。")

    outdir = os.path.join(ROOT, "results", args.dataset)
    os.makedirs(outdir, exist_ok=True)
    any_ok = False
    for method in methods:
        results = []
        for s in args.seeds:
            r = evaluate_one(method, args.dataset, s, cfg, model, preprocess,
                             device, dtype, suffix, cache=caches.get(s))
            if r is None:
                print(f"[跳过] {method} seed={s} 缺少真实划分/权重文件，不伪造结果。")
                continue
            r["seed"] = s
            results.append(r)
            print(f"{method} seed={s} {r}")

        if not results:
            continue
        summary = {}
        for k in ["base_acc", "novel_acc", "hm"]:
            vals = [r[k] for r in results]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            summary[k] = {"mean": round(mean, 4), "std": round(std, 4), "seeds": len(vals)}
        out = {"method": method, "dataset": args.dataset, "tag": tag,
               "summary": summary, "per_seed": results}
        fp = os.path.join(outdir, f"eval_{method}{suffix}.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n=== {method} 汇总（均值±标准差）===")
        for k, v in summary.items():
            print(f"  {k:<10}: {v['mean']:.4f} ± {v['std']:.4f}  (n={v['seeds']} seeds)")
        print(f"结果已写入 {fp}")
        any_ok = True

    if not any_ok:
        print("[停止] 没有任何可评估的真实结果。请先完成数据划分与训练。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
