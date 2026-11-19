#!/usr/bin/env python3
"""尺度不对称方向诊断：对比 ViT-B/16 与 RN50 骨干上 text/visual logits 的标准差。

目的：为 M5（跨骨干鲁棒性）提供量化证据——
  ViT-B/16 上 text logits 标准差 >> visual logits（文本主导门控）
  RN50 上是否反转（CIDA 报告 ResNet-50 上视觉主导）？
"""
import os, sys, argparse, json
import torch, open_clip, pandas as pd, numpy as np
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import models_scg as S

def load_model(backbone, pretrained, device):
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(backbone)
    return model, preprocess, tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dtd")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.float32

    cfgs = [
        ("ViT-B/16", "ViT-B-16-quickgelu", "openai"),
        ("RN50", "RN50", "weights/openclip_RN50_openai.pt"),
    ]

    split_fp = os.path.join(ROOT, "data", "splits", f"{args.dataset}_seed{args.seed}.csv")
    df = pd.read_csv(split_fp)
    train_df = df[df["split"] == "base_train"]
    test_df = df[df["split"].isin(["base_test", "novel_test"])].reset_index(drop=True)
    base_classes = sorted(train_df["class_name"].unique())
    all_classes = sorted(df["class_name"].unique())
    c_base = len(base_classes)
    base_idx = [all_classes.index(c) for c in base_classes]
    c2i = {c: i for i, c in enumerate(base_classes)}

    out = {}
    for name, backbone, pretrained in cfgs:
        model, preprocess, tokenizer = load_model(backbone, pretrained, device)

        def encode(subdf):
            feats = []
            bs = 64
            paths = subdf["image_path"].tolist()
            for i in range(0, len(paths), bs):
                chunk = paths[i:i+bs]
                x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)
                with torch.no_grad():
                    f = F.normalize(model.encode_image(x).to(dtype), dim=-1)
                feats.append(f)
            return torch.cat(feats) if feats else torch.empty(0, device=device, dtype=dtype)

        tr_feat = encode(train_df)
        tr_lab = torch.tensor([c2i[c] for c in train_df["class_name"].tolist()], device=device)
        te_feat = encode(test_df)

        w_text = S.encode_text_all(model, tokenizer, all_classes, device).to(dtype)
        scale = float(model.logit_scale.exp().detach())
        w_vis = S.train_visual_probe(tr_feat, tr_lab, c_base,
                                     epochs=300, lr=0.05)

        text_l = S.text_logits(te_feat, w_text, scale)
        vis_l = S.visual_logits_all(te_feat, w_vis, base_idx, len(all_classes))

        # 逐样本标准差，再平均（base 类位置才看 visual）
        text_std = text_l.std(dim=-1).mean().item()
        vis_base = vis_l[:, base_idx]
        vis_std = vis_base.std(dim=-1).mean().item()
        ratio = text_std / (vis_std + 1e-8)
        out[name] = {
            "text_logit_std": round(text_std, 4),
            "visual_logit_std": round(vis_std, 4),
            "ratio_text_over_visual": round(ratio, 2),
            "dominant": "text" if ratio > 1 else "visual",
            "logit_scale": round(scale, 4),
        }
        print(f"[{name}] text_std={text_std:.4f} visual_std={vis_std:.4f} "
              f"ratio={ratio:.2f} -> {'文本主导' if ratio>1 else '视觉主导'} (logit_scale={scale:.2f})")

    fp = os.path.join(ROOT, "results", f"scale_direction_{args.dataset}.json")
    json.dump(out, open(fp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("saved", fp)

if __name__ == "__main__":
    main()
