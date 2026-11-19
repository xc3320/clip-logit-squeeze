#!/usr/bin/env python3
"""从 Flowers102 官方数据生成标注清单 data/flowers102/annotations.csv。

官方结构（解压 102flowers.tgz 后）:
    jpg/image_00001.jpg ... image_08189.jpg
    imagelabels.mat  中 labels[0] 是 1..102 的类编号，顺序对应 image_00001..08189

类别名：Flowers102 官方无文本类名（只有编号）。为对齐 CLIP 手工模板
"a photo of a {class}"，需要人类可读的类别名。这里用官方 cat_to_name 映射表
（102 类学名），若本地无该映射则退回 'class_001' 形式并在输出中明确标注，
绝不臆造学名。

只读取真实图片与真实标签文件，不生成任何合成样本。
输出列与 prepare_dtd.py / prepare_oxfordpets.py 完全一致: image_path,class_name

用法:
  python scripts/prepare_flowers102.py
  python scripts/prepare_flowers102.py --labels-json data/flowers102/cat_to_name.json
"""
import argparse
import json
import os
import sys
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "data", "flowers102")
IMGDIR = os.path.join(DS, "jpg")
LABELS = os.path.join(DS, "imagelabels.mat")
NUM = re.compile(r"image_(\d+)\.jpg$", re.I)


def default_names():
    """本地无 cat_to_name.json 时的占位类名（明确非学名，不臆造）。"""
    return {i: f"class_{i:03d}" for i in range(1, 103)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-json", default=os.path.join(DS, "cat_to_name.json"),
                    help="可选：{类编号字符串: 学名} 的 JSON。缺失则用 class_XXX 占位")
    args = ap.parse_args()

    if not os.path.isfile(LABELS):
        print(f"[停止] 找不到标签文件 {LABELS}")
        return 1
    if not os.path.isdir(IMGDIR):
        print(f"[停止] 找不到图片目录 {IMGDIR}，请先解压 102flowers.tgz")
        return 1

    labels = loadmat(LABELS)["labels"].reshape(-1)      # 长度 8189，值 1..102
    files = sorted(f for f in os.listdir(IMGDIR) if NUM.match(f))
    if not files:
        print(f"[停止] {IMGDIR} 下无 image_XXXXX.jpg 图片")
        return 1

    # 类名映射：优先用户提供的真实学名表
    if os.path.isfile(args.labels_json):
        raw = json.load(open(args.labels_json, encoding="utf-8"))
        names = {int(k): str(v).replace(" ", "_") for k, v in raw.items()}
        name_src = f"真实学名表 {args.labels_json}"
    else:
        names = default_names()
        name_src = "占位类名 class_XXX（官方数据只有编号，未提供学名表）"

    # 按文件名编号 -> 标签（labels 顺序对应 image_00001..08189）
    lab_by_idx = {i + 1: int(labels[i]) for i in range(len(labels))}
    rows, skipped = [], []
    for f in files:
        m = NUM.match(f)
        idx = int(m.group(1))
        if idx not in lab_by_idx:
            skipped.append(f)
            continue
        c = lab_by_idx[idx]
        rows.append({"image_path": os.path.join(IMGDIR, f).replace("\\", "/"),
                     "class_name": names.get(c, f"class_{c:03d}")})

    if not rows:
        print("[停止] 没有任何图片匹配到标签。")
        return 1

    df = pd.DataFrame(rows)
    cats = sorted(df["class_name"].unique())
    out = os.path.join(DS, "annotations.csv")
    df.to_csv(out, index=False)
    print(f"[完成] 类别数={len(cats)}  图片总数={len(df)}  跳过={len(skipped)}")
    print(f"  类名来源: {name_src}")
    print(f"  每类图片数范围: {df.groupby('class_name').size().min()} ~ {df.groupby('class_name').size().max()}")
    print(f"  标注清单: {out}")
    if len(cats) != 102:
        print(f"  [注意] 官方 Flowers102 为 102 类，本次解析到 {len(cats)} 类，请核对数据完整性。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
