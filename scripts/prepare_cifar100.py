#!/usr/bin/env python3
"""从 HuggingFace 镜像下载的 CIFAR-100 parquet 生成标注清单 data/cifar100/annotations.csv。

真实来源: https://hf-mirror.com/datasets/uoft-cs/cifar100
  parquet/train.parquet (50000 行), parquet/test.parquet (10000 行)
  字段: img{bytes,path}, fine_label(0..99), coarse_label
类名映射: 从同仓库 README.md 的 fine_label class_label.names 提取（官方 100 类名），
  不臆造；若 README 缺失则明确停止。

说明:
  - CIFAR-100 不在标准 base-to-novel 11 数据集基准名单内，仅作第 5 个扩充验证集，
    检验 SCG+解耦的 novel 挤压现象是否跨更多数据分布成立；报告须如实标注。
  - 只解码真实图片字节写盘为 PNG，不生成任何合成样本。

输出列与其余 prepare_* 一致: image_path,class_name
用法:
  python scripts/prepare_cifar100.py
"""
import io
import os
import re
import sys
import pyarrow.parquet as pq
from PIL import Image
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "data", "cifar100")
PARQUET = os.path.join(DS, "parquet")
IMGDIR = os.path.join(DS, "images")
README = os.path.join(DS, "parquet", "README.md")


def load_names():
    """从 README 的 fine_label names 段提取 {0..99: 类名}。"""
    if not os.path.isfile(README):
        return None
    txt = open(README, encoding="utf-8").read().splitlines()
    names = {}
    in_fine = False
    for line in txt:
        if re.search(r"name:\s*fine_label", line):
            in_fine = True
            continue
        if in_fine and re.search(r"name:\s*coarse_label", line):
            break
        if in_fine:
            m = re.match(r"\s*'(\d+)':\s*(\S+)", line)
            if m:
                names[int(m.group(1))] = m.group(2)
    return names if len(names) == 100 else None


def main():
    names = load_names()
    if names is None:
        print(f"[停止] 无法从 {README} 提取 100 个官方 fine_label 类名。")
        print("       请把 HF 仓库 README.md 放到该路径，不臆造类名。")
        return 1

    os.makedirs(IMGDIR, exist_ok=True)
    rows = []
    for split in ["train", "test"]:
        fp = os.path.join(PARQUET, f"{split}.parquet")
        if not os.path.isfile(fp):
            print(f"[停止] 找不到 {fp}")
            return 1
        t = pq.read_table(fp, columns=["img", "fine_label"])
        imgs = t.column("img").to_pylist()
        labels = t.column("fine_label").to_pylist()
        for i, (img_struct, lab) in enumerate(zip(imgs, labels)):
            b = img_struct["bytes"]
            im = Image.open(io.BytesIO(b)).convert("RGB")
            out_fp = os.path.join(IMGDIR, f"{split}_{i:05d}.png")
            im.save(out_fp)
            rows.append({"image_path": out_fp.replace("\\", "/"),
                         "class_name": names[int(lab)]})

    df = pd.DataFrame(rows)
    cats = sorted(df["class_name"].unique())
    out = os.path.join(DS, "annotations.csv")
    df.to_csv(out, index=False)
    print(f"[完成] 类别数={len(cats)}  图片总数={len(df)}")
    print(f"  类名示例: {cats[:5]}")
    print(f"  每类图片数: {sorted(set(df.groupby('class_name').size().tolist()))}")
    print(f"  标注清单: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
