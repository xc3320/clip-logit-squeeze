#!/usr/bin/env python3
"""从 FGVC-Aircraft 2013b 官方数据生成标注清单 data/fgvcaircraft/annotations.csv。

官方结构:
  fgvcaircraft/fgvc-aircraft-2013b/data/
    variants.txt                    100 个 variant 型号名（如 707-320, 737-300）
    manufacturers.txt               厂商名（如 Boeing, Airbus）
    images_variant_trainval.txt     "图像ID variant"  6667 行
    images_variant_test.txt         "图像ID variant"  3333 行
    images_manufacturer_trainval.txt "图像ID manufacturer"
    images_manufacturer_test.txt
  图片（解压大包 fgvc-aircraft-2013b.tar.gz 后）:
    fgvc-aircraft-2013b/images/<图像ID>.jpg   （共 10000 张）

类名设计（关键）:
  CLIP 模板 "a photo of a {class}" 对裸型号 "737-300" 语义弱。标准做法用
  "厂商 + 型号" 组成可读类名，如 "Boeing 737-300"，经空格→下划线后写为
  "Boeing_737-300"。厂商来自官方 images_manufacturer_*.txt，逐图对齐，
  不臆造；若某图缺厂商标注则退回纯 variant 并在 stderr 计数。

输出列与 prepare_dtd.py / prepare_flowers102.py 完全一致: image_path,class_name
只读取真实图片与真实标注，不生成任何合成样本。

用法:
  python scripts/prepare_fgvcaircraft.py
"""
import os
import sys
from collections import defaultdict
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "data", "fgvcaircraft")
DATA = os.path.join(DS, "fgvc-aircraft-2013b", "data")
IMGDIR = os.path.join(DATA, "images")  # 官方图片在 data/images/ 下


def read_pairs(path):
    """每行 'ID label' -> {ID: label}。"""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                out[parts[0]] = " ".join(parts[1:])
    return out


def main():
    if not os.path.isdir(DATA):
        print(f"[停止] 找不到标注目录 {DATA}")
        return 1
    if not os.path.isdir(IMGDIR):
        print(f"[停止] 找不到图片目录 {IMGDIR}，请先解压 fgvc-aircraft-2013b.tar.gz")
        return 1

    var = {}
    for fn in ["images_variant_trainval.txt", "images_variant_test.txt"]:
        var.update(read_pairs(os.path.join(DATA, fn)))
    manuf = {}
    for fn in ["images_manufacturer_trainval.txt", "images_manufacturer_test.txt"]:
        manuf.update(read_pairs(os.path.join(DATA, fn)))

    if not var:
        print("[停止] 未读到任何 variant 标注。")
        return 1

    # 校验：每个 variant 至少有一个厂商标注，取该 variant 下最常见的厂商作为可读前缀
    v2m = defaultdict(lambda: defaultdict(int))
    for img_id, v in var.items():
        m = manuf.get(img_id)
        if m:
            v2m[v][m] += 1

    rows, missing_manuf = [], 0
    for img_id, v in sorted(var.items()):
        img_fp = os.path.join(IMGDIR, f"{img_id}.jpg")
        if not os.path.isfile(img_fp):
            continue  # 图片缺失则不纳入，不伪造
        m = manuf.get(img_id)
        if m:
            cls = f"{m}_{v}".replace(" ", "_")
        else:
            cls = v.replace(" ", "_")
            missing_manuf += 1
        rows.append({"image_path": img_fp.replace("\\", "/"), "class_name": cls})

    if not rows:
        print("[停止] 没有任何图片文件匹配到标注（图片可能尚未解压）。")
        return 1

    df = pd.DataFrame(rows)
    cats = sorted(df["class_name"].unique())
    out = os.path.join(DS, "annotations.csv")
    df.to_csv(out, index=False)
    print(f"[完成] 类别数={len(cats)}  图片总数={len(df)}  缺厂商标注={missing_manuf}")
    print(f"  类名示例: {cats[:5]}")
    print(f"  每类图片数范围: {df.groupby('class_name').size().min()} ~ {df.groupby('class_name').size().max()}")
    print(f"  标注清单: {out}")
    if len(cats) != 100:
        print(f"  [注意] 官方 FGVC-Aircraft 为 100 类，本次解析到 {len(cats)} 类，请核对。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
