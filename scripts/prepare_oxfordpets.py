#!/usr/bin/env python3
"""从 OxfordPets 官方数据集生成标注清单 data/oxfordpets/annotations.csv。

官方结构（解压 images.tar.gz + annotations.tar.gz 后）:
    images/<ClassName>_<idx>.jpg      37 类，共 7349 张
    annotations/trainval.txt / test.txt
类别名从文件名前缀提取（去掉末尾 _<数字>.jpg），例如:
    Abyssinian_1.jpg        -> Abyssinian
    Egyptian_Mau_12.jpg     -> Egyptian_Mau
    Yorkshire_Terrier_3.jpg -> Yorkshire_Terrier

只读取真实图片文件，不生成任何合成样本。
输出列与 prepare_dtd.py 完全一致: image_path,class_name

用法:
  python scripts/prepare_oxfordpets.py --src data/oxfordpets/images
"""
import argparse
import os
import re
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"^(.+)_\d+$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "oxfordpets", "images"),
                    help="解压后的 images 目录路径")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"[停止] 找不到图片目录: {args.src}")
        print("请先下载并解压 images.tar.gz 与 annotations.tar.gz。")
        return 1

    files = sorted(f for f in os.listdir(args.src)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        print(f"[停止] {args.src} 下没有图片文件。")
        return 1

    rows, skipped = [], []
    for f in files:
        stem = os.path.splitext(f)[0]
        m = PAT.match(stem)
        if not m:
            skipped.append(f)
            continue
        cls = m.group(1)
        rows.append({"image_path": os.path.join(args.src, f).replace("\\", "/"),
                     "class_name": cls})

    if not rows:
        print("[停止] 没有任何文件名能解析出类别，请检查数据是否完整。")
        return 1

    df = pd.DataFrame(rows)
    cats = sorted(df["class_name"].unique())
    out = os.path.join(ROOT, "data", "oxfordpets", "annotations.csv")
    df.to_csv(out, index=False)
    print(f"[完成] 类别数={len(cats)}  图片总数={len(df)}  跳过文件数={len(skipped)}")
    print(f"  每类图片数范围: {df.groupby('class_name').size().min()} ~ {df.groupby('class_name').size().max()}")
    print(f"  标注清单已写入: {out}")
    print("  类别示例:", ", ".join(cats[:6]), "...")
    if len(cats) != 37:
        print(f"  [注意] 官方 OxfordPets 为 37 类，本次解析到 {len(cats)} 类，请核对数据源完整性。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
