#!/usr/bin/env python3
"""从 DTD 官方数据集目录生成标注清单 data/dtd/annotations.csv。

DTD 官方结构（解压 dtd-r1.0.1.tar.gz 后）:
    dtd/images/<category>/*.jpg      47 类，每类 120 张
本脚本只读取真实图片文件，扫描目录生成清单，不生成任何合成样本。

用法:
  python scripts/prepare_dtd.py --src data/dtd/dtd/images
输出:
  data/dtd/annotations.csv  列: image_path,class_name
"""
import argparse
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "dtd", "dtd", "images"),
                    help="解压后的 images 目录路径")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"[停止] 找不到图片目录: {args.src}")
        print("请先下载并解压 dtd-r1.0.1.tar.gz。")
        return 1

    cats = sorted([d for d in os.listdir(args.src)
                   if os.path.isdir(os.path.join(args.src, d))])
    if not cats:
        print(f"[停止] {args.src} 下没有类别子目录。")
        return 1

    rows = []
    for c in cats:
        folder = os.path.join(args.src, c)
        imgs = [f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for f in imgs:
            rows.append({"image_path": os.path.join(folder, f).replace("\\", "/"),
                         "class_name": c})

    if not rows:
        print("[停止] 没有扫描到任何图片。")
        return 1

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "data", "dtd", "annotations.csv")
    df.to_csv(out, index=False)
    print(f"[完成] 类别数={len(cats)}  图片总数={len(df)}")
    print(f"  每类图片数范围: {df.groupby('class_name').size().min()} ~ {df.groupby('class_name').size().max()}")
    print(f"  标注清单已写入: {out}")
    print("  类别示例:", ", ".join(cats[:6]), "...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
