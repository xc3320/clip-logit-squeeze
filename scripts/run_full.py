#!/usr/bin/env python3
"""完整对照实验：4 方法 × 5 种子，训练后评估，汇总 Base/Novel/HM。

这是真实数据实验，会实际调用 train.py 与 eval.py 的逻辑（同进程复用CLIP模型以加速）。
无任何真实数据时明确停止。结果写入 results/dtd/。
"""
import os
import sys
import json
import argparse
import itertools
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable  # 当前 venv 的 python

METHODS = ["clip0", "coop", "kgcoop", "raa", "raa_rev", "raa_margin",
           "raa_margin_rev", "raa_anneal", "raa_margin_anneal"]
SEEDS = [0, 1, 2, 3, 4]
DATASET = "dtd"  # 由命令行 --dataset 覆盖


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=os.environ.copy())
    if r.returncode != 0:
        print(f"[错误] 命令失败(exit={r.returncode})，中止实验。", flush=True)
        sys.exit(r.returncode)


def main():
    global DATASET
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dtd", choices=["dtd", "oxfordpets", "caltech101", "eurosat"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS,
                    help="要训练/评估的方法；默认全部")
    ap.add_argument("--tag", default="", help="结果文件后缀；新实验建议 seedfix，不覆盖旧结果")
    args = ap.parse_args()
    DATASET = args.dataset
    seeds = args.seeds
    methods = args.methods
    tag = args.tag.strip()
    suffix = f"_{tag}" if tag else ""
    tag_cmd = ["--tag", tag] if tag else []

    # 先确认数据齐备
    r = subprocess.run([PY, os.path.join(HERE, "build_split.py"), "--check"])
    if r.returncode != 0:
        print("[停止] 数据未齐备。")
        sys.exit(1)

    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # 训练：一次加载 CLIP 模型，批量跑所有方法×种子（clip0 无训练，脚本内跳过）
    run([PY, os.path.join(HERE, "train_batch.py"),
         "--dataset", DATASET, "--seeds"] + [str(s) for s in seeds] +
        ["--methods"] + methods + tag_cmd)

    # 评估：一次加载 CLIP 模型评估全部方法，避免每个方法重复加载权重
    run([PY, os.path.join(HERE, "eval.py"), "--methods"] + methods +
        ["--dataset", DATASET, "--seeds"] + [str(s) for s in seeds] + tag_cmd)

    # 汇总对照表
    print("\n" + "=" * 70)
    print("最终对照表（均值±标准差，5 seeds）")
    print("=" * 70)
    print(f"{'方法':<10}{'Base Acc':<18}{'Novel Acc':<18}{'HM':<18}")
    table = {}
    for method in methods:
        fp = os.path.join(ROOT, "results", DATASET, f"eval_{method}{suffix}.json")
        if not os.path.exists(fp):
            continue
        d = json.load(open(fp, encoding="utf-8"))
        s = d["summary"]
        row = {k: f"{v['mean']*100:.2f}±{v['std']*100:.2f}" for k, v in s.items()}
        table[method] = {k: [s[k]["mean"], s[k]["std"]] for k in s}
        print(f"{method:<10}{row.get('base_acc',''):<18}{row.get('novel_acc',''):<18}{row.get('hm',''):<18}")
    out = os.path.join(ROOT, "results", DATASET, f"summary{suffix}.json")
    merged = {}
    if os.path.exists(out):
        try:
            merged = json.load(open(out, encoding="utf-8"))
        except Exception:
            merged = {}
    merged.update(table)  # 子集运行时保留此前其它方法的结果，不覆盖历史
    json.dump(merged, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\n汇总已写入 {out}")
    print("注：clip0=零样本参照；RAA 与 KgCoOp 平均锚定强度相同(λ=8)，仅逐类自适应不同。")


if __name__ == "__main__":
    main()
