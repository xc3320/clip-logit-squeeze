#!/usr/bin/env python3
"""解耦推理配对显著性分析：默认以 clip0 和 gate_calibrated 双参照报告配对差与 95%CI。

关键对照：
  - decouple_scg vs clip0          : novel 是否真的无损（应 Δ≈0）、HM/base 是否提升
  - decouple_scg vs gate_calibrated: novel 是否被救回（应 Δ>0 显著）、HM 是否提升
配对检验比独立均值比较更敏感（同一 seed 共享数据波动）。n=5 用 t(4)=2.776。
样本小，结论只作证据强弱判断，不宣称定论。
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
          6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def paired(y, x):
    d = [a - b for a, b in zip(y, x)]
    n = len(d)
    mean = sum(d) / n
    var = sum((v - mean) ** 2 for v in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    tc = T_CRIT.get(n - 1, 2.262)
    lo, hi = mean - tc * se, mean + tc * se
    return mean, lo, hi, (lo > 0 or hi < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dtd", "oxfordpets", "flowers102"])
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    all_res = {}
    for ds in args.datasets:
        fp = os.path.join(ROOT, "results", ds, f"decouple_{args.tag}.json")
        if not os.path.exists(fp):
            print(f"[跳过] 缺少 {fp}")
            continue
        obj = json.load(open(fp, encoding="utf-8"))
        per = obj["per_seed"]
        methods = obj["methods"]
        print(f"\n===== {ds}（n={len(per)} seeds）=====")
        print(f"  novel 被 SCG 挤压到 base 的比例: "
              f"{obj['novel_squeezed_ratio']['mean']*100:.2f}%")
        for ref in ["clip0", "gate_calibrated"]:
            print(f"  --- 对照 ref={ref} ---")
            for m in methods:
                if m == ref or not m.startswith("decouple"):
                    continue
                for k in ["hm", "base_acc", "novel_acc"]:
                    y = [r["methods"][m][k] for r in per]
                    x = [r["methods"][ref][k] for r in per]
                    mean, lo, hi, sig = paired(y, x)
                    flag = "显著" if sig else "不显著"
                    print(f"    {m:<18} {k:<10} Δ={mean*100:+.2f}pp "
                          f"95%CI[{lo*100:+.2f},{hi*100:+.2f}] {flag}")
        # 存图供并表
        all_res[ds] = obj["summary"]

    # decouple_scg 多数据集并表
    print(f"\n===== decouple_scg {len(all_res)} 数据集并表（均值%）=====")
    print(f"{'dataset':<12}{'HM':>9}{'Base':>9}{'Novel':>9}")
    for ds, s in all_res.items():
        d = s["decouple_scg"]
        print(f"{ds:<12}{d['hm']['mean']*100:>8.2f}{d['base_acc']['mean']*100:>9.2f}"
              f"{d['novel_acc']['mean']*100:>9.2f}")


if __name__ == "__main__":
    main()
