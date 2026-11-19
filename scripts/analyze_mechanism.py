#!/usr/bin/env python3
"""机制证据：验证 novel 挤压率 → SCG 的 novel 掉点 的因果关系强度。

审稿人最吃的不是"我们方法涨点"，而是"我们定位并证明了一个机制"。
本脚本把"novel 被挤压到 base 类"从定性解释升级成定量证据：
  挤压率 x = SCG 下 novel 测试样本被判为 base 类的比例（run_decouple 实测）
  掉点   y = clip0 novel_acc - SCG(gate_calibrated) novel_acc（同 seed 同数据）
若跨 15 个观测（3 数据集 × 5 seeds）Pearson r 显著为正，则
"挤压导致掉点"从作者的口头声称变成数据支持的机制结论。

同时给出解耦后"回收量" vs 挤压率的相关性——若成立，则说明
解耦正是按挤压程度按比例救回 novel，机制闭环。

诚实标注：相关 ≠ 因果；这只是观察性证据，不宣称证明因果。
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
          6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
          11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x)
    dy = sum((b - my) ** 2 for b in y)
    if dx <= 0 or dy <= 0:
        return 0.0
    return num / math.sqrt(dx * dy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dtd", "oxfordpets", "flowers102"])
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    points = []
    by_ds = {}
    for ds in args.datasets:
        fp = os.path.join(ROOT, "results", ds, f"decouple_{args.tag}.json")
        if not os.path.exists(fp):
            print(f"[跳过] 缺少 {fp}")
            continue
        obj = json.load(open(fp, encoding="utf-8"))
        rows = []
        for r in obj["per_seed"]:
            squeeze = r["novel_squeezed_ratio"]
            clip0_n = r["methods"]["clip0"]["novel_acc"]
            scg_n = r["methods"]["gate_calibrated"]["novel_acc"]
            dec_n = r["methods"]["decouple_scg"]["novel_acc"]
            drop = clip0_n - scg_n          # SCG 造成的 novel 掉点
            recover = dec_n - scg_n         # 解耦救回的量
            rows.append({"seed": r["seed"], "squeeze": squeeze,
                         "drop": round(drop, 4), "recover": round(recover, 4)})
            points.append({"dataset": ds, **rows[-1]})
        by_ds[ds] = rows

    if len(points) < 3:
        print("[停止] 观测点不足，无法做相关分析。")
        return 1

    squeeze = [p["squeeze"] for p in points]
    drop = [p["drop"] for p in points]
    recover = [p["recover"] for p in points]

    r_drop = pearson(squeeze, drop)
    r_recover = pearson(squeeze, recover)
    n = len(points)
    # t 检验相关性显著性
    def sig(r, n):
        if abs(r) >= 1:
            return True
        t = r * math.sqrt((n - 2) / (1 - r * r))
        return abs(t) > T_CRIT.get(n - 2, 2.0)
    sig_drop, sig_recover = sig(r_drop, n), sig(r_recover, n)

    print(f"=== 机制相关性（n={n} 观测 = {len(by_ds)} 数据集 × 5 seeds）===")
    print(f"挤压率  vs  SCG novel 掉点 : r={r_drop:+.3f}  "
          f"{'显著' if sig_drop else '不显著'}  （预期强正）")
    print(f"挤压率  vs  解耦救回量      : r={r_recover:+.3f}  "
          f"{'显著' if sig_recover else '不显著'}  （预期强正）")

    print("\n=== 分数据集明细 ===")
    for ds, rows in by_ds.items():
        s = sum(r["squeeze"] for r in rows) / len(rows)
        d = sum(r["drop"] for r in rows) / len(rows)
        rc = sum(r["recover"] for r in rows) / len(rows)
        print(f"  {ds:<12} 挤压={s*100:5.1f}%  SCG掉点={d*100:+5.1f}pp  "
              f"解耦救回={rc*100:+5.1f}pp")

    out = {
        "n_observations": n,
        "pearson_squeeze_vs_drop": {"r": round(r_drop, 4), "significant": sig_drop},
        "pearson_squeeze_vs_recover": {"r": round(r_recover, 4), "significant": sig_recover},
        "per_dataset_mean": {
            ds: {
                "squeeze": round(sum(r["squeeze"] for r in rows) / len(rows), 4),
                "drop": round(sum(r["drop"] for r in rows) / len(rows), 4),
                "recover": round(sum(r["recover"] for r in rows) / len(rows), 4),
            } for ds, rows in by_ds.items()
        },
        "points": points,
        "caveat": "相关≠因果；观察性证据，不宣称证明因果。",
    }
    outdir = os.path.join(ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join(outdir, "mechanism_correlation.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果写入 {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
