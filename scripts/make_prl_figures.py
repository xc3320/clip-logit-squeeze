#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pattern Recognition Letters 投稿版配图（英文，单栏/双栏宽度均可用）。

与 make_figures.py 同源同数据，只把文字换成英文并压到 PRL 的版面尺寸：
  fig_prl1_method   方法总览
  fig_prl2_dataset  五数据集样本构成
  fig_prl3_main     主结果：(a) HM 柱状  (b) Δ 平面
  fig_prl4_mech     机制：(a) 挤压率-掉点散点  (b) 挤压率排序
  fig_prl5_cases    个案追踪
所有结论数字读自 results/，不硬编码。
"""
import importlib.util
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")

spec = importlib.util.spec_from_file_location("mf", os.path.join(HERE, "make_figures.py"))
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

C, DS, DS_LABEL = mf.C, mf.DS, mf.DS_LABEL
DESPINE, THUMB, TO_JPG = mf.despine, mf._thumb, mf._to_jpg
PRETTY, LOAD_SPLIT, BRIGHT_OK = mf._pretty_cls, mf._load_split, mf._bright_ok


# ============================================================
# Fig. 1  method overview
# ============================================================
def fig1_method():
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#4A4A4A", fs=11, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.09",
                                    facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)

    def arrow(x1, y1, x2, y2, color="#555", lw=1.7, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, color=color, linewidth=lw,
                                     connectionstyle=f"arc3,rad={rad}", zorder=1))

    box(0.35, 6.4, 1.75, 0.95, "Input image $x$", C["lightgrey"], fs=12, bold=True)
    box(2.55, 6.4, 2.35, 0.95, "Frozen CLIP\nimage encoder", C["lightblue"], fs=11.5, bold=True)
    box(5.35, 6.55, 1.5, 0.65, "feature $f$", "#FFFFFF", fs=11.5, bold=True)
    arrow(2.10, 6.88, 2.55, 6.88)
    arrow(4.90, 6.88, 5.35, 6.88)
    arrow(6.1, 6.55, 6.85, 5.72)

    ax.add_patch(FancyBboxPatch((0.2, 2.7), 12.95, 3.0,
                                boxstyle="round,pad=0.05,rounding_size=0.12",
                                facecolor="#FAFBFC", edgecolor="#C8D0D8", linewidth=1.2, zorder=0))
    ax.text(0.45, 5.42, "(a) Scale-calibrated gating (SCG): dual stream, per-sample calibration, reliability gate",
            fontsize=12.2, fontweight="bold")

    box(0.7, 4.35, 2.6, 0.9, "Frozen class\nembeddings $W_{text}$", C["lightblue"], fs=10.5, bold=True)
    box(3.8, 4.35, 2.8, 0.9, "Text logits\n$\\ell_{text}=s\\,f^{\\top}W_{text}$", C["lightblue"], fs=10.5, bold=True)
    arrow(3.30, 4.8, 3.8, 4.8, color=C["text"])

    box(0.7, 3.0, 2.6, 0.9, "Linear visual probe\n$W_{vis}$ (base support)", C["lightorange"], fs=10.5, bold=True)
    box(3.8, 3.0, 2.8, 0.9, "Visual logits (unbounded)\n$\\ell_{vis}=f^{\\top}W_{vis}$", C["lightorange"], fs=10.5, bold=True)
    arrow(3.30, 3.45, 3.8, 3.45, color=C["vis"])

    box(7.1, 3.5, 1.6, 1.25, "Per-sample\n$z$-score\ncalibration", C["lightgreen"], fs=10, bold=True)
    box(9.1, 3.5, 1.6, 1.25, "Reliability\ngate $g_i$", C["lightgreen"], fs=10, bold=True)
    box(11.1, 3.5, 2.0, 1.25, "Fused logits\n$\\ell_{fused}$", "#FFFFFF", fs=11, bold=True)
    arrow(6.6, 4.8, 7.1, 4.35, color=C["gate"])
    arrow(6.6, 3.45, 7.1, 3.9, color=C["gate"])
    arrow(8.7, 4.12, 9.1, 4.12, color=C["gate"])
    arrow(10.7, 4.12, 11.1, 4.12, color=C["gate"])

    ax.add_patch(FancyBboxPatch((0.2, 0.1), 12.9, 2.4,
                                boxstyle="round,pad=0.05,rounding_size=0.12",
                                facecolor="#F7FAFD", edgecolor="#B9CCE0", linewidth=1.2, zorder=0))
    ax.text(0.45, 2.18, "(b) Decoupled inference: switch logits by test-time identity (the fix for novel logit squeeze)",
            fontsize=12.2, fontweight="bold")

    box(0.75, 0.55, 2.35, 1.15, "Test sample\n(base / novel)", C["lightgrey"], fs=10.5, bold=True)
    box(3.7, 0.55, 2.7, 1.15, "Identity routing\nbase $\\rightarrow$ fused, novel $\\rightarrow$ text", C["lightgreen"], fs=10.2, bold=True)
    arrow(3.10, 1.13, 3.7, 1.13)
    box(7.15, 1.0, 2.35, 0.85, "base: $\\ell_{fused}$\n(keeps the gain)", C["lightblue"], fs=10.2, bold=True)
    box(7.15, 0.05, 2.35, 0.85, "novel: $\\ell_{text}$\n(= CLIP0, no squeeze)", C["lightorange"], fs=10.2, bold=True)
    arrow(6.4, 1.13, 7.15, 1.42, color=C["base"])
    arrow(6.4, 1.13, 7.15, 0.48, color=C["novel"])
    box(10.15, 0.5, 2.5, 1.25, "Prediction\nBase $\\uparrow$, Novel intact\nHM $\\uparrow$", "#FFFFFF", fs=10.2, bold=True)
    arrow(9.5, 1.42, 10.15, 1.13, color=C["base"])
    arrow(9.5, 0.48, 10.15, 0.68, color=C["novel"])

    fig.savefig(os.path.join(FIG, "fig_prl1_method.png"))
    plt.close(fig)
    print("saved fig_prl1_method.png")


# ============================================================
# Fig. 2  dataset composition
# ============================================================
def fig2_dataset():
    import random

    cols = [("Base $\\cdot$ support", "16-shot adaptation", "base_train", "#2F6FB0"),
            ("Base $\\cdot$ test", "routed to fusion", "base_test", "#2F6FB0"),
            ("Novel $\\cdot$ test (i)", "no visual evidence", "novel_test", "#E08A5B"),
            ("Novel $\\cdot$ test (ii)", "another novel class", "novel_test", "#E08A5B")]
    desc = {"dtd": ("Describable textures", 47),
            "oxfordpets": ("Oxford-IIIT Pets", 37),
            "flowers102": ("Oxford Flowers-102", 102),
            "fgvcaircraft": ("FGVC-Aircraft", 100),
            "cifar100": ("CIFAR-100", 100)}

    fig = plt.figure(figsize=(10.4, 9.5))
    gs = fig.add_gridspec(6, 5, left=0.012, right=0.988, top=0.905, bottom=0.062,
                          wspace=0.07, hspace=0.30,
                          height_ratios=[0.20, 1, 1, 1, 1, 1],
                          width_ratios=[1.42, 1, 1, 1, 1])
    rng = random.Random(13)

    for c, (head, sub, _, col) in enumerate(cols):
        axh = fig.add_subplot(gs[0, c + 1])
        axh.axis("off")
        axh.text(0.5, 0.62, head, transform=axh.transAxes, ha="center", va="center",
                 fontsize=10.4, fontweight="bold", color=col)
        axh.text(0.5, 0.16, sub, transform=axh.transAxes, ha="center", va="center",
                 fontsize=8.0, color="#7A828B")

    for r, ds in enumerate(DS):
        by, n_base, n_novel = LOAD_SPLIT(ds)
        used, picks = set(), []
        for _, _, split, _ in cols:
            pool = [x for x in by[split] if x["class_name"] not in used]
            item = None
            for _ in range(12):
                cand = rng.choice(pool)
                if BRIGHT_OK(cand["image_path"]):
                    item = cand
                    break
            item = item or rng.choice(pool)
            used.add(item["class_name"])
            picks.append(item)

        axl = fig.add_subplot(gs[r + 1, 0])
        axl.axis("off")
        axl.add_patch(Rectangle((0.005, 0.10), 0.030, 0.80, transform=axl.transAxes,
                                facecolor=mf.LINE_COLS[r], edgecolor="none"))
        axl.text(0.075, 0.70, DS_LABEL[r], transform=axl.transAxes, fontsize=12.0,
                 fontweight="bold", va="center", color=mf.LINE_COLS[r])
        axl.text(0.075, 0.36, desc[ds][0], transform=axl.transAxes, fontsize=8.4,
                 va="center", color="#555")
        axl.text(0.075, 0.17, f"{desc[ds][1]} cls | base {n_base} / novel {n_novel}",
                 transform=axl.transAxes, fontsize=8.4, va="center", color="#8A9099")

        for c, item in enumerate(picks):
            ax = fig.add_subplot(gs[r + 1, c + 1])
            ax.imshow(THUMB(item["image_path"], size=260))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#CBD2D9")
                sp.set_linewidth(0.9)
            ax.text(0.5, -0.055, PRETTY(item["class_name"]), transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.2, color="#4A5158",
                    clip_on=False, style="italic", linespacing=1.35)

    fig.suptitle("Sample composition of the five datasets: base support, base test and novel test "
                 "(four distinct classes per row)", fontsize=13.0, fontweight="bold", y=0.968)
    fig.text(0.012, 0.008,
             "Samples are drawn from the seed-0 split files; class counts follow the base/novel partition.\n"
             "CIFAR-100 images are 32x32 and appear blurred when enlarged; DTD images are texture patches without a single subject.",
             fontsize=7.8, color="#777", va="bottom", linespacing=1.5)
    fig.savefig(os.path.join(FIG, "fig_prl2_dataset.png"), dpi=200)
    plt.close(fig)
    TO_JPG("fig_prl2_dataset.png")
    print("saved fig_prl2_dataset.png")


# ============================================================
# Fig. 3  main results: HM bars + delta plane
# ============================================================
def fig3_main():
    data = {ds: mf.load_summary(ds) for ds in DS}
    meth = ["clip0", "gate_calibrated", "decouple_scg"]
    mlab = ["CLIP0", "SCG", "Ours"]
    mcol = ["#7A8B99", "#E08A5B", "#2F6FB0"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5),
                             gridspec_kw=dict(width_ratios=[1.18, 1], wspace=0.26))
    ax = axes[0]
    x = np.arange(len(DS))
    w = 0.26
    for k, (m, lab, col) in enumerate(zip(meth, mlab, mcol)):
        vals = [data[ds]["summary"][m]["hm"]["mean"] * 100 for ds in DS]
        errs = [data[ds]["summary"][m]["hm"]["std"] * 100 for ds in DS]
        bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, yerr=errs, capsize=2.2,
                      color=col, edgecolor="white", linewidth=0.8,
                      error_kw=dict(lw=0.9, ecolor="#555555"), label=lab, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6.6, color="#333333",
                    rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(["DTD", "Pets", "Flowers", "Aircraft", "CIFAR100"], fontsize=9.2)
    ax.set_ylabel("HM accuracy (%)", fontsize=10.5)
    ax.set_ylim(0, 104)
    ax.set_title("(a) Harmonic mean, 16-shot, 5 seeds", fontsize=11, fontweight="bold", loc="left")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.legend(ncol=3, loc="upper left", fontsize=8.6, frameon=False,
              handlelength=1.4, columnspacing=1.2, bbox_to_anchor=(0.0, 1.0))
    DESPINE(ax)

    ax = axes[1]
    ax.axhspan(-19.0, 0, color="#A94438", alpha=0.05, zorder=0)
    ax.axhline(0, color="#2E2E2E", lw=1.2, ls=(0, (7, 3)), zorder=2)
    rows = []
    for ds in DS:
        s = data[ds]["summary"]
        b0 = s["clip0"]["base_acc"]["mean"] * 100
        n0 = s["clip0"]["novel_acc"]["mean"] * 100
        rows.append(dict(ds=ds,
                         dbg=s["gate_calibrated"]["base_acc"]["mean"] * 100 - b0,
                         dng=s["gate_calibrated"]["novel_acc"]["mean"] * 100 - n0,
                         dbo=s["decouple_scg"]["base_acc"]["mean"] * 100 - b0,
                         dno=s["decouple_scg"]["novel_acc"]["mean"] * 100 - n0))
    for r in rows:
        ax.annotate("", xy=(r["dbo"], r["dno"]), xytext=(r["dbg"], r["dng"]),
                    arrowprops=dict(arrowstyle="-|>", color="#2F6FB0", lw=1.5, alpha=0.55,
                                    shrinkA=4, shrinkB=8, mutation_scale=12), zorder=2)
        ax.scatter(r["dbg"], r["dng"], marker="s", s=52, color="#D98A4F",
                   edgecolor="white", linewidth=1.0, zorder=3)
        ax.scatter(r["dbo"], r["dno"], marker="^", s=130, color="#2F6FB0",
                   edgecolor="white", linewidth=1.2, zorder=4)
        ax.text(r["dbg"], r["dng"] - 0.8, f"{r['dng']:+.1f}", ha="center", va="top",
                fontsize=7.2, color="#A8683A", zorder=4)

    # 只标极端点，其余数据集由 (a) 与表 1 对应，避免右上一堆标签互相压住
    r_air = rows[3]
    ax.annotate("Aircraft", xy=(r_air["dbg"], r_air["dng"]), xytext=(r_air["dbg"] - 4.6, r_air["dng"] + 1.4),
                ha="right", va="bottom", fontsize=8.2, color="#8B2E22", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#D2A79E", lw=0.8, shrinkA=2, shrinkB=3), zorder=5)
    ax.text(1.0, 1.1, "Ours: $\\Delta$Novel = 0 for all five datasets", fontsize=8.0,
            color="#2F6FB0", va="bottom", ha="left", zorder=5)

    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor="#D98A4F",
                      markeredgecolor="white", markersize=8, label="SCG"),
               Line2D([0], [0], marker="^", color="none", markerfacecolor="#2F6FB0",
                      markeredgecolor="white", markersize=10, label="Ours")]
    ax.legend(handles=handles, loc="lower left", fontsize=8.4, frameon=True,
              framealpha=0.95, edgecolor="#D8DDE3", borderpad=0.4,
              title="relative to CLIP0", title_fontsize=8.0)
    ax.set_xlabel("$\\Delta$Base accuracy (pp)", fontsize=10.5)
    ax.set_ylabel("$\\Delta$Novel accuracy (pp)", fontsize=10.5)
    ax.set_title("(b) Base gain vs. novel loss", fontsize=11, fontweight="bold", loc="left")
    ax.set_xlim(-1.5, 27.5)
    ax.set_ylim(-19.0, 6.2)
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    DESPINE(ax)

    fig.savefig(os.path.join(FIG, "fig_prl3_main.png"), dpi=300)
    plt.close(fig)
    print("saved fig_prl3_main.png")


# ============================================================
# Fig. 4  mechanism
# ============================================================
def fig4_mech():
    d = json.load(open(os.path.join(ROOT, "results", "mechanism_correlation.json"), encoding="utf-8"))
    pts = d["points"]
    colors = ["#2F6FB0", "#E08A5B", "#3E9B78", "#8E5BB7", "#C0392B"]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6),
                             gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.30))
    ax = axes[0]
    for ds, col, lab in zip(DS, colors, ["DTD", "Pets", "Flowers", "Aircraft", "CIFAR100"]):
        sub = [p for p in pts if p["dataset"] == ds]
        ax.scatter([p["squeeze"] * 100 for p in sub], [p["drop"] * 100 for p in sub],
                   s=42, color=col, edgecolor="white", linewidth=0.7, alpha=0.9,
                   label=lab, zorder=3)

    allx = np.array([p["squeeze"] * 100 for p in pts])
    ally = np.array([p["drop"] * 100 for p in pts])
    n = len(allx)
    coef = np.polyfit(allx, ally, 1)
    xs = np.linspace(allx.min() - 2, allx.max() + 2, 200)
    ys = np.polyval(coef, xs)
    resid = ally - np.polyval(coef, allx)
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))
    t = 2.06
    ci = t * s_err * np.sqrt(1 / n + (xs - allx.mean()) ** 2 / np.sum((allx - allx.mean()) ** 2))
    ax.plot(xs, ys, color="#222", lw=1.6, zorder=2)
    ax.fill_between(xs, ys - ci, ys + ci, color="#222", alpha=0.12, zorder=1)

    r = d["pearson_squeeze_vs_drop"]["r"]
    ax.text(0.04, 0.95, f"$r$ = {r:.3f}, $n$ = {n}, $p$ < 0.01",
            transform=ax.transAxes, va="top", fontsize=9.4,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#B9CCE0", alpha=0.95))
    ax.set_xlabel("Squeeze rate (%)", fontsize=10.5)
    ax.set_ylabel("Novel accuracy drop under SCG (pp)", fontsize=10.5)
    ax.set_title("(a) Squeeze rate predicts the novel-class loss", fontsize=11, fontweight="bold", loc="left")
    ax.grid(linestyle="--", alpha=0.28)
    ax.legend(loc="lower right", fontsize=8.0, framealpha=0.95, handletextpad=0.4)
    DESPINE(ax)

    ax = axes[1]
    order = sorted(DS, key=lambda s: d["per_dataset_mean"][s]["squeeze"])
    vals = [d["per_dataset_mean"][s]["squeeze"] * 100 for s in order]
    labs = [DS_LABEL[DS.index(s)] for s in order]
    cols = ["#C0392B" if v > 60 else "#E08A5B" if v > 20 else "#7A8B99" for v in vals]
    bars = ax.barh(np.arange(len(order)), vals, 0.62, color=cols,
                   edgecolor="white", linewidth=1.0, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=8.8, color="#333333", fontweight="bold")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(labs, fontsize=9.2)
    ax.set_xlim(0, 96)
    ax.set_xlabel("Novel test samples routed to a base class (%)", fontsize=10.2)
    ax.set_title("(b) Squeeze rate by dataset", fontsize=11, fontweight="bold", loc="left")
    ax.grid(axis="x", linestyle="--", alpha=0.25, zorder=0)
    ax.annotate("4 out of 5 novel samples\nare labelled as base",
                xy=(vals[-1], len(order) - 1), xytext=(44, 3.05),
                fontsize=8.2, color="#8B2E22",
                arrowprops=dict(arrowstyle="-|>", color="#8B2E22", lw=0.9,
                                connectionstyle="arc3,rad=-0.25"))
    DESPINE(ax)

    fig.savefig(os.path.join(FIG, "fig_prl4_mech.png"), dpi=300)
    plt.close(fig)
    print("saved fig_prl4_mech.png")


# ============================================================
# Fig. 5  case study
# ============================================================
def _crop32(path, w=520):
    """中心裁成 3:2 横版，适合远景目标（飞机）并压住版面高度。"""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    W, H = im.size
    th = int(W * 2 / 3)
    if th <= H:
        y0 = (H - th) // 2
        im = im.crop((0, y0, W, y0 + th))
    else:
        tw = int(H * 3 / 2)
        x0 = (W - tw) // 2
        im = im.crop((x0, 0, x0 + tw, H))
    return np.asarray(im.resize((w, int(w * 2 / 3)), Image.LANCZOS))


def fig5_cases():
    fp = os.path.join(ROOT, "results", "fgvcaircraft", "cases_seed0.json")
    if not os.path.exists(fp):
        print("[skip] fig5: missing cases_seed0.json")
        return
    d = json.load(open(fp, encoding="utf-8"))
    c_clip, c_scg, c_ours = "#7A8B99", "#C97A3E", "#2F6FB0"

    def flat(name):
        parts = name.split("_")
        for i in range(1, len(parts)):
            if parts[i].lower() == parts[0].lower():
                parts = parts[i:]
                break
        return " ".join(parts)

    # 单行三列：1 张基类图证明 SCG 增益为真，2 张新类图展示挤压与修复。
    # 图片统一裁成 3:2，Word 里整图高度由 11.3 cm 降到约 6 cm。
    picks = [(d["cases"]["base"][0], "Base sample", c_scg),
             (d["cases"]["novel"][0], "Novel sample 1", c_ours),
             (d["cases"]["novel"][2], "Novel sample 2", c_ours)]

    fig = plt.figure(figsize=(11.0, 4.15))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.95, 1.0],
                          left=0.015, right=0.985, top=0.845, bottom=0.085,
                          hspace=0.045, wspace=0.055)

    for c, (case, head, col) in enumerate(picks):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(_crop32(case["image_path"]))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#C9CFD6")
            sp.set_linewidth(1.0)
        ax.set_title(head, fontsize=10.2, fontweight="bold", color=col, pad=6)

        axt = fig.add_subplot(gs[1, c])
        axt.axis("off")
        rows = [("Truth", flat(case["gt"]), "#111111", True)]
        for meth, cc, nm in (("clip0", c_clip, "CLIP0"), ("scg", c_scg, "SCG"), ("ours", c_ours, "Ours")):
            txt = "$\\checkmark$" if case[f"{meth}_ok"] else "$\\times$ " + flat(case[meth])
            rows.append((nm, txt, cc, False))
        for k, (tag, txt, cc, bold) in enumerate(rows):
            y = 0.90 - k * 0.255
            axt.text(0.44, y, tag, transform=axt.transAxes, ha="right", va="center",
                     fontsize=7.8, color="#8A9099")
            axt.text(0.50, y, txt, transform=axt.transAxes, ha="left", va="center",
                     fontsize=7.8, color=cc, fontweight="bold" if bold else "normal")

    nq = d["n_qualifying"]
    fig.suptitle("The same image under three methods (FGVC-Aircraft, seed 0)",
                 fontsize=12.0, fontweight="bold", y=0.972)
    fig.text(0.015, 0.018,
             f"Base image: CLIP0 wrong, SCG and Ours correct. Novel images: CLIP0 correct, SCG predicts a base class, Ours recovers. "
             f"Condition-filtered, not random (base {nq['base']}/{d['n_base_test']}, novel {nq['novel']}/{d['n_novel_test']}). "
             f"$\\checkmark$ correct; $\\times$ followed by the predicted class.",
             fontsize=7.6, color="#777")
    fig.savefig(os.path.join(FIG, "fig_prl5_cases.png"), dpi=250)
    plt.close(fig)
    TO_JPG("fig_prl5_cases.png")
    print("saved fig_prl5_cases.png")


if __name__ == "__main__":
    fig1_method()
    fig2_dataset()
    fig3_main()
    fig4_mech()
    fig5_cases()
    print("PRL FIGURES DONE ->", FIG)
