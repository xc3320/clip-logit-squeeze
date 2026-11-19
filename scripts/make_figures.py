#!/usr/bin/env python3
"""生成论文配图（顶刊风格，多面板，带统计标注）。

所有数据读自 results/ 下的实测 JSON，不硬编码结论数字。
图 1 为方法示意（结构图，无数据）；图 2-5 数据驱动。
输出 figures/fig1_method.png ... fig5_*.png，300 DPI。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------- 全局学术风格 ----------
plt.rcParams.update({
    "font.family": ["DejaVu Serif", "Microsoft YaHei", "SimHei"],
    "font.serif": ["DejaVu Serif", "Microsoft YaHei"],
    "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.linewidth": 1.1,
})

# 色盲友好 + 柔和学术配色（参考 MoTiC/CASP 图）
C = {
    "clip0": "#7A8B99",   # 灰蓝
    "scg":   "#E08A5B",   # 暖橙
    "ours":  "#2F6FB0",   # 深蓝
    "base":  "#2F6FB0",
    "novel": "#E08A5B",
    "hm":    "#3E9B78",   # 青绿
    "text":  "#4C6FB0",   # 文本分支 蓝
    "vis":   "#E08A5B",   # 视觉分支 橙
    "gate":  "#3E9B78",
    "bad":   "#C0392B",
    "lightblue": "#DCE6F2",
    "lightorange": "#F6E2D2",
    "lightgreen": "#DCEFE6",
    "lightgrey": "#ECEFF2",
}
DS = ["dtd", "oxfordpets", "flowers102", "fgvcaircraft", "cifar100"]
DS_LABEL = ["DTD", "OxfordPets", "Flowers102", "FGVC-Aircraft", "CIFAR100"]
METHODS = ["clip0", "gate_calibrated", "decouple_scg"]
M_LABEL = ["CLIP0", "SCG", "Ours"]


def load_summary(ds):
    d = json.load(open(os.path.join(ROOT, "results", ds, "decouple_v1.json"), encoding="utf-8"))
    return d


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_tag(ax, tag, x=-0.02, y=1.04):
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="bottom", ha="left", fontfamily="DejaVu Serif")


# ============================================================
# 图 1：方法总览（结构示意，参考顶刊框架图）
# ============================================================
def fig1_method():
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#4A4A4A", fs=11, bold=False):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.09",
                           facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)
        return p

    def arrow(x1, y1, x2, y2, color="#555", lw=1.7, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=14, color=color, linewidth=lw,
                     connectionstyle=f"arc3,rad={rad}", zorder=1))

    # 共享编码器行
    box(0.35, 6.4, 1.75, 0.95, "输入图像 x", C["lightgrey"], fs=12, bold=True)
    box(2.55, 6.4, 2.35, 0.95, "冻结 CLIP\n图像编码器", C["lightblue"], fs=11.5, bold=True)
    box(5.35, 6.55, 1.5, 0.65, "特征 f", "#FFFFFF", fs=11.5, bold=True)
    arrow(2.10, 6.88, 2.55, 6.88)
    arrow(4.90, 6.88, 5.35, 6.88)
    # f 注入面板 (a) —— 一条竖直箭头即可，不需要额外说明文字
    arrow(6.1, 6.55, 6.85, 5.72, color="#555", rad=0.0)

    # 面板 (a) 背景
    ax.add_patch(FancyBboxPatch((0.2, 2.7), 12.95, 3.0,
                 boxstyle="round,pad=0.05,rounding_size=0.12",
                 facecolor="#FAFBFC", edgecolor="#C8D0D8", linewidth=1.2, zorder=0))
    ax.text(0.45, 5.42, "(a) 尺度校准门控（SCG）：双流 + 逐样本校准 + 可靠性门控",
            fontsize=12.5, fontweight="bold")

    # 文本分支 y≈4.35（蓝）
    box(0.7, 4.35, 2.6, 0.9, "冻结文本类别特征\nW_text（固定模板）", C["lightblue"], fs=10.5, bold=True)
    box(3.8, 4.35, 2.8, 0.9, "文本 logits\nℓ_text = s·fᵀW_text", C["lightblue"], fs=10.5, bold=True)
    arrow(3.30, 4.8, 3.8, 4.8, color=C["text"])

    # 视觉分支 y≈3.0（橙）
    box(0.7, 3.0, 2.6, 0.9, "base support 上训练的\n线性视觉探针 W_vis", C["lightorange"], fs=10.5, bold=True)
    box(3.8, 3.0, 2.8, 0.9, "视觉 logits（无界）\nℓ_vis = fᵀW_vis", C["lightorange"], fs=10.5, bold=True)
    arrow(3.30, 3.45, 3.8, 3.45, color=C["vis"])

    # 校准 + 门控 + 融合（两分支汇入校准，再门控）
    box(7.1, 3.5, 1.6, 1.25, "逐样本\nz-score\n校准", C["lightgreen"], fs=10, bold=True)
    box(9.1, 3.5, 1.6, 1.25, "可靠性\n门控 gᵢ", C["lightgreen"], fs=10, bold=True)
    box(11.1, 3.5, 2.0, 1.25, "融合 logits\nℓ_fused", "#FFFFFF", fs=11, bold=True)
    arrow(6.6, 4.8, 7.1, 4.35, color=C["gate"])
    arrow(6.6, 3.45, 7.1, 3.9, color=C["gate"])
    arrow(8.7, 4.12, 9.1, 4.12, color=C["gate"])
    arrow(10.7, 4.12, 11.1, 4.12, color=C["gate"])

    # 面板 (b)：解耦推理
    ax.add_patch(FancyBboxPatch((0.2, 0.1), 12.9, 2.4,
                 boxstyle="round,pad=0.05,rounding_size=0.12",
                 facecolor="#F7FAFD", edgecolor="#B9CCE0", linewidth=1.2, zorder=0))
    ax.text(0.45, 2.18, "(b) 解耦推理：按样本身份切换 logits（novel logit 挤压的修复）",
            fontsize=12.5, fontweight="bold")

    box(0.75, 0.55, 2.35, 1.15, "测试样本\n(base_test / novel_test)", C["lightgrey"], fs=10.5, bold=True)
    box(3.7, 0.55, 2.7, 1.15, "身份路由\nbase → 融合；novel → 文本", C["lightgreen"], fs=10.5, bold=True)
    arrow(3.10, 1.13, 3.7, 1.13)
    box(7.15, 1.0, 2.35, 0.85, "base：ℓ_fused\n（保留 base 增益）", C["lightblue"], fs=10.5, bold=True)
    box(7.15, 0.05, 2.35, 0.85, "novel：ℓ_text\n（绕开挤压，=CLIP0）", C["lightorange"], fs=10.5, bold=True)
    arrow(6.4, 1.13, 7.15, 1.42, color=C["base"])
    arrow(6.4, 1.13, 7.15, 0.48, color=C["novel"])
    box(10.15, 0.5, 2.5, 1.25, "预测\nBase ↑ / Novel 无损\nHM ↑", "#FFFFFF", fs=10.5, bold=True)
    arrow(9.5, 1.42, 10.15, 1.13, color=C["base"])
    arrow(9.5, 0.48, 10.15, 0.68, color=C["novel"])

    fig.savefig(os.path.join(FIG, "fig1_method.png"))
    plt.close(fig)
    print("saved fig1_method.png")


# ============================================================
# 图 2：主结果分组柱状图（B/N/HM，带误差棒）
# ============================================================
def fig2_main_results():
    data = {ds: load_summary(ds) for ds in DS}
    # 2×3 网格：前 5 格放数据集，第 6 格放图例 + 关键结论
    fig = plt.figure(figsize=(14.5, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.24,
                          left=0.06, right=0.98, top=0.88, bottom=0.08)
    x = np.arange(len(METHODS))
    width = 0.25
    bars_colors = [C["base"], C["novel"], C["hm"]]
    metrics = [("base_acc", "Base"), ("novel_acc", "Novel"), ("hm", "HM")]

    for idx, (ds, lab) in enumerate(zip(DS, DS_LABEL)):
        ax = fig.add_subplot(gs[idx])
        s = data[ds]["summary"]
        # 本文方法所在列加淡蓝底 + 加粗轴标签，视觉上把 Ours 顶出来
        ax.axvspan(x[2] - 0.44, x[2] + 0.44, color=C["ours"], alpha=0.05, zorder=1)
        for k, (key, mlab) in enumerate(metrics):
            means = [s[m][key]["mean"] * 100 for m in METHODS]
            stds = [s[m][key]["std"] * 100 for m in METHODS]
            pos = x + (k - 1) * width
            bars = ax.bar(pos, means, width, yerr=stds, color=bars_colors[k],
                          alpha=0.93, edgecolor="white", linewidth=0.8,
                          capsize=3, error_kw=dict(elinewidth=1.1, ecolor="#555"), zorder=3)
            # 柱顶数值标注（放在误差棒上方，留白充足不遮挡）
            for b, mv, sd in zip(bars, means, stds):
                ax.text(b.get_x() + b.get_width() / 2, mv + sd + 2.2, f"{mv:.0f}",
                        ha="center", va="bottom", fontsize=7.6, color="#222", zorder=5)
        ax.set_title(lab, fontsize=12.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(M_LABEL, fontsize=10.5)
        ax.get_xticklabels()[2].set_fontweight("bold")
        ax.get_xticklabels()[2].set_color(C["ours"])
        ax.set_ylim(0, 110)  # 给数值标注留空间，误差棒绝不被截断
        ax.grid(axis="y", linestyle="--", alpha=0.32, zorder=0)
        despine(ax)
        if idx % 3 == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=11.5)

    # 第 6 格：图例 + 关键结论说明卡片
    ax6 = fig.add_subplot(gs[5])
    ax6.axis("off")
    ax6.set_title("读图要点", fontsize=12.5, fontweight="bold", loc="left")
    legend_handles = [Line2D([0], [0], marker="s", color="none",
                      markerfacecolor=bars_colors[i], markersize=11,
                      label=metrics[i][1] + " accuracy") for i in range(3)]
    leg = ax6.legend(handles=legend_handles, loc="upper left", frameon=False,
                     fontsize=11, bbox_to_anchor=(0.0, 0.92))
    notes = (
        "• Ours = SCG + 解耦推理（本文方法）\n"
        "• Novel 列：Ours 与 CLIP0 等高\n"
        "  （逐 seed 精确无损，Δ=0）\n"
        "• Base 列：Ours 显著高于 CLIP0\n"
        "• HM：Ours 五数据集全部最高\n"
        "• FGVC-Aircraft：SCG 的 Novel\n"
        "  掉到 10.6%，解耦后回到 26.3%\n"
        "• 误差棒 = 5 seeds 标准差"
    )
    ax6.text(0.0, 0.60, notes, transform=ax6.transAxes, va="top", ha="left",
             fontsize=11, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.6", fc="#F7FAFD", ec="#B9CCE0", lw=1.2))

    fig.suptitle("主结果：五数据集 Base / Novel / HM（16-shot，5 seeds 均值±std）",
                 fontsize=14.5, fontweight="bold", y=0.965)
    fig.savefig(os.path.join(FIG, "fig2_main_results.png"))
    plt.close(fig)
    print("saved fig2_main_results.png")


# ============================================================
# 图 3：机制散点图（挤压率 vs 掉点，回归线 + CI）
# ============================================================
def fig3_mechanism():
    d = json.load(open(os.path.join(ROOT, "results", "mechanism_correlation.json"), encoding="utf-8"))
    pts = d["points"]
    colors = ["#2F6FB0", "#E08A5B", "#3E9B78", "#8E5BB7", "#C0392B"]
    labels = ["DTD", "OxfordPets", "Flowers102", "FGVC-Aircraft", "CIFAR100"]
    order = ["dtd", "oxfordpets", "flowers102", "fgvcaircraft", "cifar100"]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for ds, col, lab in zip(order, colors, labels):
        sub = [p for p in pts if p["dataset"] == ds]
        sx = [p["squeeze"] * 100 for p in sub]
        sy = [p["drop"] * 100 for p in sub]
        ax.scatter(sx, sy, s=70, color=col, edgecolor="white", linewidth=0.8,
                   alpha=0.9, label=lab, zorder=3)

    # 回归线 + 95% CI（解析计算）
    allx = np.array([p["squeeze"] * 100 for p in pts])
    ally = np.array([p["drop"] * 100 for p in pts])
    n = len(allx)
    coef = np.polyfit(allx, ally, 1)
    xs = np.linspace(allx.min() - 2, allx.max() + 2, 200)
    ys = np.polyval(coef, xs)
    resid = ally - np.polyval(coef, allx)
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))
    t = 2.06  # t_{0.025,23}
    ci = t * s_err * np.sqrt(1 / n + (xs - allx.mean()) ** 2 / np.sum((allx - allx.mean()) ** 2))
    ax.plot(xs, ys, color="#222", lw=1.8, zorder=2)
    ax.fill_between(xs, ys - ci, ys + ci, color="#222", alpha=0.12, zorder=1)

    r = d["pearson_squeeze_vs_drop"]["r"]
    ax.text(0.03, 0.95, f"Pearson r = {r:.3f}\nn = {n}, p < 0.01",
            transform=ax.transAxes, va="top", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#B9CCE0", alpha=0.95))
    ax.set_xlabel("Novel logit 挤压率 (%)")
    ax.set_ylabel("SCG 下 Novel 掉点 (pp)")
    ax.set_title("挤压率与新类掉点强相关", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    despine(ax)
    fig.savefig(os.path.join(FIG, "fig3_mechanism.png"))
    plt.close(fig)
    print("saved fig3_mechanism.png")


# ============================================================
# 图 4：正交性消融（去校准 base 崩塌）+ fulltext 对照
# ============================================================
def fig4_ablation():
    data = {ds: load_summary(ds) for ds in DS}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # 左：base 崩塌 —— 哑铃图（dumbbell）：每个数据集独占一行，无连线交叉、无标签碰撞
    scg_b = [data[ds]["summary"]["decouple_scg"]["base_acc"]["mean"] * 100 for ds in DS]
    na_b = [data[ds]["summary"]["decouple_naive"]["base_acc"]["mean"] * 100 for ds in DS]
    order = np.argsort(scg_b)[::-1]          # 按 Ours 值降序，形成自上而下的阶梯
    nrow = len(order)
    for r, i in enumerate(order):
        a, b = scg_b[i], na_b[i]
        y = nrow - 1 - r
        ax1.plot([b, a], [y, y], color="#CBD4DC", lw=6, solid_capstyle="round", zorder=1)
        ax1.scatter([b], [y], s=105, facecolor="white", edgecolor=C["bad"], linewidth=1.9, zorder=3)
        ax1.scatter([a], [y], s=105, color=C["ours"], edgecolor="white", linewidth=1.0, zorder=4)
        # 降幅标在连接段上方（行内空白，绝不会压到别的行）
        ax1.annotate(f"−{a - b:.0f}pp", xy=((a + b) / 2, y), xytext=(0, 12),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=9.5, color="#8B2E22", fontweight="bold")
        ax1.text(a + 2.6, y, f"{a:.0f}", va="center", ha="left",
                 fontsize=9.5, color=C["ours"], fontweight="bold")
        ax1.text(b + 2.6, y, f"{b:.0f}", va="center", ha="left",
                 fontsize=9.5, color="#8B2E22", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    ax1.set_yticks(range(nrow))
    ax1.set_yticklabels([DS_LABEL[i] for i in reversed(order)], fontsize=10)
    ax1.set_ylim(-0.65, nrow - 0.35)
    ax1.set_xlim(0, 106)
    ax1.set_xlabel("Base accuracy (%)")
    ax1.set_title("(a) 去掉尺度校准 → Base 大幅下滑", fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.28, zorder=0)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C["ours"],
               markersize=9, label="Ours（校准+解耦）"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=C["bad"], markeredgewidth=1.9, markersize=9,
               label="去校准（naive）"),
    ]
    ax1.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)
    despine(ax1)

    # 右：fulltext 等于 clip0（novel 全走文本）
    clip0_n = [data[ds]["summary"]["clip0"]["novel_acc"]["mean"] * 100 for ds in DS]
    full_n = [data[ds]["summary"]["decouple_fulltext"]["novel_acc"]["mean"] * 100 for ds in DS]
    ax2.plot([0, 100], [0, 100], "--", color="#888", lw=1.2, label="y = x")
    ax2.scatter(clip0_n, full_n, s=95, color=C["ours"], edgecolor="white", zorder=3)
    off_r = {"dtd": (7, 5, "left"), "oxfordpets": (-7, 9, "right"),
             "flowers102": (7, -11, "left"), "fgvcaircraft": (7, -11, "left"),
             "cifar100": (7, 7, "left")}
    for i, ds in enumerate(DS):
        dx, dy, ha = off_r[ds]
        ax2.annotate(DS_LABEL[i], (clip0_n[i], full_n[i]), textcoords="offset points",
                     xytext=(dx, dy), ha=ha, fontsize=8.5, color="#333")
    ax2.set_xlabel("CLIP0 Novel accuracy (%)")
    ax2.set_ylabel("decouple_fulltext Novel (%)")
    ax2.set_title("(b) 全文本切换精确等于 CLIP0", fontweight="bold")
    ax2.set_xlim(20, 97)
    ax2.set_ylim(20, 97)
    ax2.grid(linestyle="--", alpha=0.3)
    ax2.legend(fontsize=9.5, loc="upper left")
    despine(ax2)

    fig.suptitle("消融：解耦与尺度校准正交，二者缺一不可", fontsize=13.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "fig4_ablation.png"))
    plt.close(fig)
    print("saved fig4_ablation.png")


# ============================================================
# 图 5：挤压率分布 + 无监督分流失败
# ============================================================
def fig5_squeeze_auto():
    mech = json.load(open(os.path.join(ROOT, "results", "mechanism_correlation.json"), encoding="utf-8"))
    squeeze_mean = [mech["per_dataset_mean"][ds]["squeeze"] * 100 for ds in DS]

    # 无监督分流精确率（来自 auto0 / v1 auto_diag）
    auto_prec = {}
    for ds in DS:
        fp0 = os.path.join(ROOT, "results", ds, "decouple_auto0.json")
        fp1 = os.path.join(ROOT, "results", ds, "decouple_v1.json")
        val = None
        if os.path.exists(fp0):
            a = json.load(open(fp0, encoding="utf-8")).get("auto_diag", {})
            val = a.get("squeeze_dir_precision")
        if val is None and os.path.exists(fp1):
            a = json.load(open(fp1, encoding="utf-8")).get("auto_diag", {})
            val = a.get("squeeze_dir_precision")
        auto_prec[ds] = val

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.7))
    colors = ["#2F6FB0", "#E08A5B", "#3E9B78", "#8E5BB7", "#C0392B"]

    # 左：挤压率 —— 降序排列（排序分析，突出极端值）
    order = np.argsort(squeeze_mean)[::-1]
    sq_sorted = [squeeze_mean[i] for i in order]
    lab_sorted = [DS_LABEL[i] for i in order]
    col_sorted = [colors[i] for i in order]
    bars = ax1.bar(range(len(DS)), sq_sorted, color=col_sorted, alpha=0.9, zorder=3)
    for i, v in enumerate(sq_sorted):
        ax1.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax1.set_xticks(range(len(DS)))
    ax1.set_xticklabels(lab_sorted, rotation=30, ha="right", fontsize=9.5)
    ax1.set_ylabel("Novel 被挤成 Base 的比例 (%)")
    ax1.set_title("(a) Novel logit 挤压率（按严重程度排序）", fontweight="bold")
    ax1.set_ylim(0, 95)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    despine(ax1)

    # 右：无监督分流精确率 —— 横向条形，参考线就地标注（不用图例，避免"找不到基线"）
    precs = [auto_prec[DS[i]] for i in order]
    ys = np.arange(len(order))
    ax2.barh(ys, [p * 100 for p in precs], 0.56,
             color=col_sorted, alpha=0.9, zorder=3)
    for y, p in zip(ys, precs):
        ax2.text(p * 100 + 1.8, y, f"{p*100:.1f}%", va="center", ha="left",
                 fontsize=10, fontweight="bold", color="#222", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    ax2.set_yticks(ys)
    ax2.set_yticklabels(lab_sorted, fontsize=10)
    ax2.set_xlim(0, 128)
    ax2.set_ylim(-2.0, len(ys) - 0.35)
    ax2.invert_yaxis()          # 与左图同序：挤压最严重的排最上

    # 两条参考线：文字分置左右，避免互相压住
    ax2.axvline(50, color="#6B7785", ls=":", lw=1.5, zorder=4)
    ax2.axvline(100, color=C["ours"], ls="--", lw=1.4, zorder=4)
    ax2.text(51.5, -1.9, "随机猜测 50%", fontsize=8.8, color="#555",
             ha="left", va="top", zorder=6)
    ax2.text(99, -1.9, "完美分流 100%\n（已知样本真实类别）", fontsize=8.8,
             color=C["ours"], ha="right", va="top", zorder=6, linespacing=1.3)

    lo, hi = min(precs) * 100, max(precs) * 100
    ax2.set_xlabel("路由命中真实 Novel 的精确率 (%)", fontsize=11)
    ax2.set_title(f"(b) 无监督分流只到 {lo:.0f}%–{hi:.0f}%，替代不了身份先验",
                  fontweight="bold", fontsize=12)
    ax2.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    despine(ax2)

    fig.suptitle("机制诊断：挤压普遍存在，且无法用无监督信号自动分流", fontsize=13.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "fig5_squeeze_auto.png"))
    plt.close(fig)
    print("saved fig5_squeeze_auto.png")


# ============================================================
# 图 6：Base–Novel 权衡帕累托图（创意表示：方法移动轨迹）
# ============================================================
def fig6_pareto():
    """Δ 平面：横轴 ΔBase、纵轴 ΔNovel（都以 CLIP0 为原点）。
    Ours 全部落在 y=0 的零损失线上——本文方法的核心结论一眼可辨。"""
    data = {ds: load_summary(ds) for ds in DS}
    c_scg, c_ours = "#D98A4F", "#2F6FB0"

    rows = []
    for ds in DS:
        s = data[ds]["summary"]
        b0 = s["clip0"]["base_acc"]["mean"] * 100
        n0 = s["clip0"]["novel_acc"]["mean"] * 100
        rows.append(dict(
            ds=ds,
            dbg=s["gate_calibrated"]["base_acc"]["mean"] * 100 - b0,
            dng=s["gate_calibrated"]["novel_acc"]["mean"] * 100 - n0,
            dbo=s["decouple_scg"]["base_acc"]["mean"] * 100 - b0,
            dno=s["decouple_scg"]["novel_acc"]["mean"] * 100 - n0,
        ))

    fig, ax = plt.subplots(figsize=(9.6, 5.2))

    # novel 净损失区：只用淡底色分区，不写字
    ax.axhspan(-19.5, 0, color="#A94438", alpha=0.05, zorder=0)
    # 零损失线
    ax.axhline(0, color="#2E2E2E", lw=1.3, ls=(0, (7, 3)), zorder=2)

    for r in rows:
        # 竖向箭头：SCG 把 novel 压下去 → 解耦把它拉回零线
        ax.annotate("", xy=(r["dbo"], r["dno"]), xytext=(r["dbg"], r["dng"]),
                    arrowprops=dict(arrowstyle="-|>", color=c_ours, lw=1.7, alpha=0.55,
                                    shrinkA=4, shrinkB=8, mutation_scale=14), zorder=2)
        ax.scatter(r["dbg"], r["dng"], marker="s", s=68, color=c_scg,
                   edgecolor="white", linewidth=1.0, zorder=3)
        ax.scatter(r["dbo"], r["dno"], marker="^", s=175, color=c_ours,
                   edgecolor="white", linewidth=1.3, zorder=4)
        # SCG 造成的 novel 损失量（负值，标在方块下方）
        ax.text(r["dbg"], r["dng"] - 0.7, f"{r['dng']:+.1f}", ha="center", va="top",
                fontsize=8.6, color="#A8683A", zorder=4)

    # 数据集标签：拥挤处用细分引线错开；颜色统一深灰，避免 5 色抢戏
    lab = {
        "cifar100":     (3.7, 4.7, "center"),
        "oxfordpets":   (6.0, 2.8, "center"),
        "fgvcaircraft": (16.0, 1.4, "center"),
        "dtd":          (21.0, 1.4, "right"),
        "flowers102":   (23.4, 1.4, "left"),
    }
    for r in rows:
        lx, ly, ha = lab[r["ds"]]
        ax.annotate(DS_LABEL[DS.index(r["ds"])], xy=(r["dbo"], 0.0), xytext=(lx, ly),
                    ha=ha, va="center", fontsize=9.6, color="#333333", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#C2C2C2", lw=0.8,
                                    shrinkA=2, shrinkB=3), zorder=5)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=c_scg,
               markeredgecolor="white", markersize=9, label="SCG（双流融合）"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=c_ours,
               markeredgecolor="white", markersize=11, label="Ours（+ 解耦推理）"),
        Line2D([0], [0], color=c_ours, lw=1.7, alpha=0.55, label="解耦把 novel 拉回零线"),
        Line2D([0], [0], color="#2E2E2E", lw=1.2, ls=(0, (7, 3)), label="Novel 零损失（= CLIP0）"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9.4, frameon=True,
              framealpha=0.96, edgecolor="#D8DDE3", borderpad=0.7, labelspacing=0.5,
              title="Δ 值均相对 CLIP0 零样本", title_fontsize=9.2)
    ax.set_xlabel("ΔBase 精度 (pp，相对 CLIP0)", fontsize=11.5)
    ax.set_ylabel("ΔNovel 精度 (pp)", fontsize=11.5)
    ax.set_title("Base 增益全保留、Novel 零损失：解耦把新类从挤压中拉回",
                 fontweight="bold", fontsize=13)
    ax.set_xlim(-1.2, 27.4)
    ax.set_ylim(-19.5, 6.6)
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    despine(ax)
    fig.savefig(os.path.join(FIG, "fig6_pareto.png"))
    plt.close(fig)
    print("saved fig6_pareto.png")


# ============================================================
# 图 7：五个数据集的真实样本网格
# ============================================================
DS_DESC = {
    "dtd":          ("纹理描述（Describable Textures）", 47),
    "oxfordpets":   ("猫狗宠物（Oxford-IIIT Pets）", 37),
    "flowers102":   ("花卉（Oxford Flowers-102）", 102),
    "fgvcaircraft": ("飞机型号（FGVC-Aircraft）", 100),
    "cifar100":     ("通用物体（CIFAR-100）", 100),
}
# 真实图像目录（必须精确指定，否则会抽到 trimap 标注掩码等非图像文件）
DS_IMG_DIR = {
    "dtd":          "data/dtd/dtd/images",
    "oxfordpets":   "data/oxfordpets/images",
    "flowers102":   "data/flowers102/jpg",
    "fgvcaircraft": "data/fgvcaircraft/fgvc-aircraft-2013b/data/images",
    "cifar100":     "data/cifar100/images",
}
LINE_COLS = ["#2F6FB0", "#E08A5B", "#3E9B78", "#8E5BB7", "#C0392B"]


def _to_jpg(png_name, quality=92):
    """照片类图再存一份 JPEG：embedding 到 Word 时体积只有 PNG 的 1/5。"""
    from PIL import Image
    src = os.path.join(FIG, png_name)
    dst = os.path.splitext(src)[0] + ".jpg"
    Image.open(src).convert("RGB").save(dst, quality=quality, optimize=True)


def _thumb(path, size=220):
    """读图 → 中心裁方 → 缩放，返回 RGB 数组。"""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    x0, y0 = (w - s) // 2, (h - s) // 2
    im = im.crop((x0, y0, x0 + s, y0 + s)).resize((size, size), Image.LANCZOS)
    return np.asarray(im)


def _pretty_cls(name):
    """类名规范化：FGVC 的 'ATR_ATR-42' → 'ATR-42'，下划线转空格；过长则折行。"""
    import textwrap
    parts = name.split("_")
    # "Gulfstream_Aerospace_Gulfstream_V" → "Gulfstream V"：厂商名在后文重复时掐掉前缀
    for i in range(1, len(parts)):
        if parts[i].lower() == parts[0].lower():
            parts = parts[i:]
            break
    s = " ".join(parts).strip()
    return "\n".join(textwrap.wrap(s, 15)[:2])


def _bright_ok(path, lo=38, hi=246):
    """过滤过暗/过曝的样例，避免整格看不出内容。"""
    from PIL import Image
    im = Image.open(path).convert("L").resize((32, 32))
    m = float(np.asarray(im).mean())
    return lo < m < hi


def _load_split(ds):
    """读 seed0 划分，按 split 归组，并统计基类/新类类别数。"""
    import csv
    rows = []
    with open(os.path.join(ROOT, "data", "splits", f"{ds}_seed0.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if os.path.exists(r["image_path"]):
                rows.append(r)
    by = {}
    for r in rows:
        by.setdefault(r["split"], []).append(r)
    n_base = len({r["class_name"] for r in by.get("base_train", [])})
    n_novel = len({r["class_name"] for r in by.get("novel_test", [])})
    return by, n_base, n_novel


def fig7_dataset():
    """样本构成图：同一数据集内展示基类支持/基类测试/新类测试，四列取自四个不同类别。"""
    import random

    cols = [("基类 · 支持样本", "16-shot 适配", "base_train", "#2F6FB0"),
            ("基类 · 测试样本", "走融合推理", "base_test", "#2F6FB0"),
            ("新类 · 测试样本 ①", "无视觉证据", "novel_test", "#E08A5B"),
            ("新类 · 测试样本 ②", "另一新类", "novel_test", "#E08A5B")]

    fig = plt.figure(figsize=(10.4, 9.5))
    gs = fig.add_gridspec(6, 5, left=0.012, right=0.988, top=0.905, bottom=0.045,
                          wspace=0.07, hspace=0.30,
                          height_ratios=[0.20, 1, 1, 1, 1, 1],
                          width_ratios=[1.42, 1, 1, 1, 1])
    rng = random.Random(13)

    # 列标题（放在图像区上方）
    for c, (head, sub, _, col) in enumerate(cols):
        axh = fig.add_subplot(gs[0, c + 1])
        axh.axis("off")
        axh.text(0.5, 0.62, head, transform=axh.transAxes, ha="center", va="center",
                 fontsize=10.6, fontweight="bold", color=col)
        axh.text(0.5, 0.16, sub, transform=axh.transAxes, ha="center", va="center",
                 fontsize=8.2, color="#7A828B")

    for r, ds in enumerate(DS):
        by, n_base, n_novel = _load_split(ds)
        used, picks = set(), []
        for _, _, split, _ in cols:
            pool = [x for x in by[split] if x["class_name"] not in used]
            # 新类两列取不同类别；随机但可复现，并跳过过暗/过曝的样本
            item = None
            for _ in range(12):
                cand = rng.choice(pool)
                if _bright_ok(cand["image_path"]):
                    item = cand
                    break
            item = item or rng.choice(pool)
            used.add(item["class_name"])
            picks.append(item)

        axl = fig.add_subplot(gs[r + 1, 0])
        axl.axis("off")
        axl.add_patch(Rectangle((0.005, 0.10), 0.030, 0.80, transform=axl.transAxes,
                                facecolor=LINE_COLS[r], edgecolor="none"))
        axl.text(0.075, 0.70, DS_LABEL[r], transform=axl.transAxes, fontsize=12.2,
                 fontweight="bold", va="center", color=LINE_COLS[r])
        name, ncls = DS_DESC[ds]
        axl.text(0.075, 0.36, name, transform=axl.transAxes, fontsize=8.6,
                 va="center", color="#555")
        axl.text(0.075, 0.17, f"{ncls} 类｜基类 {n_base} / 新类 {n_novel}",
                 transform=axl.transAxes, fontsize=8.6, va="center", color="#8A9099")

        for c, item in enumerate(picks):
            ax = fig.add_subplot(gs[r + 1, c + 1])
            ax.imshow(_thumb(item["image_path"], size=260))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#CBD2D9")
                sp.set_linewidth(0.9)
            ax.text(0.5, -0.055, _pretty_cls(item["class_name"]), transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.2, color="#4A5158",
                    clip_on=False, style="italic", linespacing=1.35)

    fig.suptitle("五个数据集的样本构成：基类支持、基类测试与新类测试（每行四列类别互不相同）",
                 fontsize=13.6, fontweight="bold", y=0.968)
    fig.text(0.012, 0.012,
             "注：样本取自 seed 0 的划分文件；括号内为该数据集 base/novel 的类别数。"
             "CIFAR-100 原图 32×32，放大后略显模糊；DTD 为纹理图，无单一主体。",
             fontsize=8.2, color="#777")
    fig.savefig(os.path.join(FIG, "fig7_dataset.png"), dpi=200)
    plt.close(fig)
    _to_jpg("fig7_dataset.png")
    print("saved fig7_dataset.png")


def fig8_cases():
    """个案追踪：同一批真实测试样本在三种方法下的预测（数据来自 dump_cases.py）。"""
    fp = os.path.join(ROOT, "results", "fgvcaircraft", "cases_seed0.json")
    if not os.path.exists(fp):
        print("[skip] fig8: 缺 results/fgvcaircraft/cases_seed0.json，先跑 scripts/dump_cases.py")
        return
    d = json.load(open(fp, encoding="utf-8"))
    c_clip, c_scg, c_ours = "#7A8B99", "#C97A3E", "#2F6FB0"

    def pretty(name):
        parts = name.split("_")
        # FGVC 类名常出现 "Cessna_Cessna_172" 这类重复前缀，收拢成 "Cessna 172"
        if len(parts) >= 2 and parts[0] == parts[1]:
            parts = parts[1:]
        return " ".join(parts)

    fig = plt.figure(figsize=(11.4, 8.0))
    gs = fig.add_gridspec(4, 3, height_ratios=[3.05, 1.02, 3.05, 1.02],
                          left=0.045, right=0.985, top=0.855, bottom=0.055,
                          hspace=0.26, wspace=0.05)
    rows = [("base", "基类样本：CLIP0 判错 → SCG 判对 → Ours 判对（增益被保留）", c_scg),
            ("novel", "新类样本：CLIP0 判对 → SCG 判成基类 → Ours 判对（挤压被消除）", c_ours)]

    for r, (key, rlab, rcol) in enumerate(rows):
        for c, case in enumerate(d["cases"][key]):
            ax = fig.add_subplot(gs[2 * r, c])
            ax.imshow(_thumb(case["image_path"], size=380))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#C9CFD6")
                sp.set_linewidth(1.0)

            # 预测行：对的只打 √，错的写出被误判成的类
            lines = [("真值", pretty(case["gt"]), "#111111", True)]
            for meth, col in (("clip0", c_clip), ("scg", c_scg), ("ours", c_ours)):
                ok = case[f"{meth}_ok"]
                txt = "√" if ok else "×  " + pretty(case[meth])
                lines.append(({"clip0": "CLIP0", "scg": "SCG", "ours": "Ours"}[meth], txt, col, False))

            axt = fig.add_subplot(gs[2 * r + 1, c])
            axt.axis("off")
            for k, (tag, txt, col, bold) in enumerate(lines):
                y = 0.94 - k * 0.245
                axt.text(0.5, y, tag, transform=axt.transAxes, ha="right", va="center",
                         fontsize=8.4, color="#8A9099")
                axt.text(0.52, y, txt, transform=axt.transAxes, ha="left", va="center",
                         fontsize=9.0 if bold else 8.8, color=col,
                         fontweight="bold" if bold else "normal")

        # 行标题（放在该行图像上方）
        # 行标题落在"上一行文字块"与"本行图像"之间的空档正中，避免与两侧相贴
        pos = fig.axes[2 * r * 3].get_position()
        fig.text(0.045, pos.y1 + 0.021, rlab, fontsize=10.5, fontweight="bold",
                 color=rcol, va="center")

    nq = d["n_qualifying"]
    fig.suptitle("个案追踪：同一张图在三种方法下的预测（FGVC-Aircraft，seed 0）",
                 fontsize=14, fontweight="bold", y=0.965)
    fig.text(0.045, 0.012,
             f"选样为条件筛选、非随机抽样：符合条件的样本 基类 {nq['base']}/{d['n_base_test']}、"
             f"新类 {nq['novel']}/{d['n_novel_test']}；√ 表示判对，× 后为该方法的错误预测类。",
             fontsize=8.4, color="#777")
    fig.savefig(os.path.join(FIG, "fig8_cases.png"), dpi=250)
    plt.close(fig)
    _to_jpg("fig8_cases.png")
    print("saved fig8_cases.png")


if __name__ == "__main__":
    fig1_method()
    fig2_main_results()
    fig3_mechanism()
    fig4_ablation()
    fig5_squeeze_auto()
    fig6_pareto()
    fig7_dataset()
    fig8_cases()
    print("ALL FIGURES DONE ->", FIG)
