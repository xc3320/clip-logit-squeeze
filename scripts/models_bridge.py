#!/usr/bin/env python3
"""Text→Visual Bridge（文本到视觉桥接）。

问题背景：SCG(base-to-novel) 的 novel 掉点根因是 novel 类没有 support 样本，
视觉分支对 novel 填 0，只能靠文本；base 类有视觉证据，softmax 竞争里 novel 被挤掉。

本模块思路：base 类同时有真实视觉原型 v_i（support 特征均值）和文本特征 t_i。
学一个仿射映射 M: t -> v（text 空间映射到 visual 空间），把它作用到 novel 的
文本特征上，生成 novel 的"伪视觉原型"。这样 novel 在视觉分支也有证据，
base/novel 的证据结构对称，门控融合不再系统性偏向 base。

**可证伪性（最重要）**：
  base 类数(18~23) << CLIP 维度(512)，D×D 映射极易记忆 base、对 novel 无泛化。
  因此必须用「留一类交叉验证」测量映射泛化余弦：留出一类，用其余 base 类拟合，
  预测该类的伪视觉原型，与真实原型比余弦。若留一余弦显著低于 in-sample，
  说明映射只是记忆，novel 伪原型不可信 —— 如实报告，不硬套。

全部为闭式岭回归，无迭代训练，冻结 CLIP，显存开销可忽略。
"""
import torch
import torch.nn.functional as F


def class_vis_prototypes(feats, labels, c_base):
    """base 各类真实视觉原型 = support 特征均值（归一化）[C_base, D]。"""
    d = feats.shape[1]
    protos = torch.zeros(c_base, d, device=feats.device, dtype=feats.dtype)
    for i in range(c_base):
        mask = labels == i
        if mask.any():
            protos[i] = F.normalize(feats[mask].mean(0, keepdim=True)).squeeze(0)
    return protos


def _fit_ridge(text_x, vis_y, ridge):
    """拟合仿射映射 [text,1] -> vis。返回增广权重 W [D+1, D]。"""
    ones = torch.ones(text_x.shape[0], 1, device=text_x.device, dtype=text_x.dtype)
    X = torch.cat([text_x, ones], dim=1)              # [C, D+1]
    A = X.T @ X + ridge * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(A, X.T @ vis_y)            # [D+1, D]
    return W


def _apply(W, text_x):
    ones = torch.ones(text_x.shape[0], 1, device=text_x.device, dtype=text_x.dtype)
    return torch.cat([text_x, ones], dim=1) @ W       # [N, D]


def bridge_diagnostics(text_base, vis_base, ridge=1e3):
    """留一类交叉验证 + in-sample，测 text→visual 映射是否真有泛化能力。

    返回 dict，全部是标量（余弦）。diagnostics 与映射是否可用直接挂钩。
    """
    c = text_base.shape[0]
    # in-sample 拟合余弦（乐观上界）
    W_in = _fit_ridge(text_base, vis_base, ridge)
    cos_in = F.cosine_similarity(_apply(W_in, text_base), vis_base, dim=-1).mean().item()
    # 留一类：用其余 c-1 类拟合，预测留出类
    loo = []
    for i in range(c):
        keep = [j for j in range(c) if j != i]
        W = _fit_ridge(text_base[keep], vis_base[keep], ridge)
        pred = _apply(W, text_base[i:i + 1])
        loo.append(F.cosine_similarity(pred, vis_base[i:i + 1], dim=-1).item())
    cos_loo = sum(loo) / len(loo)
    # 平凡基线：文本特征与视觉原型的直接余弦（无映射）
    cos_raw = F.cosine_similarity(text_base, vis_base, dim=-1).mean().item()
    return {
        "ridge": ridge,
        "cos_insample": round(cos_in, 4),
        "cos_leave_one_class_out": round(cos_loo, 4),
        "cos_raw_no_mapping": round(cos_raw, 4),
        "cos_loo_min": round(min(loo), 4),
        "cos_loo_max": round(max(loo), 4),
        "gap_in_minus_loo": round(cos_in - cos_loo, 4),
    }


def build_pseudo_visual(text_all, text_base, vis_base, base_idx, novel_idx, c_all,
                        ridge=1e3, center_pseudo=False, W=None):
    """生成全部类的视觉原型：base 用真实 support 原型，novel 用桥接伪原型。

    center_pseudo=True 时，映射后减去 base 伪原型的均值方向再归一化。
    动机：CLIP 文本落入 modality gap 的窄锥，仿射映射会把 novel 一起压到
    同一方向（实测 novel 伪原型两两余弦≈1.0，零区分度）。减去全局均值方向
    是 modality-gap 文献的常见去偏处理，可恢复类间区分度。

    返回 (vis_proto_all [C_all,D] 归一化, W 增广映射)。
    """
    d = text_all.shape[1]
    if W is None:
        W = _fit_ridge(text_base, vis_base, ridge)
    vis_proto = torch.zeros(c_all, d, device=text_all.device, dtype=text_all.dtype)
    vis_proto[base_idx] = vis_base
    if novel_idx:
        pseudo = _apply(W, text_all[novel_idx])        # [C_novel, D]
        if center_pseudo:
            base_pseudo = _apply(W, text_base)
            pseudo = pseudo - base_pseudo.mean(0, keepdim=True)
        vis_proto[novel_idx] = F.normalize(pseudo, dim=-1)
    return vis_proto, W


def pseudo_diversity(text_base, vis_base, text_novel, ridge=1e3, center_pseudo=False):
    """诊断：novel 伪原型的类间区分度（off-diagonal 余弦，越低越好）。
    用留一类模拟未见过类：留出的 base 类当"伪 novel"，测映射后区分度。
    """
    c = text_base.shape[0]
    if c < 2:
        return {}
    # 用全部 base 拟合映射，novel 文本走映射
    W = _fit_ridge(text_base, vis_base, ridge)
    if text_novel is not None and text_novel.shape[0] >= 2:
        p = _apply(W, text_novel)
        if center_pseudo:
            p = p - _apply(W, text_base).mean(0, keepdim=True)
        p = F.normalize(p, dim=-1)
        cos = (p @ p.T)
        n = p.shape[0]
        off = cos[~torch.eye(n, dtype=torch.bool, device=p.device)].mean().item()
        return {"novel_pseudo_offdiag": round(off, 4), "center": center_pseudo}
    return {}
