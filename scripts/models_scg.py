#!/usr/bin/env python3
"""SCG (Scale-Calibrated Gating) 模型组件。

目标：修复 base-to-novel 下"无界视觉 logits 挤掉有界文本 logits"的门控塌缩。
这是跟进 CIDA (IEEE CCSB 2026, DOI 10.1109/CCSB71176.2026.11716458) 自述的
开放问题 "branch calibration is the next step"，不是声称首次提出双流门控。

本文件只实现可复用的组件；是否真有提升由 run_scg.py 在真实数据上检验。
任何中间量都如实记录，不伪造。
"""
import torch
import torch.nn.functional as F


def encode_text_all(model, tokenizer, classnames, device):
    """用冻结 CLIP 手工模板编码所有类的文本特征 W_text [C_all, D]（归一化）。"""
    texts = tokenizer([f"a photo of a {c.replace('_', ' ')}." for c in classnames]).to(device)
    with torch.no_grad():
        w = model.encode_text(texts)
    return F.normalize(w.float(), dim=-1)


def train_visual_probe(feats, labels, c_base, epochs=300, lr=0.05, wd=1e-4):
    """在 base_train 冻结特征上训练一个线性视觉探针 W [C_base, D]。

    线性 logits（无 softmax 上界）= 无界视觉残差，对应 CIDA 所说的
    "unbounded visual residual"。只学 base 类；novel 类在融合时补 0，
    因此 novel 只能靠文本分支——这是门控塌缩会掐死 novel 的机制根源。
    """
    d = feats.shape[1]
    w = torch.zeros(c_base, d, device=feats.device, dtype=feats.dtype, requires_grad=True)
    opt = torch.optim.SGD([w], lr=lr, momentum=0.9, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(feats @ w.T, labels)
        loss.backward()
        opt.step()
    return w.detach()


def visual_logits_all(feats, w, base_idx_in_all, c_all):
    """视觉分支 logits [B, C_all]：base 类填线性探针输出，novel 类填 0。"""
    out = torch.zeros(feats.shape[0], c_all, device=feats.device, dtype=feats.dtype)
    out[:, base_idx_in_all] = feats @ w.T
    return out


def text_logits(feats, w_text, logit_scale):
    """文本分支 logits [B, C_all] = scale * cos(f, W_text)。cos 有界 → logits 有界。"""
    return (feats @ w_text.T) * float(logit_scale)


def zscore_per_sample(logits, eps=1e-6):
    """尺度校准核心：逐样本 z-score，把每条分支 logits 归一到均值0/标准差1。

    减均值不改变单分支 argmax；除标准差把两分支拉到可比尺度，
    使门控权重真正反映"信谁"而非"谁的数大"。
    """
    m = logits.mean(dim=1, keepdim=True)
    s = logits.std(dim=1, keepdim=True) + eps
    return (logits - m) / s


def estimate_reliability(feats, w_text, logit_scale, labels, base_idx_in_all, c_base):
    """逐类文本零样本可靠性 rel_i ∈ [0,1]：base_train 上文本分支正确率。

    可靠性高 → 文本分支已够用 → 门控偏文本；
    可靠性低 → 文本分支不可信 → 门控偏视觉探针。
    """
    logits = text_logits(feats, w_text, logit_scale)[:, base_idx_in_all]
    pred = logits.argmax(dim=-1)
    rel = torch.zeros(c_base, device=feats.device)
    for i in range(c_base):
        mask = labels == i
        rel[i] = (pred[mask] == i).float().mean() if mask.any() else 0.5
    return rel


def build_gate(rel, gate_k=4.0, gate_floor=0.2):
    """由可靠性映射文本门控权重 g_i = floor + (1-floor)*sigmoid(k*(rel-0.5))。

    保底 floor 防止低可靠类完全放弃文本（文本仍覆盖语义）。
    返回 base 类的 g [C_base]。
    """
    import torch
    return gate_floor + (1.0 - gate_floor) * torch.sigmoid(gate_k * (rel - 0.5))


def fuse(text_l, vis_l, g_base, base_idx_in_all, c_all, novel_g=1.0):
    """门控融合。novel 类无视觉证据 → 门控强制 g=1（纯文本）。

    text_l, vis_l: [B, C_all]
    g_base: [C_base] base 类文本权重
    """
    g = torch.full((c_all,), novel_g, device=text_l.device, dtype=text_l.dtype)
    g[base_idx_in_all] = g_base.to(dtype=text_l.dtype)
    return g * text_l + (1.0 - g) * vis_l


def fuse_calibrated(text_l, vis_l, g_base, base_idx_in_all, c_all, novel_g=1.0):
    """SCG：先逐样本 z-score 校准两分支，再门控融合。"""
    return fuse(zscore_per_sample(text_l), zscore_per_sample(vis_l),
                g_base, base_idx_in_all, c_all, novel_g)
