#!/usr/bin/env python3
"""RAA 模型：冻结 CLIP + 可学习 prompt（CoOp 式）+ 逐类可靠性自适应锚定损失。

依赖 torch 与 open_clip_torch；缺失时明确停止，不生成虚假结果。

RAA 核心（相对 KgCoOp 的改动）：
  KgCoOp:  L = L_ce + λ * mean_i || W_coop,i - W_clip,i ||^2      (标量 λ, 一刀切)
  RAA:     L = L_ce + sum_i λ_i * || W_coop,i - W_clip,i ||^2     (逐类 λ_i)
  其中 λ_i = center + spread * (± (rel_i - mean_rel))
       rel_i 是第 i 类的零样本可靠性，从 support 集前向估计（无需训练）。
       正向：可靠类强锚定；reverse 变体：不可靠类强锚定。
       新增 margin 模式用连续竞争边距替代粗粒度 16-shot 正确率。

注意：本文件是候选算法骨架，新颖性需系统查新确认（见 docs/实验设计_RAA...md 第 7 节）。
"""
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    print("[停止] 缺少 torch，无法加载模型。请先按 docs/环境搭建.md 安装。")
    print(f"       原始错误: {e}")
    sys.exit(1)


def encode_text_features(clip_model, tokens, ctx=None, n_ctx=0):
    """用 open_clip 官方 _encode_text 的完整路径编码文本，返回归一化特征 [C, D]。

    关键：必须包含 token_embedding + positional_embedding + attn_mask + ln_final
    + argmax(EOT) pool + text_projection。此前手写版本漏了 positional_embedding
    与 attn_mask，导致文本特征错误（与官方余弦仅 0.44），已修复。

    ctx: 可选 [n_ctx, D] 可学习上下文，注入到 SOT 之后的位置（CoOp 式）。
    """
    dt = clip_model.transformer.get_cast_dtype()
    x = clip_model.token_embedding(tokens).to(dt)          # [C, L, D]
    if ctx is not None:
        x = x.clone()
        x[:, 1:1 + n_ctx, :] = ctx.to(dt)                  # 注入可学习上下文
    x = x + clip_model.positional_embedding.to(dt)         # 位置编码（不可漏）
    x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)  # 因果掩码（不可漏）
    x = clip_model.ln_final(x)
    pooled = x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)]
    pooled = pooled @ clip_model.text_projection
    return F.normalize(pooled, dim=-1)


class CoOpPrompt(nn.Module):
    """CoOp 式可学习上下文：[V]_1 ... [V]_M [class]。"""

    def __init__(self, clip_model, classnames, n_ctx=16, ctx_init=None):
        super().__init__()
        self.classnames = list(classnames)
        self.n_ctx = n_ctx
        embed_dim = clip_model.text_projection.shape[1]
        ctx = torch.empty(n_ctx, embed_dim)
        nn.init.normal_(ctx, std=0.02)
        self.ctx = nn.Parameter(ctx)
        # 用普通属性保存冻结 CLIP 组件引用；不注册为子模块，避免 .to(dtype) 重复转换。
        # 冻结由 train.py 中 model.requires_grad_(False) 保证。
        object.__setattr__(self, "_clip_model", clip_model)
        object.__setattr__(self, "token_embedding", clip_model.token_embedding)
        object.__setattr__(self, "transformer", clip_model.transformer)
        object.__setattr__(self, "ln_final", clip_model.ln_final)
        object.__setattr__(self, "text_projection", clip_model.text_projection)
        # open_clip 的 CLIP 对象无 .dtype 属性，从参数推断
        object.__setattr__(self, "dtype", next(clip_model.parameters()).dtype)

    def forward_text_features(self):
        """返回归一化文本特征 W_coop [C, D]（走官方编码路径，含位置编码+attn_mask）。

        模板: SOT + [V]_1..[V]_M + 类名 token + EOT，前 M 个上下文位置换成可学习向量。
        依赖全局 tokenize(fn) 由训练脚本注入（open_clip.tokenize）。
        """
        prompts = [" ".join(["X"] * self.n_ctx) + " " + c.replace("_", " ") + "."
                   for c in self.classnames]
        toks = tokenize(prompts)                       # [C, L] long
        return encode_text_features(self._clip_model, toks, ctx=self.ctx, n_ctx=self.n_ctx)


# tokenize 由训练脚本注入（避免此处强依赖 simple_tokenizer 的实现细节）
def set_tokenizer(fn):
    global tokenize
    tokenize = fn


def clip_anchor_features(clip_model, classnames, dtype, device):
    """计算手工模板的 CLIP 文本锚点 W_clip [C, D]（冻结，无梯度，走官方编码路径）。"""
    with torch.no_grad():
        texts = tokenize([f"a photo of a {c.replace('_', ' ')}." for c in classnames]).to(device)
        return encode_text_features(clip_model, texts)


def estimate_reliability(image_features, text_anchor, labels, center, spread,
                         mode="acc", reverse=False, margin_scale=10.0):
    """估计逐类零样本可靠性 rel_i ∈ [0,1]，映射到锚定权重 λ_i。

    image_features: [N, D] 归一化；text_anchor: [C, D] 归一化；labels: [N]

    mode:
      'acc'    : support 样本被零样本 CLIP 正确分类的比例（粗粒度，原始 RAA）
      'conf'   : 与类锚点的平均余弦相似度（DTD 上过于集中，仅作消融）
      'margin' : sigmoid(margin_scale * (真类余弦 − 最强竞争类余弦)) 的均值，
                 连续信号，避免 16-shot 下 acc 只有 0/1/16 粒度过粗的问题

    reverse=False（原始 RAA）：可靠类锚定更强，λ_i = center + spread*(rel_i − mean_rel)
    reverse=True（反向变体）  ：不可靠类锚定更强，λ_i = center + spread*(mean_rel − rel_i)

    两种方向都保证 mean(λ_i) = center，与固定 λ=center 的 KgCoOp 平均锚定强度一致。
    """
    sim = image_features @ text_anchor.T            # [N, C]
    pred = sim.argmax(dim=-1)
    C = text_anchor.shape[0]
    rel = torch.zeros(C, device=image_features.device, dtype=torch.float32)
    for i in range(C):
        mask = (labels == i)
        if mask.sum() > 0:
            if mode == "acc":
                rel[i] = (pred[mask] == i).float().mean()
            elif mode == "conf":
                rel[i] = (sim[mask, i].mean() + 1) / 2
            elif mode == "margin":
                s = sim[mask]                                  # [n_i, C]
                true_sim = s[:, i]                             # 真类余弦
                wrong = s.clone()
                wrong[:, i] = torch.finfo(s.dtype).min         # 屏蔽真类
                best_wrong = wrong.max(dim=-1).values          # 最强竞争类余弦
                rel[i] = torch.sigmoid(margin_scale * (true_sim - best_wrong)).mean()
            else:
                raise ValueError(f"未知可靠性模式: {mode}")
        else:
            rel[i] = rel.mean() if i > 0 else 0.5  # 无样本类用已有均值，不臆造
    mean_rel = rel.mean()
    if reverse:
        lam_i = center + spread * (mean_rel - rel)
    else:
        lam_i = center + spread * (rel - mean_rel)
    return rel, lam_i


def raa_loss(w_coop, w_clip, lam_i):
    """RAA 损失: mean_i λ_i * || w_coop,i - w_clip,i ||^2。

    用 mean 与 KgCoOp 聚合方式一致；当 λ_i 全等于 center 时严格退化为 KgCoOp。
    """
    diff = w_coop - w_clip
    per_class = diff.pow(2).sum(dim=-1)
    return (lam_i * per_class).mean()


def kgcoop_loss(w_coop, w_clip, lam):
    """对照基线 KgCoOp: 标量 λ 的一刀切锚定（mean 聚合）。"""
    return lam * (w_coop - w_clip).pow(2).sum(dim=-1).mean()
