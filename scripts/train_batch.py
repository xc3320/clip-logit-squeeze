#!/usr/bin/env python3
"""批量训练：一次加载 CLIP 模型，顺序训练多个方法 × 多个种子。

与逐个调用 train.py 数值口径完全一致（都调用 train.run_one），但省去每次
重复加载权重的开销。结果文件名由 --tag 隔离，不覆盖旧实验。

用法:
  python scripts/train_batch.py --dataset dtd --seeds 0 1 2 3 4 --tag seedfix
"""
import argparse
import itertools
import os
import sys

try:
    import torch
    import open_clip
except ImportError as e:
    print(f"[停止] 缺少 torch/open_clip: {e}")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from train import ALL_METHODS, load_cfg, run_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "raa.yaml"))
    ap.add_argument("--dataset", default="dtd")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--tag", default="seedfix")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    cfg = load_cfg(args.config)
    if not torch.cuda.is_available():
        print("[停止] CUDA 不可用。")
        sys.exit(1)
    device = "cuda"
    dtype = torch.float32 if cfg.get("fp16") is False else torch.float16
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["backbone"], pretrained=cfg["pretrained"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[批量训练] dataset={args.dataset} methods={args.methods} "
          f"seeds={args.seeds} tag={args.tag}", flush=True)

    todo = [m for m in args.methods if m != "clip0"]
    for seed, method in itertools.product(args.seeds, todo):
        run_one(method, args.dataset, seed, cfg, model, preprocess, device, dtype,
                tag=args.tag)
    print(f"[批量训练完成] 共 {len(todo) * len(args.seeds)} 次训练", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
