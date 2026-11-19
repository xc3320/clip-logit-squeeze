#!/usr/bin/env python3
"""环境检查：报告本机配置与依赖状态，不安装任何东西、不生成虚假结果。

用法: python scripts/check_env.py
"""
import sys
import platform


def main():
    print("=" * 60)
    print("RAA 实验环境检查")
    print("=" * 60)
    print(f"Python      : {platform.python_version()}")

    # 轻量依赖
    for mod in ["numpy", "pandas"]:
        try:
            m = __import__(mod)
            print(f"{mod:<11}: OK  {getattr(m, '__version__', '')}")
        except Exception:
            print(f"{mod:<11}: MISSING")

    # 训练依赖
    torch_ok = False
    try:
        import torch
        torch_ok = True
        print(f"torch       : OK  {torch.__version__}")
        print(f"CUDA build  : {torch.version.cuda}")
        print(f"cuda.is_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU         : {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"VRAM        : {total:.0f} MB")
            print(f"12GB limit  : {'OK' if total <= 12500 else 'OVER'}")
    except Exception as e:
        print(f"torch       : MISSING ({type(e).__name__})")

    print("-" * 60)
    if not torch_ok:
        print("训练脚本当前不可运行。安装命令（Python 3.13 需 torch>=2.6）：")
        print("  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124")
        print("  pip install open_clip_torch tqdm pyyaml scipy")
        print("驱动已实测 572.83，支持 cu124。")
        print("安装后重新运行本脚本确认 cuda.is_available=True。")
        return 1
    print("环境就绪。可以运行 build_split.py 与 train.py。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
