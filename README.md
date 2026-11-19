# Suppressing Novel-Class Misclassification in CLIP Few-Shot Adaptation

Official code for the paper:

> **Suppressing Novel-Class Misclassification in CLIP Few-Shot Adaptation: Scale-Calibrated Gating and Decoupled Inference**

This repository implements **Scale-Calibrated Gating (SCG)** and **Decoupled Inference**, a training-free method that fixes the *novel logit squeeze* — a failure mode in which dual-stream few-shot adaptation of CLIP misclassifies novel-class test images as base classes in bulk.

## Overview

Adapting CLIP to a base-to-novel few-shot task by fusing text and visual logits raises base accuracy, but systematically pushes novel-class samples toward base classes. We quantify this effect (the *squeeze rate*: 7.8%–81.4% across five datasets), show it predicts the novel accuracy drop (r = 0.952, ViT-B/16; r = 0.864, ResNet-50), and remove it by:

1. **Scale-calibrated gating (SCG)** — per-sample standardization of both branches so fusion reflects reliability, not raw logit magnitude.
2. **Decoupled inference** — base samples use the calibrated fused logits, novel samples use the frozen text logits, restoring novel accuracy to the zero-shot baseline *exactly, seed by seed*.

The pipeline is training-free (only a linear probe on frozen CLIP features is fitted).

## Requirements

- Python 3.9+
- PyTorch ≥ 1.13
- See `requirements.txt` for the full list.

```bash
pip install -r requirements.txt
```

## Data preparation

Datasets are downloaded and split with the scripts in `scripts/`. We use five datasets: **DTD**, **OxfordPets**, **Flowers102**, **FGVC-Aircraft**, and **CIFAR100** (the first four are standard base-to-novel benchmarks; CIFAR100 is an additional validation set).

Each `prepare_<dataset>.py` script downloads the public data, extracts real images, and writes `annotations.csv` (columns `image_path, class_name`). No synthetic samples are used.

## Reproducing the experiments

### 1. Train the visual probe and run SCG

```bash
python scripts/run_scg.py --dataset dtd --seeds 0 1 2 3 4
python scripts/run_scg.py --dataset oxfordpets --seeds 0 1 2 3 4
python scripts/run_scg.py --dataset flowers102 --seeds 0 1 2 3 4
python scripts/run_scg.py --dataset fgvcaircraft --seeds 0 1 2 3 4
python scripts/run_scg.py --dataset cifar100 --seeds 0 1 2 3 4
```

### 2. Run decoupled inference

```bash
python scripts/run_decouple.py --dataset dtd --seeds 0 1 2 3 4
# ... same for the other four datasets
```

### 3. Measure the scale direction (per-backbone asymmetry)

```bash
python scripts/scale_direction.py --config configs/rn50.yaml   # ResNet-50
python scripts/scale_direction.py --config configs/raa.yaml    # ViT-B/16
```

### 4. Analyze results and regenerate figures

```bash
python scripts/analyze_decouple.py
python scripts/analyze_mechanism.py
python scripts/make_prl_figures.py
```

## Backbones

- **ViT-B/16** (`ViT-B/16`, OpenAI weights) — main results.
- **ResNet-50** (`RN50`, OpenAI weights) — cross-backbone control (`configs/rn50.yaml`).

The scale asymmetry direction depends on the visual branch construction (linear probe vs. unbounded residual), not solely on the backbone; the magnitude varies (ViT ≈ 11.7×, RN50 ≈ 8.0×). See Section 3.2 of the paper.

## Results

Per-seed JSON results live under `results/<dataset>/`. The key reported numbers are:

| Dataset | Squeeze rate (ViT) | Squeeze rate (RN50) |
|---|---:|---:|
| DTD | 33.5% | — |
| OxfordPets | 12.2% | — |
| Flowers102 | 35.5% | — |
| FGVC-Aircraft | 85.2% | — |
| CIFAR100 | 28.7% | — |

Full per-seed tables are in the paper (Tables 1–3).

## Citation

If you use this code, please cite:

```bibtex
@article{xu2026squeeze,
  title={Suppressing Novel-Class Misclassification in CLIP Few-Shot Adaptation: Scale-Calibrated Gating and Decoupled Inference},
  author={Xu, Che and Zeng, Haiping and Ma, Qiuyu and Cao, Xinliang},
  journal={Pattern Recognition Letters},
  year={2026}
}
```

## License

This project is released for research purposes. Contact the corresponding author for questions.
