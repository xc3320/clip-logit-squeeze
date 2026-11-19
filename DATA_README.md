# 数据目录说明

本工程**只使用真实公开数据，不内置、不生成任何合成样本**。此前目标检测工程用于代码自检的合成标注已全部删除，任何结论都不得基于合成数据。

## 当前数据状态（2026-11-19 实测）

| 数据集 | 类别 | 图片 | 状态 | 真实来源 |
|---|---:|---:|---|---|
| DTD | 47 | 5640 | 已就位 | Oxford VGG `https://www.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz` |
| OxfordPets | 37 | 7390 | 已就位 | Oxford VGG images.tar.gz + annotations.tar.gz |
| Flowers102 | 102 | 8189 | 已就位 | Oxford VGG `.../flowers/102/102flowers.tgz` + `imagelabels.mat`；类名表 cat_to_name.json（102 类学名，来源 av192/Flower-Classifier 公开仓库，已核验键 1..102 完整） |
| FGVCAircraft | 100 | 10000 | 已就位 | Oxford VGG `.../fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz`（2.75GB，字节校验一致）+ 独立标注包 annotations.tar.gz（375KB）；类名用官方厂商+型号（如 Boeing_737-300）|
| CIFAR100 | 100 | 60000 | 已就位 | HF 镜像 `uoft-cs/cifar100` parquet（train 118518617B + test 23772751B，字节校验一致）；类名从仓库 README fine_label 映射提取（官方 100 类）。Toronto 官方源限速 18KB/s 弃用 |

Flowers102 官方数据只有 1..102 类编号，CLIP 手工模板需要文本类名，故使用公开 cat_to_name.json 学名表（如 "pink primrose"/"blackberry lily"）。解析脚本 `scripts/prepare_flowers102.py`，缺失学名表时明确退回占位类名并在输出中标注，绝不臆造。

FGVCAircraft 类名由官方 `images_manufacturer_*.txt`（厂商）+ `images_variant_*.txt`（型号）逐图对齐组成（如 "Boeing 737-300"→`Boeing_737-300`），避免裸型号 "737-300" 对 CLIP 语义过弱；解析脚本 `scripts/prepare_fgvcaircraft.py`。

CIFAR100 **不在标准 base-to-novel 11 数据集基准名单内**，仅作第 5 个扩充验证集，用于检验 novel 挤压现象是否跨更多数据分布成立；报告须如实标注其非基准身份。解析脚本 `scripts/prepare_cifar100.py`（HF parquet→PNG）。因每类 600 张测试集过大，评估用 `--max-test-per-class 100` 按类均衡采样（固定 seed、只采真实样本、采样数写入结果文件），非全量测试集，报告须标注。

尚未就位：Caltech101（官方 vision.caltech.edu 旧链 404、HF 镜像 404）、EuroSAT（HF 镜像/Zenodo 404）、StanfordCars（官方 404）。已用 FGVCAircraft + CIFAR100 补位到 5 数据集。

## 放置格式

```
data/
  <dataset>/
    annotations.csv       # 列: image_path, class_name
    images/...             # 图片文件，image_path 指向这里
```

`annotations.csv` 每行一张图，两列：
- `image_path`：图片路径（绝对或相对工程根）
- `class_name`：类别名（下划线或空格均可）

## 流程

1. 放好数据后运行 `python scripts/build_split.py --check` 确认齐备。
2. `python scripts/build_split.py --dataset dtd --shot 16 --seeds 0 1 2 3 4` 生成划分。
3. 装好 torch 后按 docs/实验设计 中的命令训练与评估。

不代为下载需要条款同意的数据；不为跑通流程伪造样本。
