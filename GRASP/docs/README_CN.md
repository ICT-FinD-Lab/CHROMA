# 基于结构化扰动的图重构防御

[English](../../README.md)

## 方法简介

本仓库提供 GRASP 及其拓扑扩展方法 TGRASP 的实现，用于防御基于相似度的图重构攻击。

GRASP 通过伯努利分配结合独立高斯噪声与相同高斯噪声，干扰节点相似度的相对排序。
TGRASP 进一步利用图结构决定哪些节点使用相同扰动：先通过独立集着色和约束合并构建节点分组，再在组内加入共享的高斯扰动。

[分组代码](../coloring/)生成节点划分，[模型代码](../model.py)将扰动加到节点嵌入上。

原 GRASP：[论文](https://doi.org/10.1145/3711896.3736992) ·
[代码](https://github.com/ZhiyuZone/GRASP)

## 环境安装

使用 Python 3.11、PyTorch 2.5.1 和 PyTorch Geometric 2.6.1。
依赖见 [requirements.txt](../requirements.txt)。以下命令适用于 CUDA 12.4：

```bash
git clone https://github.com/ICT-FinD-Lab/GRASP-TGRASP.git
cd GRASP-TGRASP/GRASP
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

Windows 下将环境激活命令替换为 `.\.venv\Scripts\Activate.ps1`。
使用其他 CUDA 版本时，选择相应的 PyTorch 和 PyG 安装包。

## 运行方法

以下命令均在 `GRASP/` 目录中执行。模型和训练参数按数据集从
[configs/main_table.json](../configs/main_table.json) 中读取。

### Cora

```bash
python main.py --dataset cora --backbone gcn --method USP --sigma 1.0 --seed 0 --device 0
python main.py --dataset cora --backbone gcn --method GRASP --sigma 1.0 --seed 0 --device 0
python main.py --dataset cora --backbone gcn --method TGRASP --sigma 1.0 --seed 0 --device 0
```

同样的命令支持 `citeseer`、`pubmed`、`amazon-computer`、`amazon-photo`、`coauthor-cs`、
`coauthor-physics` 和 `wikics`。使用 GAT 时，改为 `--backbone gat`。
数据默认下载到 `data/`，已有数据可通过 `--data-dir PATH` 指定。

### ogbn-arxiv

先构建 TGRASP 分组，再运行训练：

```bash
python prepare_groups.py --dataset ogbn-arxiv
python main.py --dataset ogbn-arxiv --backbone gcn --method GRASP --sigma 1.0 --seed 0 --device 0
python main.py --dataset ogbn-arxiv --backbone gcn --method TGRASP --sigma 1.0 --seed 0 --device 0
```

### 实验设置与结果

主实验使用种子 0–4，噪声标准差为 0.5、1.0、1.5、2.0。
无噪声基线使用 `--method CLEAN --sigma 0`。
使用 `--device cpu` 可在处理器上运行；加上 `--smoke` 可进行两轮短程检查。
详细设置见 [GRASP/docs/experiments.md](../docs/experiments.md)。

运行 `python scripts/make_main_grid.py --output main_grid.csv` 可列出整张主表的实验命令。
每次训练的日志、模型和指标都保存在 `runs/`，用下面的命令汇总已完成的运行。

```bash
python summarize.py --input runs --output results/my_summary.csv
```

[参考结果](../results/README.md)和[代码说明](../docs/code.md)位于 `GRASP/` 中。
原代码署名见 [NOTICE.md](NOTICE.md)。
