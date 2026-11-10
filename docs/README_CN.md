# CHROMA

*基于图着色与受限合并的图重构防御*

[English](../README.md) · [快速开始](#快速开始) · [实验设置](experiments.md) · [参考结果](../results/README.md)

CHROMA 用图着色和受限合并给节点分组，让同组节点共享高斯噪声，以防御基于节点嵌入相似度的图重构攻击。

想先试跑，可以在 [CPU 上用 Cora 跑两轮](#cpu-试跑)。要比较 CHROMA、GRASP 和独立噪声基线，按[复现实验](#复现实验)里的命令运行。

## 快速开始

### 安装依赖

使用 Python 3.11。克隆后在仓库根目录执行以下命令。

```bash
git clone https://github.com/ICT-FinD-Lab/CHROMA.git
cd CHROMA
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell 下使用 `.\.venv\Scripts\Activate.ps1` 激活环境。

在 CPU 上试运行或跑测试时，安装以下依赖：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=torch-scatter,torch-sparse -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```

在 CUDA 12.4 上跑正式实验时，在新环境中安装对应的依赖：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --only-binary=torch-scatter,torch-sparse -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

### CPU 试跑

先检查仓库提供的数据文件，再用 Cora 在 CPU 上训练和评估两轮。

```bash
python scripts/verify_data.py
python main.py --dataset cora --method CHROMA --device cpu --smoke
```

## 复现实验

### Cora

```bash
python main.py --dataset cora --backbone gcn --method USP --sigma 1.0 --seed 0 --device 0
python main.py --dataset cora --backbone gcn --method GRASP --sigma 1.0 --seed 0 --device 0
python main.py --dataset cora --backbone gcn --method CHROMA --sigma 1.0 --seed 0 --device 0
```

同样的命令支持 `citeseer`、`pubmed`、`amazon-computer`、`amazon-photo`、`coauthor-cs`、
`coauthor-physics` 和 `wikics`。使用 GAT 时，改为 `--backbone gat`。
数据默认下载到 `data/`，已有数据可通过 `--data-dir PATH` 指定。

### ogbn-arxiv

训练前先构建 CHROMA 的节点分组。

```bash
python prepare_groups.py --dataset ogbn-arxiv
python main.py --dataset ogbn-arxiv --backbone gcn --method GRASP --sigma 1.0 --seed 0 --device 0
python main.py --dataset ogbn-arxiv --backbone gcn --method CHROMA --sigma 1.0 --seed 0 --device 0
```

### 主表复现

主实验使用种子 0–4，噪声标准差为 0.5、1.0、1.5、2.0。
无噪声基线使用 `--method CLEAN --sigma 0`。
详细设置见[实验说明](experiments.md)。

下面两条命令分别列出主表的训练命令、从随库的逐种子结果生成汇总表：

```bash
python scripts/make_main_grid.py --output main_grid.csv
python summarize.py --input results/main_table_per_seed.csv --output results/reference_summary.csv
```

每次训练的日志、模型和指标都保存在 `runs/`。汇总自行完成的实验时，使用
`--input runs --output results/my_summary.csv`。主实验和对照实验的记录见[参考结果](../results/README.md)。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| [`main.py`](../main.py)、[`train.py`](../train.py) | 实验入口与训练 |
| [`coloring/`](../coloring/)、[`model.py`](../model.py) | 构建 CHROMA 分组并对节点嵌入加噪 |
| [`configs/`](../configs/)、[`assets/`](../assets/) | 实验配置、预计算分组和数据划分 |
| [`scripts/`](../scripts/)、[`tests/`](../tests/) | 实验命令、输入校验和测试 |
| [`results/`](../results/)、[`summarize.py`](../summarize.py) | 逐种子参考结果与汇总工具 |

各模块及运行产物见[代码说明](code.md)。

## 引用

原 GRASP 论文的 BibTeX 见[英文说明](../README.md#citation)。

## 致谢

CHROMA 基于 [GRASP（Guo 等，KDD 2025）](https://doi.org/10.1145/3711896.3736992)
及其[原始实现](https://github.com/ZhiyuZone/GRASP)。本项目使用
[PyTorch](https://pytorch.org/)、[PyTorch Geometric](https://pyg.org/)、
[OGB](https://ogb.stanford.edu/) 和 [NetworkX](https://networkx.org/)。
训练日志模块改编自 [OGB 仓库](https://github.com/snap-stanford/ogb)。
