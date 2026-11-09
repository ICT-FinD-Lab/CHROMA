# CHROMA: Coloring-based Graph Reconstruction Defense with Organized Merging of Perturbation

[English](../README.md)

## 方法简介

本仓库实现了 GRASP 及其扩展方法 CHROMA。两种方法研究的是同一类攻击，即攻击者根据节点嵌入的相似度推断哪些节点之间有边。

GRASP 通过伯努利分配结合独立高斯噪声与相同高斯噪声，干扰节点相似度的相对排序。
CHROMA 进一步用图结构决定哪些节点共享扰动。它先对图着色，使同色节点之间没有边，再拆分过大的颜色组。随后逐步合并这些组，同时限制组的大小和落入同一组的边。CHROMA 根据每一步中相邻节点对和非相邻节点对各自的同组比例选出最终划分。同组节点使用同一个高斯噪声向量，不同组独立采样。

[分组代码](../coloring/)生成节点划分，[模型代码](../model.py)将扰动加到节点嵌入上。

原 GRASP：[论文](https://doi.org/10.1145/3711896.3736992) ·
[代码](https://github.com/ZhiyuZone/GRASP)

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| [`main.py`](../main.py)、[`train.py`](../train.py) | 实验入口与训练 |
| [`coloring/`](../coloring/)、[`model.py`](../model.py) | 构建 CHROMA 分组并对节点嵌入加噪 |
| [`configs/`](../configs/)、[`assets/`](../assets/) | 实验配置、预计算分组和数据划分 |
| [`scripts/`](../scripts/)、[`tests/`](../tests/) | 实验命令、输入校验和测试 |
| [`results/`](../results/)、[`summarize.py`](../summarize.py) | 逐种子参考结果与汇总工具 |

各模块及运行产物见[代码说明](code.md)。

## 复现实验

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

### 快速检查

校验随库文件，并用 Cora 在 CPU 上训练和评估两轮：

```bash
python scripts/verify_data.py
python main.py --dataset cora --method CHROMA --device cpu --smoke
```

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

## 引用

原 GRASP 论文的 BibTeX 见[英文说明](../README.md#citation)。原代码和第三方来源见 [NOTICE.md](NOTICE.md)。
