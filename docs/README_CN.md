# CHROMA: Coloring-based Graph Reconstruction Defense with Organized Merging of Perturbation

[English](../README.md)

## 方法简介

本仓库实现了 GRASP 及其扩展方法 CHROMA。两种方法研究的是同一类攻击，即攻击者根据节点嵌入的相似度推断哪些节点之间有边。

GRASP 通过伯努利分配结合独立高斯噪声与相同高斯噪声，干扰节点相似度的相对排序。
CHROMA 进一步用图结构决定哪些节点共享扰动。它先对图着色，使同色节点之间没有边，再拆分过大的颜色组。随后逐步合并这些组，同时限制组的大小和落入同一组的边。CHROMA 根据每一步中相邻节点对和非相邻节点对各自的同组比例选出最终划分。同组节点使用同一个高斯噪声向量，不同组独立采样。

[分组代码](../coloring/)生成节点划分，[模型代码](../model.py)将扰动加到节点嵌入上。

原 GRASP：[论文](https://doi.org/10.1145/3711896.3736992) ·
[代码](https://github.com/ZhiyuZone/GRASP)

## 环境安装

使用 Python 3.11、PyTorch 2.5.1 和 PyTorch Geometric 2.6.1。
依赖见 [requirements.txt](../requirements.txt)。以下命令适用于 CUDA 12.4：

```bash
git clone https://github.com/ICT-FinD-Lab/CHROMA.git
cd CHROMA
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

Windows 下将环境激活命令替换为 `.\.venv\Scripts\Activate.ps1`。
使用其他 CUDA 版本时，选择相应的 PyTorch 和 PyG 安装包。

## 运行方法

以下命令均在仓库根目录中执行。模型和训练参数按数据集从
[configs/main_table.json](../configs/main_table.json) 中读取。

可以先用随库提供的 Cora 分组在 CPU 上跑两轮，检查环境能否正常训练和评估。

```bash
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

### 实验设置与结果

主实验使用种子 0–4，噪声标准差为 0.5、1.0、1.5、2.0。
无噪声基线使用 `--method CLEAN --sigma 0`。
详细设置见[实验说明](experiments.md)。

运行 `python scripts/make_main_grid.py --output main_grid.csv` 可列出整张主表的实验命令。
每次训练的日志、模型和指标都保存在 `runs/`，用下面的命令汇总已完成的运行。

```bash
python summarize.py --input runs --output results/my_summary.csv
```

[参考结果](../results/README.md)保留了原始实验标签 `TGRASP` 和 `TGRASP-default`，汇总时分别显示为 `CHROMA` 和 `CHROMA-default`。
各模块的实现位置见[代码说明](code.md)。
原代码署名见 [NOTICE.md](NOTICE.md)。
