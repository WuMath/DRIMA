# DRIMA

DRIMA 是一个基于单细胞 RNA/ATAC 多组学数据推断动态基因调控网络的研究项目。项目沿伪时间构建时序异质图，并使用代码中名为 **DyGMamba** 的模型学习动态调控关系，最终用于分析 TF–Region、Region–Gene 和 TF–Gene 网络。

## 项目内容

```text
code/
├── benchmark/   # 核心模型、基准数据处理、对比方法与结果评估
├── case1/       # BMMC 分化轨迹：B cell、erythroid、myeloid
├── case2/       # 阿尔茨海默病：CRND8/WT 的 Microglia、Astrocyte
└── case3/       # 不同组织：brain、lung、skin
```

每个案例通常包含：

- `src/`：数据预处理、伪时间推断、图构建和模型训练；
- `bash/`：适用于 Slurm 集群的运行脚本；
- `*.ipynb`：结果分析与绘图。

核心实现位于 `code/benchmark/dygmamba/src/`，其中 `models/` 为 DyGMamba 模型，`pdata/` 和 `self_utils/` 为数据与先验网络处理工具。

## 环境要求

推荐使用 Linux、Python 3.9 和 CUDA GPU。主要依赖包括：

```bash
conda create -n drima python=3.9 -y
conda activate drima

pip install torch mamba-ssm numpy pandas scipy scikit-learn \
  scanpy anndata muon networkx h5py tqdm matplotlib seaborn \
  gtfparse pyranges pybedtools pyfaidx psutil dill

export PYTHONPATH="$PWD/code/benchmark/dygmamba/src:$PYTHONPATH"
```

伪时间推断还需要 R 及 `slingshot`、`SingleCellExperiment`、`Matrix`、`dplyr` 等包。JASPAR motif 扫描需要额外安装 **BEDTools** 和 **MEME Suite（FIMO）**。

## 运行方法

仓库不包含原始数据，且脚本中的 `data_root`、`output_path`、`gtf_file_path` 和 `sys.path.append(...)` 使用了作者环境的绝对路径。运行前请先将所选案例 `src/` 与 `bash/` 中的这些路径改为本机路径。输入目录至少应包含配对的：

```text
rna_origin.h5ad
atac_origin.h5ad
```

以 `case1/myeloid` 为例，依次执行：

```bash
cd code/case1/myeloid/src

python preprocess_data.py
Rscript trajectory_inference.R
python downsample.py
Rscript trajectory_inference.R
python dygmamba_preprocess.py
python main_train_hpc.py
```

如需构建 JASPAR 先验并进行评估，再执行：

```bash
python peak.py
bedtools getfasta -fi /path/to/hg38.fa -bed /path/to/peaks.bed -fo /path/to/peaks.fa
fimo --thresh 1e-4 --oc /path/to/fimo_out /path/to/JASPAR.meme /path/to/peaks.fa
python benchmark_data_read.py
```

在 Slurm 集群上，可修改对应 `bash/*.sh` 中的分区、环境和路径后，通过 `sbatch` 提交：

```bash
sbatch code/case1/myeloid/bash/data_process.sh
sbatch code/case1/myeloid/bash/dyg_bash.sh
```

模型输出、检查点和中间图数据会写入各脚本配置的 `process/` 目录；最终统计与绘图可运行 `code/benchmark/` 及各案例目录中的 Notebook。
