# HFSmodel (OpenHFSmodelFNO)

[English Version](README.md)

本仓库提供了一个基于傅里叶神经算子（Fourier Neural Operator, FNO）的代理模型训练与评估框架。该项目旨在高效地训练并评估相关时空演化或物理场预测任务。

> **注意：** 本项目中依赖的训练和测试数据（`data/` 目录的内容及具体数据结构的说明）暂不开源。如果您希望在自己的数据集上运行，请自行准备输入数据。

## 目录结构
```text
.
├── common/              # 公共库：定义损失函数(loss.py)、可视化(visualization.py)等
├── data/                # 数据处理模块 (暂不开源数据)
├── experiments/         # 实验记录与输出目录（模型权重、日志、可视化图表等）
├── models/              # 网络模型定义 (基于 fno.py 的傅里叶神经算子)
├── training/            # 训练逻辑实现 (trainer.py)
├── config.yml           # 训练配置文件
├── main.py              # 模型训练入口脚本
├── test_config.yml      # 测试与评估配置文件
├── test_eval.py         # GPU版本的测试评估脚本（测试集版本 1）
└── test_eval_cpu.py     # CPU版本的测试评估脚本（测试集版本 2）
```

## 环境依赖
推荐使用 Python 3.8+，并安装主流版本的 PyTorch。您可以通过以下命令快速安装基本环境：
```bash
pip install torch torchvision torchaudio
pip install numpy pyyaml matplotlib h5py
# 根据实际需要，您可能还需要额外安装科学计算或进度条相关库：
# pip install tqdm scipy
```

## 快速开始

### 1. 模型训练
项目的唯一训练入口为 `main.py`。所有的超参数（如网络结构、学习率、batch size 等）均通过 `config.yml` 制定。
在配置好 `config.yml` 之后，在终端中执行以下命令开始训练：
```bash
python -m main
```
*(或者执行 `python main.py`)*

训练过程中的权重、日志和可视化图表将会自动存入 `experiments/` 或相关配置的输出目录中。

### 2. 模型测试与评估
本项目针对不同的评测场景和可用硬件，提供了两个版本的测试脚本：
- **`test_eval.py`**：通常基于 GPU 运行的主力测试集评估脚本。
- **`test_eval_cpu.py`**：基于 CPU 运行的评估脚本，主要用于无 GPU 节点的推理，或特定测试流程的验证。

所有的测试参数由 `test_config.yml` 配置。运行时请使用：
```bash
python test_eval.py
# 或
python test_eval_cpu.py
```

## 评估指标说明
- **MSE / MAE / Increment MSE**：衡量绝对误差与增量预测能力的基础指标。
- **Relative L2 Error**：逐样本计算 $\lVert \hat{y}-y \rVert_2 / (\lVert y \rVert_2 + 10^{-8})$，训练阶段记录在 `metrics/relative_l2_curve.png`，测试阶段生成 `relative_l2_error_curve.png` 并输出逐样本 JSON。
- **Normalized RMSE (NRMSE)**：采用 $\mathrm{RMSE} / (\max(y)-\min(y)+10^{-8})$ 进行归一化，同样提供训练/验证曲线与测试逐样本图表。

所有指标的历史数据会写入 `metrics/history.json`，测试流程还会额外输出 `*_per_sample.json` 文件，便于后续分析与可视化。

## 自定义测试集加载
在 `testing.test_data_path` 中填写一个 pickle 文件路径（绝对路径或相对`config.yml`的路径）。测试阶段将优先加载该文件作为测试集合，而不是依赖默认保存/划分的数据集。这对复现特定样本集或复用已有切分很有帮助。

## 误差驱动的测试可视化
测试阶段现在会根据逐样本 MSE 排序，仅为前 30 个误差最大的样本绘制 `figures/` 下的预测/增量图表，便于快速定位表现最差的案例；所有逐样本指标仍会以 `*_per_sample.json` 保存，供定制分析。
