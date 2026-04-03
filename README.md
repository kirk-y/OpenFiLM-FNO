# HFSmodel (OpenHFSmodelFNO)

[中文版本](README_cn.md)

This repository provides a training and evaluation framework for a surrogate model based on the Fourier Neural Operator (FNO). It is designed to efficiently train and test predictions of spatiotemporal evolution or physical fields.

> **Note:** The training and testing data required for this project (the contents of the `data/` directory and specific data structure definitions) are temporarily **closed-source**. If you wish to run the model on your own dataset, please prepare your own input data. Specific instructions on the required data formats will be provided in a future update.

## Directory Structure
```text
.
├── common/              # Common utilities: loss functions (loss.py), visualization (visualization.py), etc.
├── data/                # Data processing modules (data itself is not open-sourced)
├── experiments/         # Directory for experimental outputs (model weights, logs, charts, etc.)
├── models/              # Model definitions (FNO based on fno.py)
├── training/            # Training pipeline logic (trainer.py)
├── config.yml           # Configuration file for training
├── main.py              # Main entry point for training
├── test_config.yml      # Configuration file for testing & evaluation
├── test_eval.py         # Testing script typically for GPU execution (Test Set Version 1)
└── test_eval_cpu.py     # Testing script explicitly for CPU execution (Test Set Version 2)
```

## Dependencies
It is recommended to use Python 3.8+ with any modern, mainstream version of PyTorch. You can install the basic dependencies using the following commands:
```bash
pip install torch torchvision torchaudio
pip install numpy pyyaml matplotlib h5py
# You might also need other standard scientific or progress tracking libraries depending on your env:
# pip install tqdm scipy
```

## Quick Start

### 1. Training
The sole entry point for model training is `main.py`. All hyperparameters (e.g., network structure, learning rate, batch size) are specified in `config.yml`.
Once `config.yml` is configured, execute the following command in the terminal to start training:
```bash
python -m main
```
*(Alternatively, you can run `python main.py`)*

During training, weights, logs, and visualization charts will be automatically saved under the `experiments/` or associated output directories.

### 2. Testing and Evaluation
This project provides two versions of testing scripts catering to different hardware availability and test scenarios:
- **`test_eval.py`**: The primary testing evaluation script (typically executed on GPUs).
- **`test_eval_cpu.py`**: The CPU-only evaluation script, useful for inference on non-GPU nodes or validating specific test flows.

All testing parameters are governed by `test_config.yml`. To run the tests, execute:
```bash
python test_eval.py
# OR
python test_eval_cpu.py
```

## Evaluation Metrics
- **MSE / MAE / Increment MSE**: Fundamental metrics measuring absolute errors and incremental prediction proficiency.
- **Relative L2 Error**: Calculated per sample as $\lVert \hat{y}-y \rVert_2 / (\lVert y \rVert_2 + 10^{-8})$. During training, it records to `metrics/relative_l2_curve.png`. Testing phases generate `relative_l2_error_curve.png` and output the sample-wise JSON.
- **Normalized RMSE (NRMSE)**: Normalized via $\mathrm{RMSE} / (\max(y)-\min(y)+10^{-8})$, it also provides training/validation curves and sample-wise testing charts.

Historical data of all metrics will be written to `metrics/history.json`. The testing pipeline additionally outputs `*_per_sample.json` files for downstream analysis and customized visualizations.

## Custom Test Set Loading
You can specify a custom pickle file path in `testing.test_data_path` within the config (use an absolute path or relative to `config.yml`). The testing phase will prioritize loading this file as the test set rather than relying on the default partitioned dataset. This is highly useful for reproducing benchmarks on specific sample sets or reusing existing splits.

## Error-Driven Testing Visualization
The testing pipeline now sorts validation runs by sample-wise MSE, plotting prediction/increment charts in the `figures/` directory exclusively for the top 30 samples with the highest errors. This ensures rapid identification of the worst-performing cases. The complete per-sample metrics are still preserved in `*_per_sample.json` for individualized analysis.

