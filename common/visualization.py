import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Sequence, Union

def draw(pred: np.ndarray, target: np.ndarray, save_path: str):
    pass

def draw_all(preds: np.ndarray, targets:  np.ndarray, w_old: np.ndarray, 
             ns: int, fi: np.ndarray, save_path: str, sample_indices: Union[Sequence[int], None] = None):
    # print(f'preds shape: {preds[0].shape}, targets shape: {targets[0].shape}')
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    if sample_indices is None:
        indices = range(len(preds))
    else:
        indices = [int(i) for i in sample_indices if 0 <= i < len(preds)]
        if not indices:
            print('draw_all: 无有效样本索引，跳过绘图。')
            return
    for i in indices:
        plt.figure(figsize=(15, 5))
        truth, pred, w = targets[i][:ns[i]], preds[i][:ns[i]], w_old[i][:ns[i]]
        plt.plot(w, label='Old W', color='green', lw=0.4)
        plt.plot(truth, label='Target', color='orange', lw=0.4)
        plt.plot(pred+w, label='Predicted', color='blue', lw=0.4)
        plt.title(f'Prediction vs Target for Sample {i}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_path, f'sample_{i}.svg'), format='svg')
        plt.close()
    for i in indices:
        plt.figure(figsize=(15, 5))
        truth, pred, w = targets[i][:ns[i]], preds[i][:ns[i]], w_old[i][:ns[i]]
        plt.plot(truth - w, label='increment')
        plt.plot(pred, label='predicted increment', lw=0.5)
        plt.plot((fi[i][:ns[i]]==2)*0.1, label='fi', lw=0.5, alpha=0.5)
        plt.title(f'Increment for Sample {i}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_path, f'sample_{i}_increment.svg'), format='svg')
        plt.close()


def plot_metric_series(metric_values, metric_name: str, save_path: str):
    if not metric_values:
        return
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(12, 4))
    indices = np.arange(1, len(metric_values) + 1)
    plt.plot(indices, metric_values, marker='o', linewidth=1.0, markersize=3)
    plt.xlabel('Sample Index')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} per Sample')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{metric_name}_curve.png'), dpi=300)
    plt.close()