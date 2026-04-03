import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from collections import defaultdict
try:
    import pynvml
except ImportError:
    pynvml = None
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免 GUI 开销
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW
from copy import deepcopy

from data.data_v3 import build_ddm_coef_matrix
from main import prepare_datasets
from models.fno import FluidSolidFNOmodel
from training.trainer import FluidSolidTrainer
from common.loss import loss_fn


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]
        tensor_state = {k: v for k, v in checkpoint_obj.items() if isinstance(v, torch.Tensor)}
        if tensor_state:
            return tensor_state
    raise ValueError('无法从 checkpoint 中解析 state_dict')


def _remap_legacy_neuralop_keys(state_dict: dict) -> dict:
    remapped = {}
    conv_bias = None

    for key, value in state_dict.items():
        new_key = key

        if key.startswith('fno.fno_blocks.convs.weight.') and key.endswith('.tensor'):
            layer_idx = key[len('fno.fno_blocks.convs.weight.'):-len('.tensor')]
            if layer_idx.isdigit():
                new_key = f'fno.fno_blocks.convs.{layer_idx}.weight.tensor'

        elif key == 'fno.fno_blocks.convs.bias' and isinstance(value, torch.Tensor):
            conv_bias = value
            continue

        elif key.startswith('fno.fno_blocks.fno_skips.') and key.endswith('.weight'):
            parts = key.split('.')
            # Legacy key: fno.fno_blocks.fno_skips.<idx>.weight
            # New key:    fno.fno_blocks.fno_skips.<idx>.conv.weight
            if len(parts) == 5 and parts[3].isdigit() and parts[4] == 'weight':
                new_key = f'fno.fno_blocks.fno_skips.{parts[3]}.conv.weight'

        remapped[new_key] = value

    if isinstance(conv_bias, torch.Tensor) and conv_bias.ndim >= 3:
        n_layers = int(conv_bias.shape[0])
        for i in range(n_layers):
            remapped[f'fno.fno_blocks.convs.{i}.bias'] = conv_bias[i]

    return remapped


def _merge_model_config_from_checkpoint(config: dict, checkpoint_path: Path) -> dict:
    merged = deepcopy(config)
    ckpt_cfg_path = checkpoint_path.resolve().parent.parent / 'config.yml'
    if not ckpt_cfg_path.exists():
        print(f"未找到同目录训练配置，跳过结构对齐: {ckpt_cfg_path}")
        return merged

    try:
        with open(ckpt_cfg_path, 'r', encoding='utf-8') as f:
            ckpt_cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
    except Exception as e:
        print(f"读取训练配置失败，使用当前测试配置: {e}")
        return merged

    if isinstance(ckpt_cfg.get('model'), dict):
        merged['model'] = ckpt_cfg['model']
        print(f"已使用训练配置中的 model 参数: {ckpt_cfg_path}")
    else:
        print(f"训练配置中缺少 model 字段，保持当前测试配置: {ckpt_cfg_path}")

    return merged


def _inject_fno_ratio_from_state_dict(config: dict, state_dict: dict) -> None:
    model_cfg = config.setdefault('model', {})
    fno_cfg = model_cfg.setdefault('fno', {})
    hidden_channels = int(fno_cfg.get('hidden_channels', 32))
    if hidden_channels <= 0:
        return

    lifting_w = state_dict.get('fno.lifting.fcs.0.weight')
    if isinstance(lifting_w, torch.Tensor) and lifting_w.ndim >= 1:
        lifting_channels = int(lifting_w.shape[0])
        lifting_ratio = float(lifting_channels) / float(hidden_channels)
        fno_cfg['lifting_channel_ratio'] = lifting_ratio
        print(f"从 checkpoint 推断 lifting_channel_ratio={lifting_ratio:g}")

    projection_w = state_dict.get('fno.projection.fcs.0.weight')
    if isinstance(projection_w, torch.Tensor) and projection_w.ndim >= 1:
        projection_channels = int(projection_w.shape[0])
        projection_ratio = float(projection_channels) / float(hidden_channels)
        fno_cfg['projection_channel_ratio'] = projection_ratio
        print(f"从 checkpoint 推断 projection_channel_ratio={projection_ratio:g}")


def _filter_compatible_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    target_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            skipped.append((key, 'non-Tensor metadata'))
            continue
        if key not in target_state:
            skipped.append((key, 'missing in current model'))
            continue

        target_tensor = target_state[key]
        # neuralop compatibility: legacy checkpoints may store complex spectral
        # weights as real tensors with trailing dimension 2.
        if (
            torch.is_complex(target_tensor)
            and (not torch.is_complex(value))
            and value.ndim == target_tensor.ndim + 1
            and value.shape[-1] == 2
            and tuple(value.shape[:-1]) == tuple(target_tensor.shape)
        ):
            value = torch.view_as_complex(value.contiguous())

        if value.shape != target_state[key].shape:
            skipped.append((key, f'shape mismatch {tuple(value.shape)} vs {tuple(target_state[key].shape)}'))
            continue
        filtered_state[key] = value

    if skipped:
        print('跳过以下不可加载的参数：')
        for name, reason in skipped:
            print(f'  {name}: {reason}')

    return filtered_state

def get_gpu_metrics(device_index=0):
    if pynvml is None:
        return 0.0, 0.0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        power = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
        return float(util.gpu), power / 1000.0
    except Exception:
        return 0.0, 0.0


def safe_nanmean(values):
    if not values:
        return float('nan')
    arr = np.array(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float('nan')
    return float(np.nanmean(arr))


def compute_increment_relative_l2(pred_increment: torch.Tensor, true_increment: torch.Tensor, eps: float = 1e-12) -> float:
    diff = pred_increment - true_increment
    denom = torch.norm(true_increment) + eps
    return float((torch.norm(diff) / denom).item())


# 矩阵不做全局缓存（每样本矩阵独立且只用一次，缓存只会浪费内存）
# 线程安全由 ThreadPoolExecutor 保证，各线程加载各自的局部变量，用完即释放


def compute_equation_residual_vector(sample_dir: str, cb_modified: np.ndarray) -> np.ndarray:
    tag = f"[方程残差][{Path(sample_dir).name if sample_dir else 'None'}]"
    if not sample_dir:
        print(f"{tag} 跳过：sample_dir 为空")
        return np.array([], dtype=np.float64)
    cm_path = Path(sample_dir) / 'coupled_matrix.npy'
    cc_path = Path(sample_dir) / 'coupled_cc.npy'
    if not cm_path.exists():
        print(f"{tag} 跳过：缺少文件 coupled_matrix.npy")
        return np.array([], dtype=np.float64)
    if not cc_path.exists():
        print(f"{tag} 跳过：缺少文件 coupled_cc.npy")
        return np.array([], dtype=np.float64)
    try:
        A = np.load(cm_path)
        b = np.load(cc_path)
    except Exception as e:
        print(f"{tag} 跳过：读取矩阵文件失败 — {e}")
        return np.array([], dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        print(f"{tag} 跳过：coupled_matrix 形状异常 {A.shape}，期望方阵")
        return np.array([], dtype=np.float64)
    if b.shape[0] != A.shape[0]:
        print(f"{tag} 跳过：b 长度 {b.shape[0]} 与矩阵行数 {A.shape[0]} 不匹配")
        return np.array([], dtype=np.float64)
    if cb_modified.shape[0] != A.shape[1]:
        print(f"{tag} 跳过：cb_modified 长度 {cb_modified.shape[0]} 与矩阵列数 {A.shape[1]} 不匹配")
        return np.array([], dtype=np.float64)
    try:
        resid = b - A.dot(cb_modified)
        resid = np.asarray(resid, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(resid)):
            print(f"{tag} 警告：残差向量可能含 NaN/Inf 输入")
        return resid
    except Exception as e:
        print(f"{tag} 跳过：计算残差时异常 — {e}")
        return np.array([], dtype=np.float64)


def compute_equation_residual(sample_dir: str, cb_modified: np.ndarray) -> float:
    resid = compute_equation_residual_vector(sample_dir, cb_modified)
    if resid.size == 0:
        return float('nan')
    val = float(np.linalg.norm(resid))
    if not np.isfinite(val):
        tag = f"[方程残差][{Path(sample_dir).name if sample_dir else 'None'}]"
        print(f"{tag} 警告：残差范数为 {val}，可能含 NaN/Inf 输入")
    return val


def compute_stress_residual_vector(sample_dir: str, w_new: np.ndarray, w_pred: np.ndarray) -> np.ndarray:
    tag = f"[应力残差][{Path(sample_dir).name if sample_dir else 'None'}]"
    if not sample_dir:
        print(f"{tag} 跳过：sample_dir 为空")
        return np.array([], dtype=np.float64)
    tep_c_path = Path(sample_dir) / 'tep_C.npy'
    try:
        if tep_c_path.exists():
            C = np.load(tep_c_path)
        else:
            meta_path = Path(sample_dir) / 'sample_metadata.json'
            if not meta_path.exists():
                print(f"{tag} 跳过：tep_C.npy 不存在且缺少 sample_metadata.json，无法重建")
                return np.array([], dtype=np.float64)
            C = build_ddm_coef_matrix(sample_dir, part='all')
    except Exception as e:
        print(f"{tag} 跳过：加载/重建系数矩阵失败 — {e}")
        return np.array([], dtype=np.float64)
    if C.ndim != 2:
        print(f"{tag} 跳过：系数矩阵维度异常 {C.ndim}，期望 2 维")
        return np.array([], dtype=np.float64)
    if C.shape[1] != w_new.shape[0]:
        print(f"{tag} 跳过：C 列数 {C.shape[1]} 与 w_new 长度 {w_new.shape[0]} 不匹配")
        return np.array([], dtype=np.float64)
    if w_new.shape != w_pred.shape:
        print(f"{tag} 跳过：w_new {w_new.shape} 与 w_pred {w_pred.shape} 形状不一致")
        return np.array([], dtype=np.float64)
    try:
        diff = C.dot(w_new) - C.dot(w_pred)
        diff = np.asarray(diff, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(diff)):
            print(f"{tag} 警告：应力残差向量可能含 NaN/Inf 输入")
        return diff
    except Exception as e:
        print(f"{tag} 跳过：计算应力残差时异常 — {e}")
        return np.array([], dtype=np.float64)


def compute_stress_residual(sample_dir: str, w_new: np.ndarray, w_pred: np.ndarray) -> float:
    diff = compute_stress_residual_vector(sample_dir, w_new, w_pred)
    if diff.size == 0:
        return float('nan')
    val = float(np.linalg.norm(diff))
    if not np.isfinite(val):
        tag = f"[应力残差][{Path(sample_dir).name if sample_dir else 'None'}]"
        print(f"{tag} 警告：应力残差范数为 {val}，可能含 NaN/Inf 输入")
    return val


def fit_vector_length(vec: np.ndarray, target_len: int) -> np.ndarray:
    """将残差向量裁剪/补齐到裂缝节点长度，便于逐节点导出。"""
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if target_len <= 0:
        return np.array([], dtype=np.float64)
    if arr.size >= target_len:
        return arr[:target_len].copy()
    out = np.full(target_len, np.nan, dtype=np.float64)
    out[:arr.size] = arr
    return out

def _prewarm_matrix_caches(sample_dirs: list, workers: int) -> None:
    """已禁用：全量预热会耗尽内存（1527样本 × 矩阵尺寸 >> 可用 RAM）。
    各线程在 Phase 2 中按需加载、用完即释放，内存峰值 = post_workers × 单矩阵大小。
    """
    pass


def read_sample_metadata_info(sample_dir: str) -> dict:
    """
    从 sample_metadata.json 中读取规模、时间步、排量等信息。
    同时从目录名 sample_t{timeStep}_k{outerIter} 中解析外层迭代编号。
    返回字典包含: n_size, time_step, outer_iter, flow_rate_Q, sim_params
    """
    result = {
        'n_size': float('nan'),
        'time_step': float('nan'),
        'outer_iter': float('nan'),
        'flow_rate_Q': float('nan'),
        'sim_Ta': float('nan'),
        'sim_E': float('nan'),
        'sim_PR': float('nan'),
    }
    if not sample_dir:
        return result
    meta_path = Path(sample_dir) / 'sample_metadata.json'
    if meta_path.exists():
        try:
            with meta_path.open('r', encoding='utf-8') as f:
                meta = json.load(f)
            result['n_size'] = float(meta.get('n_size', float('nan')))
            result['time_step'] = float(meta.get('time_step', float('nan')))
            params = meta.get('parameters', {})
            if isinstance(params, dict):
                result['flow_rate_Q'] = float(params.get('Q', float('nan')))
                result['sim_Ta'] = float(params.get('Ta', float('nan')))
                result['sim_E'] = float(params.get('E', float('nan')))
                result['sim_PR'] = float(params.get('PR', float('nan')))
        except Exception:
            pass
    # 从目录名解析 outer_iter (sample_t{timeStep}_k{outerIter})
    dir_name = Path(sample_dir).name
    try:
        if '_k' in dir_name:
            result['outer_iter'] = float(dir_name.split('_k')[-1])
    except Exception:
        pass
    return result


def save_sample_plot(target, pred, w_old, fi, save_dir, sample_name, fmt='png', dpi=100):
    """
    绘制单个样本的对比图和增量图。
    使用 Figure/FigureCanvasAgg，不依赖 plt 全局状态，线程安全。
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 预测对比图
    fig = Figure(figsize=(12, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(w_old, label='Old W', color='green', lw=0.8, alpha=0.7)
    ax.plot(target, label='Target', color='orange', lw=0.8, alpha=0.8)
    ax.plot(w_old + pred, label='Predicted', color='blue', lw=0.8, alpha=0.8, linestyle='--')
    ax.set_title('Prediction vs Target')
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Width')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(save_dir / f'{sample_name}.{fmt}'), format=fmt, dpi=dpi)

    # 2. 增量对比图
    fig2 = Figure(figsize=(12, 5))
    FigureCanvasAgg(fig2)
    ax2 = fig2.add_subplot(111)
    increment_target = target - w_old
    ax2.plot(increment_target, label='Target Increment', color='orange', lw=1.0)
    ax2.plot(pred, label='Predicted Increment', color='blue', lw=1.0, linestyle='--')
    if increment_target.size > 0 and np.max(np.abs(increment_target)) > 0:
        ax2.plot(fi * (np.max(increment_target) * 0.5), label='fi (scaled)', color='grey', lw=0.5, alpha=0.3)
    ax2.set_title('Increment Comparison')
    ax2.set_xlabel('Node Index')
    ax2.set_ylabel('Increment')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(str(save_dir / f'{sample_name}_increment.{fmt}'), format=fmt, dpi=dpi)

def plot_global_metrics(all_data, save_dir):
    """
    绘制全局指标分布图和序列图，并应用过滤条件
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter: Keep samples where solve_speedup >= 10
    if 'solve_speedup' not in all_data or len(all_data['solve_speedup']) == 0:
        print("Warning: No solve_speedup data found for plotting.")
        return

    # Convert to numpy for easier indexing
    data_np = {k: np.array(v) for k, v in all_data.items()}
    
    mask = data_np['solve_speedup'] >= 10
    n_original = len(data_np['solve_speedup'])
    n_filtered = np.sum(mask)
    
    print(f"\n[绘图信息] 总样本数: {n_original}")
    print(f"[绘图信息] 剔除纯求解加速比<10的样本: 剩余 {n_filtered} ({n_filtered/n_original*100:.2f}%)")
    
    if n_filtered == 0:
        print("Warning: 所有样本均被过滤，跳过绘图。")
        return
        
    filtered_data = {k: v[mask] for k, v in data_np.items()}
    
    # Metrics to plot
    metrics_map = {
        'relative_l2': 'Relative L2 Error',
        'increment_l2': 'Increment L2 Relative Error',
        'max_error': 'Max Absolute Error',
        'max_width_error': 'Max Width Error',
        'equation_residual': 'Equation Residual',
        'stress_residual': 'Stress Residual',
        'nrmse': 'NRMSE',
        'speedup': 'Process Speedup',
        'solve_speedup': 'Pure Solve Speedup',
        'time_infer': 'Inference Time (s)',
        'time_pure_solve': 'Pure Solve Time (s)',
        'time_prep': 'Prep Time (s)',
        'gpu_memory_mb': 'GPU Memory (MB)',
        'throughput_ratio': 'Throughput Ratio (FNO/Trad)', # FNO相对于传统方法的吞吐比
        'gpu_utilization': 'GPU Utilization (%)',
        'gpu_power_watts': 'GPU Power (W)'
    }
    
    for key, label in metrics_map.items():
        if key not in filtered_data:
            continue
            
        values = filtered_data[key]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        
        # 1. Histogram
        fig = Figure(figsize=(10, 6))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.hist(values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {label} (Filtered, N={n_filtered})')
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(str(save_dir / f'hist_{key}.png'), dpi=300)

        # 2. Series Plot (Index)
        fig2 = Figure(figsize=(12, 5))
        FigureCanvasAgg(fig2)
        ax2 = fig2.add_subplot(111)
        ax2.plot(values, linewidth=0.8, alpha=0.8)
        ax2.set_title(f'{label} per Sample (Filtered, N={n_filtered})')
        ax2.set_xlabel('Sample Index (Filtered)')
        ax2.set_ylabel(label)
        ax2.grid(True, linestyle='--', alpha=0.5)
        fig2.tight_layout()
        fig2.savefig(str(save_dir / f'series_{key}.png'), dpi=300)

def clean_directory(path: Path):
    if path.exists():
        print(f"清理目录: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _process_one_sample(task: dict) -> dict:
    """
    单个样本的后处理：指标计算 + 奇异残差 + 元数据读取 + 绘图。
    在线程池中并行执行＋numpy 相关操作会释放 GIL，真实并行。
    """
    # 解包
    sample_id   = task['sample_id']
    n_i         = task['n_i']
    cond        = task['cond']
    pred_norm   = task['pred_norm']       # torch.Tensor CPU
    target_raw  = task['target_raw']      # torch.Tensor CPU
    w_old_norm  = task['w_old_norm']      # torch.Tensor CPU
    fi_raw      = task['fi_raw']          # np.ndarray
    scaler      = task['scaler']
    t_prep      = task['t_prep']
    t_old       = task['t_old']
    t_pure      = task['t_pure']
    t_infer     = task['t_infer']
    gpu_mem     = task['gpu_mem']
    gpu_util    = task['gpu_util']
    gpu_power   = task['gpu_power']
    sample_dir  = task['sample_dir']
    ddm_info    = task['ddm_info']
    cb_raw      = task.get('cb_raw')      # np.ndarray 或 None
    figures_dir = task['figures_dir']
    plot_interval = task['plot_interval']
    plot_fmt    = task['plot_fmt']
    trainer     = task['trainer']
    save_timestep_csv = task.get('save_timestep_csv', [])

    # 反归一化
    pred_raw  = trainer._denormalize_tensor(pred_norm, scaler, 'pred')
    w_old_raw = trainer._denormalize_tensor(w_old_norm, scaler, 'w_old')
    pred_final = w_old_raw + pred_raw

    # 基础指标
    rel_l2, nrmse = trainer._compute_relative_l2_and_nrmse(target_raw, pred_final)
    increment_true = target_raw - w_old_raw
    increment_l2 = compute_increment_relative_l2(pred_raw, increment_true)
    max_err = torch.max(torch.abs(pred_final - target_raw)).item()
    max_width_err = torch.abs(torch.max(target_raw) - torch.max(pred_final)).item()

    # 时间指标
    t_new = t_prep + t_infer
    speedup = t_old / t_new if t_new > 1e-6 else 0.0
    solve_speedup = t_pure / t_infer if t_infer > 1e-9 else 0.0
    throughput_fno  = 1.0 / t_new if t_new > 1e-9 else 0.0
    throughput_trad = 1.0 / t_old if t_old > 1e-9 else 0.0
    throughput_ratio = throughput_fno / throughput_trad if throughput_trad > 1e-9 else 0.0

    # 方程残差
    equation_residual = float('nan')
    equation_residual_vec = np.array([], dtype=np.float64)
    if cb_raw is not None:
        ntip_count = None
        if ddm_info and isinstance(ddm_info, dict):
            ranges = ddm_info.get('ranges', {})
            ntip = ranges.get('ntip', {}) if isinstance(ranges, dict) else {}
            ntip_count = ntip.get('count') if isinstance(ntip, dict) else None
        if ntip_count is not None:
            cb_modified = cb_raw.copy()
            cb_modified[: int(ntip_count)] = pred_final.numpy()[: int(ntip_count)]
            equation_residual_vec = compute_equation_residual_vector(sample_dir, cb_modified)
            equation_residual = float(np.linalg.norm(equation_residual_vec)) if equation_residual_vec.size else float('nan')

    # 应力残差
    stress_residual_vec = compute_stress_residual_vector(
        sample_dir, target_raw.numpy(), pred_final.numpy()
    )
    stress_residual = float(np.linalg.norm(stress_residual_vec)) if stress_residual_vec.size else float('nan')

    # 元数据信息
    sample_info = read_sample_metadata_info(sample_dir)

    # 按需携带指定时间步的预测/真实值数组（用于落盘 CSV）
    csv_data = None
    if save_timestep_csv:
        ts = sample_info['time_step']
        if np.isfinite(ts) and int(ts) in save_timestep_csv:
            csv_data = {
                'prediction':        pred_final.numpy().copy(),
                'target':            target_raw.numpy().copy(),
                'n_size':            int(pred_final.shape[0]),
                'time_step':         int(ts),
                'outer_iter':        sample_info['outer_iter'],
                'cond':              cond,
                'equation_residual': equation_residual,
                'equation_residual': fit_vector_length(equation_residual_vec, int(pred_final.shape[0])),
                'stress_residual':   fit_vector_length(stress_residual_vec, int(pred_final.shape[0])),
            }

    # 绘图
    if sample_id % plot_interval == 0:
        save_sample_plot(
            target=target_raw.numpy(),
            pred=pred_raw.numpy(),
            w_old=w_old_raw.numpy(),
            fi=fi_raw,
            save_dir=figures_dir / cond,
            sample_name=f'sample_{sample_id}',
            fmt=plot_fmt,
        )

    return {
        'sample_id': sample_id,
        'cond': cond,
        'csv_data': csv_data,
        'rel_l2': rel_l2,
        'nrmse': nrmse,
        'increment_l2': increment_l2,
        'max_err': max_err,
        'max_width_err': max_width_err,
        'speedup': speedup,
        'solve_speedup': solve_speedup,
        'throughput_fno': throughput_fno,
        'throughput_trad': throughput_trad,
        'throughput_ratio': throughput_ratio,
        'equation_residual': equation_residual,
        'stress_residual': stress_residual,
        't_infer': t_infer,
        't_prep': t_prep,
        't_old': t_old,
        't_pure': t_pure,
        'gpu_mem': gpu_mem,
        'gpu_util': gpu_util,
        'gpu_power': gpu_power,
        'sample_info': sample_info,
    }


def main(config):
    # Setup Paths
    testing_cfg = config.get('testing', {})
    checkpoint_path = Path(testing_cfg['checkpoint_path'])

    # Keep test-specific data/testing settings, but align model architecture with checkpoint training config.
    config = _merge_model_config_from_checkpoint(config, checkpoint_path)

    # Handle safe globals for torch load if needed (reusing logic from trainer)
    torch.serialization.add_safe_globals([block for block in [torch.nn.GELU] if hasattr(torch.nn, 'GELU')])
    checkpoint_obj = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = _remap_legacy_neuralop_keys(_extract_state_dict(checkpoint_obj))
    _inject_fno_ratio_from_state_dict(config, state_dict)

    exp_dir = Path(testing_cfg.get('exp_dir', 'experiments/testing_outputs'))
    figures_dir = exp_dir / 'figures'
    metrics_dir = exp_dir / 'metrics'
    
    # Clean output directory
    clean_directory(exp_dir)
    figures_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    
    # Load Dataset
    print("正在准备测试数据集...")
    _, _, test_dataset, collate_fn = prepare_datasets(config)

    # Setup Device (must be before DataLoader for pin_memory)
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu')
    print(f"使用设备: {device}")

    num_workers = testing_cfg.get('num_workers', 0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(num_workers > 0 and device.type == 'cuda'),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
    print(f"测试集样本数: {len(test_dataset)}")
    
    # Initialize NVML
    if pynvml and device.type == 'cuda':
        try:
            pynvml.nvmlInit()
            print("NVML Initialized for GPU monitoring")
        except Exception as e:
            print(f"Failed to initialize NVML: {e}")

    # Load Model
    print("加载模型...")
    model = FluidSolidFNOmodel(config)
    print(f"加载权重: {checkpoint_path}")

    filtered_state = _filter_compatible_state_dict(model, state_dict)
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()
    
    # Initialize Trainer just for helper methods (normalization, etc) if needed, 
    # but here we can do manual inference to have full control over plotting loop.
    # Actually, let's just do manual loop to be precise about "per condition saving".
    
    trainer = FluidSolidTrainer(config, model, 
                                AdamW(model.parameters()), 
                                None, 
                                loss_fn(config['training']['loss_type']), 
                                None, None, test_loader, device, exp_dir)
    
    # Metrics Container
    total_metrics = defaultdict(float)
    global_metric_values = defaultdict(list)
    condition_metrics = defaultdict(lambda: {
        'relative_l2': [], 'increment_l2': [], 'nrmse': [], 
        'max_error': [], 'max_width_error': [],
        'equation_residual': [], 'stress_residual': [],
        'time_infer': [], 'time_prep': [], 
        'time_old': [], 'time_pure_solve': [],
        'speedup': [], 'solve_speedup': [],
        'gpu_memory_mb': [], 'throughput_ratio': [],
        'gpu_utilization': [], 'gpu_power_watts': [],
        'n_size': [], 'time_step': [], 'outer_iter': [],
        'flow_rate_Q': [], 'sim_Ta': [], 'sim_E': [], 'sim_PR': []
    })
    all_sample_records = []

    plot_interval = testing_cfg.get('plot_interval', 5)
    plot_fmt      = testing_cfg.get('plot_format', 'png')
    post_workers  = testing_cfg.get('post_workers', 4)  # 后处理并行线程数
    save_timestep_csv = [int(t) for t in testing_cfg.get('save_timestep_csv', [])]

    total_start_time = time.perf_counter()
    sample_count = 0
    pending_tasks = []   # 收集每个样本的后处理任务描述

    print("开始测试（Phase 1: GPU 推理）...")

    # ── Phase 1: 纯 GPU 推理，尽量让 GPU 保持满负荷 ──────────────────────
    with torch.inference_mode():
        for batch_idx, batch in enumerate(test_loader):
            trainer._ensure_amp_compatibility(batch)

            batch_prep_time      = batch['prep_time'].numpy()      if 'prep_time'       in batch else np.zeros(batch['tipn'].shape[0])
            batch_solver_time    = batch['solver_time'].numpy()    if 'solver_time'     in batch else np.zeros(batch['tipn'].shape[0])
            batch_pure_solve_time= batch['pure_solve_time'].numpy()if 'pure_solve_time' in batch else np.zeros(batch['tipn'].shape[0])

            batch = trainer.batch_to_device(batch)

            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.cuda.amp.autocast(enabled=trainer.use_amp):
                w_pred = model(batch)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            gpu_util, gpu_power = 0.0, 0.0
            if device.type == 'cuda':
                gpu_util, gpu_power = get_gpu_metrics(device.index if device.index else 0)

            batch_infer_cycle_time = time.perf_counter() - t0
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == 'cuda' else 0

            batch_size   = batch['tipn'].shape[0]
            avg_infer_time = batch_infer_cycle_time / batch_size

            n_list      = batch['n']
            scalers     = batch.get('scalers')
            conditions  = batch.get('condition', ['unknown'] * batch_size)
            sample_dirs = batch.get('sample_dir', [None] * batch_size)
            ddm_infos   = batch.get('ddm_info',   [None] * batch_size)

            for i in range(batch_size):
                n_i   = int(n_list[i])
                fi_raw = batch['fi'][i, :n_i].cpu().numpy()
                if fi_raw.ndim == 0:
                    fi_raw = np.zeros(n_i)

                cb_raw = batch['cb'][i, :n_i].cpu().numpy() if 'cb' in batch else None

                pending_tasks.append({
                    'sample_id':    sample_count + i,
                    'n_i':          n_i,
                    'cond':         conditions[i],
                    'pred_norm':    w_pred[i, :n_i].cpu().clone(),
                    'target_raw':   batch['w_new'][i, :n_i].cpu(),
                    'w_old_norm':   batch['w_old'][i, :n_i].cpu(),
                    'fi_raw':       fi_raw,
                    'scaler':       scalers[i] if scalers else None,
                    't_prep':       float(batch_prep_time[i]),
                    't_old':        float(batch_solver_time[i]),
                    't_pure':       float(batch_pure_solve_time[i]),
                    't_infer':      avg_infer_time,
                    'gpu_mem':      gpu_mem,
                    'gpu_util':     gpu_util,
                    'gpu_power':    gpu_power,
                    'sample_dir':   sample_dirs[i],
                    'ddm_info':     ddm_infos[i] if i < len(ddm_infos) else None,
                    'cb_raw':       cb_raw,
                    'figures_dir':  figures_dir,
                    'plot_interval':plot_interval,
                    'plot_fmt':     plot_fmt,
                    'trainer':      trainer,
                    'save_timestep_csv': save_timestep_csv,
                })

            sample_count += batch_size
            if sample_count % 50 == 0:
                print(f"\r[Phase 1] 已推理: {sample_count}/{len(test_dataset)}", end="")

    print(f"\n[Phase 1 完成] 共 {sample_count} 个样本，开始 Phase 2 并行后处理 (workers={post_workers})...")

    # ── Phase 2: 多线程并行后处理（残差/指标/绘图）────────────────────────
    results = [None] * len(pending_tasks)
    with ThreadPoolExecutor(max_workers=post_workers) as executor:
        future_to_idx = {executor.submit(_process_one_sample, task): idx
                         for idx, task in enumerate(pending_tasks)}
        done_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"\n[警告] 样本 {idx} 后处理失败: {e}")
                results[idx] = None
            done_count += 1
            if done_count % 50 == 0:
                print(f"\r[Phase 2] 已完成: {done_count}/{len(pending_tasks)}", end="")

    print(f"\n[Phase 2 完成]")

    # ── 汇总指标（单线程，顺序合并结果）────────────────────────────────────
    for res in results:
        if res is None:
            continue
        cond        = res['cond']
        rel_l2      = res['rel_l2']
        nrmse       = res['nrmse']
        speedup     = res['speedup']
        solve_sp    = res['solve_speedup']
        sample_info = res['sample_info']

        total_metrics['relative_l2']  += rel_l2
        total_metrics['nrmse']        += nrmse
        total_metrics['speedup']      += speedup
        total_metrics['solve_speedup']+= solve_sp

        global_metric_values['increment_l2'].append(res['increment_l2'])
        global_metric_values['max_width_error'].append(res['max_width_err'])
        global_metric_values['equation_residual'].append(res['equation_residual'])
        global_metric_values['stress_residual'].append(res['stress_residual'])

        condition_metrics[cond]['relative_l2'].append(rel_l2)
        condition_metrics[cond]['increment_l2'].append(res['increment_l2'])
        condition_metrics[cond]['nrmse'].append(nrmse)
        condition_metrics[cond]['max_error'].append(res['max_err'])
        condition_metrics[cond]['max_width_error'].append(res['max_width_err'])
        condition_metrics[cond]['equation_residual'].append(res['equation_residual'])
        condition_metrics[cond]['stress_residual'].append(res['stress_residual'])
        condition_metrics[cond]['time_infer'].append(res['t_infer'])
        condition_metrics[cond]['time_prep'].append(res['t_prep'])
        condition_metrics[cond]['time_old'].append(res['t_old'])
        condition_metrics[cond]['time_pure_solve'].append(res['t_pure'])
        condition_metrics[cond]['speedup'].append(speedup)
        condition_metrics[cond]['solve_speedup'].append(solve_sp)
        condition_metrics[cond]['gpu_memory_mb'].append(res['gpu_mem'])
        condition_metrics[cond]['throughput_ratio'].append(res['throughput_ratio'])
        condition_metrics[cond]['gpu_utilization'].append(res['gpu_util'])
        condition_metrics[cond]['gpu_power_watts'].append(res['gpu_power'])
        condition_metrics[cond]['n_size'].append(sample_info['n_size'])
        condition_metrics[cond]['time_step'].append(sample_info['time_step'])
        condition_metrics[cond]['outer_iter'].append(sample_info['outer_iter'])
        condition_metrics[cond]['flow_rate_Q'].append(sample_info['flow_rate_Q'])
        condition_metrics[cond]['sim_Ta'].append(sample_info['sim_Ta'])
        condition_metrics[cond]['sim_E'].append(sample_info['sim_E'])
        condition_metrics[cond]['sim_PR'].append(sample_info['sim_PR'])

        all_sample_records.append({
            'sample_id':        res['sample_id'],
            'condition':        cond,
            'n_size':           sample_info['n_size'],
            'time_step':        sample_info['time_step'],
            'outer_iter':       sample_info['outer_iter'],
            'flow_rate_Q':      sample_info['flow_rate_Q'],
            'sim_Ta':           sample_info['sim_Ta'],
            'sim_E':            sample_info['sim_E'],
            'sim_PR':           sample_info['sim_PR'],
            'relative_l2':      rel_l2,
            'increment_l2':     res['increment_l2'],
            'max_error':        res['max_err'],
            'max_width_error':  res['max_width_err'],
            'equation_residual':res['equation_residual'],
            'stress_residual':  res['stress_residual'],
            'nrmse':            nrmse,
            'time_infer':       res['t_infer'],
            'time_prep':        res['t_prep'],
            'time_solver_total':res['t_old'],
            'time_solver_pure': res['t_pure'],
            'speedup_process':  speedup,
            'speedup_pure_solve':solve_sp,
            'throughput_fno':   res['throughput_fno'],
            'throughput_trad':  res['throughput_trad'],
            'throughput_ratio': res['throughput_ratio'],
            'gpu_memory_mb':    res['gpu_mem'],
            'gpu_utilization':  res['gpu_util'],
            'gpu_power_watts':  res['gpu_power'],
        })

    # ── 保存指定时间步的预测/真实值 CSV ──────────────────────────────────
    if save_timestep_csv:
        ts_csv_dir = exp_dir / 'timestep_csvs'
        ts_csv_dir.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        for res in results:
            if res is None or res.get('csv_data') is None:
                continue
            d = res['csv_data']
            ts  = d['time_step']
            ok  = d['outer_iter']
            n   = d['n_size']
            safe_cond = str(d['cond']).replace('/', '_').replace('\\', '_')
            ok_str = f"_k{int(ok):02d}" if np.isfinite(ok) else ''
            fname = f"prediction_target_{safe_cond}_t{ts:04d}{ok_str}_n{n}.csv"
            df_csv = pd.DataFrame({
                'Node_Index':        np.arange(n),
                'Prediction':        d['prediction'],
                'Target':            d['target'],
                'Equation_Residual': d['equation_residual'],
                'Stress_Residual':   d['stress_residual'],
            })
            df_csv.to_csv(ts_csv_dir / fname, index=False)
            saved_count += 1
        print(f"指定时间步 CSV 已保存 {saved_count} 个文件至: {ts_csv_dir}")

    total_duration = time.perf_counter() - total_start_time
    print(f"\n测试完成。总耗时: {total_duration:.2f}s")
    
    # Finalize Metrics
    all_infer_times = [t for m in condition_metrics.values() for t in m['time_infer']]
    all_latencies = np.sort(all_infer_times)
    total_samples = len(all_latencies)
    
    global_p90 = all_latencies[int(total_samples * 0.9)] if total_samples > 0 else 0
    global_p99 = all_latencies[int(total_samples * 0.99)] if total_samples > 0 else 0
    global_std = np.std(all_infer_times) if total_samples > 0 else 0

    final_global_metrics = {
        'total_samples': sample_count,
        'total_duration_seconds': total_duration,
        
        'avg_inference_time_per_sample': sum(all_infer_times) / total_samples if total_samples else 0,
        'latency_p90': float(global_p90),
        'latency_p99': float(global_p99),
        'latency_std': float(global_std),
        'avg_relative_l2': total_metrics['relative_l2'] / sample_count if sample_count else 0,
        'avg_increment_l2': safe_nanmean(global_metric_values['increment_l2']),
        'avg_nrmse': total_metrics['nrmse'] / sample_count if sample_count else 0,
        'avg_max_width_error': safe_nanmean(global_metric_values['max_width_error']),
        'avg_equation_residual': safe_nanmean(global_metric_values['equation_residual']),
        'avg_stress_residual': safe_nanmean(global_metric_values['stress_residual']),
        'avg_speedup_process': total_metrics['speedup'] / sample_count if sample_count else 0,
        'avg_speedup_pure_solve': total_metrics['solve_speedup'] / sample_count if sample_count else 0,
        'model_params': sum(p.numel() for p in model.parameters())
    }
    
    # Aggregating per condition stats
    final_cond_metrics = {}
    for cond, m in condition_metrics.items():
        count = len(m['relative_l2'])
        # Calculate Latency Percentiles
        latencies = np.sort(m['time_infer'])
        p90 = latencies[int(count * 0.9)] if count > 0 else 0
        p95 = latencies[int(count * 0.95)] if count > 0 else 0
        p99 = latencies[int(count * 0.99)] if count > 0 else 0
        std_dev = np.std(m['time_infer']) if count > 0 else 0

        final_cond_metrics[cond] = {
            'count': count,
            'avg_n_size': safe_nanmean(m['n_size']),
            'avg_time_step': safe_nanmean(m['time_step']),
            'avg_outer_iter': safe_nanmean(m['outer_iter']),
            'flow_rate_Q': safe_nanmean(m['flow_rate_Q']),
            'sim_Ta': safe_nanmean(m['sim_Ta']),
            'sim_E': safe_nanmean(m['sim_E']),
            'sim_PR': safe_nanmean(m['sim_PR']),
            'avg_relative_l2': sum(m['relative_l2']) / count,
            'avg_increment_l2': safe_nanmean(m['increment_l2']),
            'avg_nrmse': sum(m['nrmse']) / count,
            'avg_time_infer': sum(m['time_infer']) / count,
            'latency_p90': float(p90),
            'latency_p99': float(p99),
            'latency_std': float(std_dev),
            'avg_time_prep': sum(m['time_prep']) / count,
            'avg_time_old': sum(m['time_old']) / count,
            'avg_time_pure_solve': sum(m['time_pure_solve']) / count,
            'avg_speedup_process': sum(m['speedup']) / count,
            'avg_speedup_pure_solve': sum(m['solve_speedup']) / count,
            'avg_max_width_error': safe_nanmean(m['max_width_error']),
            'avg_equation_residual': safe_nanmean(m['equation_residual']),
            'avg_stress_residual': safe_nanmean(m['stress_residual']),
            'avg_gpu_utilization': sum(m['gpu_utilization']) / count if m['gpu_utilization'] else 0,
            'avg_gpu_power': sum(m['gpu_power_watts']) / count if m['gpu_power_watts'] else 0,
        }
        
    # Save Metrics
    with open(metrics_dir / 'global_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(final_global_metrics, f, indent=4, ensure_ascii=False)
        
    with open(metrics_dir / 'condition_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(final_cond_metrics, f, indent=4, ensure_ascii=False)
        
    # Save Comprehensive CSV
    if all_sample_records:
        df_all = pd.DataFrame(all_sample_records)
        csv_path = metrics_dir / 'metrics_comprehensive.csv'
        df_all.to_csv(csv_path, index=False)
        print(f"完整指标CSV已保存至: {csv_path}")
        
    # --- Generate Global Plots ---
    print("正在生成全局指标图表...")
    all_metrics_flat = defaultdict(list)
    for m in condition_metrics.values():
        for k, v in m.items():
            all_metrics_flat[k].extend(v)
            
    plot_global_metrics(all_metrics_flat, figures_dir / 'global_plots')
        
    # Save Time Plots per condition (optional, sticking to requested structure)
    
    for cond, m in condition_metrics.items():
        cond_dir = figures_dir / cond
        cond_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sample metrics CSV in condition dir
        df = pd.DataFrame({
            'sample_id': range(len(m['time_infer'])),
            'n_size': m['n_size'],
            'time_step': m['time_step'],
            'outer_iter': m['outer_iter'],
            'flow_rate_Q': m['flow_rate_Q'],
            'sim_Ta': m['sim_Ta'],
            'sim_E': m['sim_E'],
            'sim_PR': m['sim_PR'],
            'time_infer': m['time_infer'],
            'time_prep': m['time_prep'],
            'time_old': m['time_old'],
            'time_pure_solve': m['time_pure_solve'],
            'speedup_process': m['speedup'],
            'speedup_pure_solve': m['solve_speedup'],
            'relative_l2': m['relative_l2'],
            'increment_l2': m['increment_l2'],
            'max_width_error': m['max_width_error'],
            'equation_residual': m['equation_residual'],
            'stress_residual': m['stress_residual'],
            'nrmse': m['nrmse']
        })
        df.to_csv(cond_dir / 'metrics.csv', index=False)
        
        # Save condition summary text
        with open(cond_dir / 'summary.txt', 'w') as f:
            f.write(f"Condition: {cond}\n")
            f.write(f"Count: {len(m['time_infer'])}\n")
            f.write(f"Avg N Size: {safe_nanmean(m['n_size']):.1f}\n")
            f.write(f"Avg Time Step: {safe_nanmean(m['time_step']):.1f}\n")
            f.write(f"Avg Outer Iter: {safe_nanmean(m['outer_iter']):.2f}\n")
            f.write(f"Flow Rate Q: {safe_nanmean(m['flow_rate_Q']):.6g}\n")
            f.write(f"Sim Ta: {safe_nanmean(m['sim_Ta']):.1f}\n")
            f.write(f"Sim E: {safe_nanmean(m['sim_E']):.1f}\n")
            f.write(f"Sim PR: {safe_nanmean(m['sim_PR']):.4f}\n")
            f.write(f"Avg Time Infer: {sum(m['time_infer'])/len(m['time_infer']):.6f}s\n")
            f.write(f"Avg Time Prep: {sum(m['time_prep'])/len(m['time_prep']):.6f}s\n")
            f.write(f"Avg Time Old Total: {sum(m['time_old'])/len(m['time_old']):.6f}s\n")
            f.write(f"Avg Time Pure Solve: {sum(m['time_pure_solve'])/len(m['time_pure_solve']):.6f}s\n")
            f.write(f"Avg Speedup Process: {sum(m['speedup'])/len(m['speedup']):.2f}x\n")
            f.write(f"Avg Speedup Pure Solve: {sum(m['solve_speedup'])/len(m['solve_speedup']):.2f}x\n")
            f.write(f"Avg Rel L2: {sum(m['relative_l2'])/len(m['relative_l2']):.6f}\n")
            f.write(f"Avg Increment L2: {safe_nanmean(m['increment_l2']):.6f}\n")
            f.write(f"Avg Max Width Error: {safe_nanmean(m['max_width_error']):.6f}\n")
            f.write(f"Avg Equation Residual: {safe_nanmean(m['equation_residual']):.6f}\n")
            f.write(f"Avg Stress Residual: {safe_nanmean(m['stress_residual']):.6f}\n")

    print(f"结果已保存至: {exp_dir}")

if __name__ == '__main__':
    with open('test_config.yml', 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    main(config)
