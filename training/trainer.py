import json
import math
import os
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
try:
    from torch.amp import autocast, GradScaler
    AMP_MODE = 'torch_amp'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_MODE = 'cuda_amp'
from common.visualization import draw_all, plot_metric_series


class _FluidSolidTrainer:
    def __init__(
        self,
        config: Dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: torch.device,
        exp_dir: Union[Path, str],
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.exp_dir = Path(exp_dir)
        self.amp_device_type = 'cuda' if 'cuda' in self.device.type.lower() else 'cpu'
        # 强制禁用 AMP，因为旧版 PyTorch 下 cuFFT 的 fp16 反向传播存在已知导致的 CUFFT_INTERNAL_ERROR bug
        self.use_amp = False # bool(self.config['training'].get('amp', False)) and self.amp_device_type == 'cuda'
        if self.use_amp:
            if AMP_MODE == 'torch_amp':
                self.scaler = GradScaler(self.amp_device_type, enabled=True)
            else:
                self.scaler = GradScaler(enabled=True)
        else:
            if AMP_MODE == 'torch_amp':
                self.scaler = GradScaler(self.amp_device_type if self.amp_device_type else 'cuda', enabled=False)
            else:
                self.scaler = GradScaler(enabled=False)
        self._amp_disabled_reason: Union[str, None] = None
        self.best_checkpoint: Union[Path, str] = ''
        self.best_val_loss: float = float('inf')
        self.early_patience: int = int(self.config['training']['early_patience'])
        self.patience: int = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': [],
            'train_relative_l2': [],
            'val_relative_l2': [],
            'train_nrmse': [],
            'val_nrmse': [],
        }

    def train(self) -> Tuple[nn.Module, Union[Path, str]]:
        epochs = int(self.config['training']['epochs'])
        for epoch in range(epochs):
            train_loss_epoch, train_r2_epoch, train_rel_l2_epoch, train_nrmse_epoch = self.train_one_epoch()
            self.history['train_loss'].append(train_loss_epoch)
            self.history['train_r2'].append(train_r2_epoch)
            self.history['train_relative_l2'].append(train_rel_l2_epoch)
            self.history['train_nrmse'].append(train_nrmse_epoch)

            val_loss_epoch, val_r2_epoch, val_rel_l2_epoch, val_nrmse_epoch = self.validate()
            self.history['val_loss'].append(val_loss_epoch)
            self.history['val_r2'].append(val_r2_epoch)
            self.history['val_relative_l2'].append(val_rel_l2_epoch)
            self.history['val_nrmse'].append(val_nrmse_epoch)

            if self.scheduler is not None:
                # self.scheduler.step(float(val_loss_epoch))
                self.scheduler.step(epoch)

            if self.early_stopping(val_loss_epoch, epoch):
                break

            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Train Loss: {train_loss_epoch:.6f}, Val Loss: {val_loss_epoch:.6f}, "
                f"Train R2: {train_r2_epoch:.4f}, Val R2: {val_r2_epoch:.4f}, "
                f"lr: {self.optimizer.param_groups[0]['lr']}"
            )

        completed_epochs = len(self.history['train_loss'])
        print(
            "Training completed: \n"
            f" - Best validation loss: {self.best_val_loss}\n"
            f" - Epoch {completed_epochs}\n"
            f" - Best checkpoint: {self.best_checkpoint}\n"
            f" - Saved figures to {self.exp_dir}/figures\n"
        )
        # 保存最后一轮验证集的可视化结果（逐样本时间序列与真实值-预测值散点图）
        try:
            self.save_last_epoch_validation_visuals(max_samples=50)
        except Exception as e:
            print(f"保存最后一轮验证集可视化时发生错误: {e}")

        self.save_metrics()
        return self.model, self.best_checkpoint

    def _get_autocast_context(self):
        """Helper to create the correct autocast context manager based on PyTorch version."""
        if AMP_MODE == 'torch_amp':
            return autocast(device_type=self.amp_device_type, enabled=self.use_amp)
        else:
            return autocast(enabled=self.use_amp)

    def train_one_epoch(self) -> Tuple[float, float, float, float]:
        self.model.train()
        total_loss = 0.0
        ss_res, ss_tot = 0.0, 0.0
        sample_count = 0
        rel_l2_sum, nrmse_sum = 0.0, 0.0
        from tqdm import tqdm

        for batch in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            if batch['w_old'].shape[0] != self.config['data']['batch_size']:
                continue

            batch = self.batch_to_device(batch)
            self._ensure_amp_compatibility(batch)
            with self._get_autocast_context():
                w_pred = self.model(batch)
                loss = self.criterion(w_pred, batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

            with torch.no_grad():
                batch_size = batch['tipn'].shape[0]
                scalers = batch.get('scalers')
                for i in range(batch_size):
                    n_i = int(batch['n'][i].item()) if isinstance(batch['n'], torch.Tensor) else int(batch['n'][i])
                    scaler = scalers[i] if scalers else None
                    
                    targets = batch['w_new'][i, :n_i].detach().cpu()
                    
                    # Denormalize w_old and pred
                    old_norm = batch['w_old'][i, :n_i].detach().cpu()
                    pred_norm = w_pred[i, :n_i].detach().cpu()
                    
                    old_raw = self._denormalize_tensor(old_norm, scaler, 'w_old')
                    pred_raw = self._denormalize_tensor(pred_norm, scaler, 'pred')
                    
                    preds = old_raw + pred_raw
                    
                    resid = targets - preds
                    ss_res += float(torch.sum(resid ** 2))
                    mean_true = torch.mean(targets)
                    ss_tot += float(torch.sum((targets - mean_true) ** 2))
                    sample_count += 1
                    rel_l2_val, nrmse_val = self._compute_relative_l2_and_nrmse(targets, preds)
                    rel_l2_sum += rel_l2_val
                    nrmse_sum += nrmse_val

        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else float('nan')
        train_r2 = self._compute_r2(ss_res, ss_tot, sample_count)
        train_rel_l2 = rel_l2_sum / sample_count if sample_count > 0 else float('nan')
        train_nrmse = nrmse_sum / sample_count if sample_count > 0 else float('nan')
        return avg_loss, train_r2, train_rel_l2, train_nrmse

    def batch_to_device(self, batch: Dict) -> Dict:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch

    def validate(self) -> Tuple[float, float, float, float]:
        self.model.eval()
        total_loss = 0.0
        ss_res, ss_tot = 0.0, 0.0
        sample_count = 0
        rel_l2_sum, nrmse_sum = 0.0, 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self.batch_to_device(batch)
                self._ensure_amp_compatibility(batch)
                batch_size = batch['tipn'].shape[0]
                with self._get_autocast_context():
                    w_pred = self.model(batch)
                    loss = self.criterion(w_pred, batch)
                total_loss += loss.item()

                scalers = batch.get('scalers')
                for i in range(batch_size):
                    n_i = int(batch['n'][i].item()) if isinstance(batch['n'], torch.Tensor) else int(batch['n'][i])
                    scaler = scalers[i] if scalers else None
                    
                    targets = batch['w_new'][i, :n_i].detach().cpu()
                    
                    # Denormalize w_old and pred
                    old_norm = batch['w_old'][i, :n_i].detach().cpu()
                    pred_norm = w_pred[i, :n_i].detach().cpu()
                    
                    old_raw = self._denormalize_tensor(old_norm, scaler, 'w_old')
                    pred_raw = self._denormalize_tensor(pred_norm, scaler, 'pred')
                    
                    preds = old_raw + pred_raw
                    
                    resid = targets - preds
                    ss_res += float(torch.sum(resid ** 2))
                    mean_true = torch.mean(targets)
                    ss_tot += float(torch.sum((targets - mean_true) ** 2))
                    sample_count += 1
                    rel_l2_val, nrmse_val = self._compute_relative_l2_and_nrmse(targets, preds)
                    rel_l2_sum += rel_l2_val
                    nrmse_sum += nrmse_val

        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else float('nan')
        if sample_count > 0 and ss_tot < 1e-12:
            print('警告: 验证集目标方差极小，R2 可能不稳定。')
        val_r2 = self._compute_r2(ss_res, ss_tot, sample_count)
        val_rel_l2 = rel_l2_sum / sample_count if sample_count > 0 else float('nan')
        val_nrmse = nrmse_sum / sample_count if sample_count > 0 else float('nan')
        return avg_loss, val_r2, val_rel_l2, val_nrmse

    def test(self, max_samples: Union[int, None] = None, show_progress: bool = False) -> Tuple[list, list, list, list, list, Dict[str, float], Dict[str, List[float]]]:
        self.model.eval()
        predictions, targets, w_old, fi, ns = [], [], [], [], []
        times = []
        metrics = {"mse": 0.0, "mae": 0.0, "increment_mse": 0.0, "relative_l2": 0.0, "nrmse": 0.0}
        per_sample_metrics: Dict[str, List[float]] = {
            'relative_l2': [],
            'nrmse': [],
            'mse': [],
            'time': [],
            'condition': [],
        }
        sample_count = 0

        with torch.no_grad():
            loader = self.test_loader
            if show_progress:
                from tqdm import tqdm
                loader = tqdm(self.test_loader, desc='Testing', unit='batch')
            for batch in loader:
                batch = self.batch_to_device(batch)
                self._ensure_amp_compatibility(batch)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()

                with self._get_autocast_context():
                    w_pred = self.model(batch)
                    
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.time()
                batch_cost = t1 - t0
                
                batch_size = batch['tipn'].shape[0]
                avg_cost = batch_cost / batch_size
                scalers = batch.get('scalers', None)

                conditions = batch.get('condition')
                for i in range(batch_size):
                    n_i = int(batch['n'][i].item()) if isinstance(batch['n'], torch.Tensor) else int(batch['n'][i])
                    pred_norm = w_pred[i, :n_i].detach().cpu()
                    target_raw = batch['w_new'][i, :n_i].detach().cpu()
                    old_norm = batch['w_old'][i, :n_i].detach().cpu()
                    scaler = None
                    if scalers and i < len(scalers):
                        scaler = scalers[i]

                    # Denormalize
                    pred_raw = self._denormalize_tensor(pred_norm, scaler, 'pred')
                    old_raw = self._denormalize_tensor(old_norm, scaler, 'w_old')
                    
                    pred_full_raw = old_raw + pred_raw # 预测的新裂缝宽度
                    increment_true_raw = target_raw - old_raw # 真实的增量宽度

                    sample_mse = torch.mean((pred_full_raw - target_raw) ** 2).item()
                    metrics['mse'] += sample_mse
                    metrics['mae'] += torch.mean(torch.abs(pred_full_raw - target_raw)).item()
                    metrics['increment_mse'] += torch.mean((pred_raw - increment_true_raw) ** 2).item()
                    rel_l2_val, nrmse_val = self._compute_relative_l2_and_nrmse(target_raw, pred_full_raw)
                    metrics['relative_l2'] += rel_l2_val
                    metrics['nrmse'] += nrmse_val
                    per_sample_metrics['relative_l2'].append(rel_l2_val)
                    per_sample_metrics['nrmse'].append(nrmse_val)
                    per_sample_metrics['mse'].append(sample_mse)
                    per_sample_metrics['time'].append(avg_cost)
                    if conditions is not None and i < len(conditions):
                        per_sample_metrics['condition'].append(conditions[i])
                    sample_count += 1

                    predictions.append(pred_raw.numpy())
                    w_old.append(old_raw.numpy())
                    targets.append(target_raw.numpy())
                    fi.append(batch['fi'][i].detach().cpu().numpy())
                    ns.append(int(batch['n'][i].detach().cpu().item()) if isinstance(batch['n'], torch.Tensor) else int(batch['n'][i]))
                    times.append(avg_cost)

                    if max_samples is not None and sample_count >= max_samples:
                        break
                if max_samples is not None and sample_count >= max_samples:
                    break

        if sample_count > 0:
            for key in ('mse', 'mae', 'increment_mse', 'relative_l2', 'nrmse'):
                metrics[key] /= sample_count

        # Save time analysis
        if len(ns) > 0:
            try:
                # Group by size and take mean of time
                df_raw = pd.DataFrame({'size': ns, 'time': times})
                df_time = df_raw.groupby('size')['time'].mean().reset_index()
                df_time = df_time.sort_values(by='size')[2:]
                
                metrics_dir = self.exp_dir / 'metrics'
                metrics_dir.mkdir(parents=True, exist_ok=True)
                df_time.to_csv(metrics_dir / 'inference_time_vs_size.csv', index=False)
                print(f"Inference time data saved to {metrics_dir / 'inference_time_vs_size.csv'}")
                
                # Downsample for plotting
                df_plot = df_time
                if len(df_plot) > 50:
                    step = len(df_plot) // 50
                    df_plot = df_plot.iloc[::step]

                plt.figure(figsize=(10, 6))
                plt.plot(df_plot['size'], df_plot['time'], 'o-', linewidth=2, label='Average Inference Time')
                
                plt.xlabel('Sample Size (n)')
                plt.ylabel('Inference Time (s)')
                plt.title('Inference Time vs Sample Size')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                figures_dir = self.exp_dir / 'figures'
                figures_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(figures_dir / 'inference_time_vs_size.png', dpi=300)
                plt.close()
                print(f"Inference time plot saved to {figures_dir / 'inference_time_vs_size.png'}")
            except Exception as e:
                print(f"Error saving time analysis: {e}")

        return predictions, targets, w_old, fi, ns, metrics, per_sample_metrics

    def early_stopping(self, loss_value: float, epoch: int) -> bool:
        if loss_value is None or math.isnan(loss_value):
            raise ValueError(f'Loss is NaN! in epoch {epoch + 1}')

        if loss_value < self.best_val_loss:
            self.best_val_loss = loss_value
            self.save_checkpoint(epoch)
            self.patience = 0
        else:
            self.patience += 1
            print(f'patience: {self.patience}/{self.early_patience}')
            if self.patience > self.early_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}, "
                    f"best validation loss: {self.best_val_loss:.6f}"
                )
                if self.best_checkpoint:
                    self.load_checkpoint(self.best_checkpoint)
                return True
        return False

    def save_checkpoint(self, epoch: int) -> None:
        checkpoint_dir = self.exp_dir / 'checkpoint'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = checkpoint_dir / 'best_model.pth'
        self.best_checkpoint = checkpoint_path
        print(f"保存模型参数到: {checkpoint_path}")
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> nn.Module:
        torch.serialization.add_safe_globals([nn.GELU, torch._C._nn.gelu])
        try:
            import neuralop
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            neuralop = None
        if neuralop is not None:
            torch.serialization.add_safe_globals([neuralop.layers.spectral_convolution.SpectralConv])

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        target_state = self.model.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in checkpoint.items():
            if not isinstance(value, torch.Tensor):
                skipped.append((key, 'non-Tensor metadata'))
                continue
            if key not in target_state:
                skipped.append((key, 'missing in current model'))
                continue
            if value.shape != target_state[key].shape:
                skipped.append((key, f'shape mismatch {value.shape} vs {target_state[key].shape}'))
                continue
            filtered_state[key] = value

        if skipped:
            print('跳过以下不可加载的参数：')
            for name, reason in skipped:
                print(f'  {name}: {reason}')

        return self.model.load_state_dict(filtered_state, strict=False)

    def predict(self):
        pass

    def save_metrics(self) -> None:
        if not self.history['train_loss']:
            return

    def save_last_epoch_validation_visuals(self, max_samples: Union[int, None] = None) -> None:
        """Run the model on the validation set and save per-sample visualizations
        (time-series of target / predicted / old, and scatter plot target vs predicted)
        into `exp_dir/figures/val_last_epoch`.

        Args:
            max_samples: optional cap on number of samples to save (None => save all)
        """
        figures_dir = self.exp_dir / 'figures' / 'val_last_epoch'
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Collect arrays like test visualization expects
        preds_list, targets_list, w_old_list, fi_list, ns_list = [], [], [], [], []
        per_sample_rel_l2, per_sample_nrmse = [], []

        self.model.eval()
        sample_idx = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self.batch_to_device(batch)
                self._ensure_amp_compatibility(batch)
                with self._get_autocast_context():
                    w_pred = self.model(batch)
                batch_size = batch['tipn'].shape[0]

                scalers = batch.get('scalers')
                for i in range(batch_size):
                    n_i = int(batch['n'][i].item()) if isinstance(batch['n'], torch.Tensor) else int(batch['n'][i])
                    if n_i == 0:
                        continue
                    
                    scaler = scalers[i] if scalers else None

                    pred_norm = w_pred[i, :n_i].detach().cpu()
                    target_raw = batch['w_new'][i, :n_i].detach().cpu()
                    old_norm = batch['w_old'][i, :n_i].detach().cpu()
                    fi_arr = batch['fi'][i].detach().cpu().numpy() if 'fi' in batch else np.zeros(n_i)
                    
                    pred_raw = self._denormalize_tensor(pred_norm, scaler, 'pred')
                    old_raw = self._denormalize_tensor(old_norm, scaler, 'w_old')

                    # Store in formats expected by draw_all
                    preds_list.append(pred_raw.numpy())
                    targets_list.append(target_raw.numpy())
                    w_old_list.append(old_raw.numpy())
                    fi_list.append(fi_arr)
                    ns_list.append(n_i)

                    # compute per-sample metrics for plotting
                    rel_l2_val, nrmse_val = self._compute_relative_l2_and_nrmse(target_raw, old_raw + pred_raw)
                    per_sample_rel_l2.append(rel_l2_val)
                    per_sample_nrmse.append(nrmse_val)

                    sample_idx += 1
                    if max_samples is not None and sample_idx >= max_samples:
                        break
                if max_samples is not None and sample_idx >= max_samples:
                    break

        if sample_idx == 0:
            print('验证集为空，未生成可视化。')
            return

        # Use common visualization utilities to save figures (SVG) and metric curves
        try:
            draw_all(np.array(preds_list, dtype=object), np.array(targets_list, dtype=object), np.array(w_old_list, dtype=object), ns_list, np.array(fi_list, dtype=object), str(figures_dir))
        except Exception as e:
            # If draw_all can't handle object arrays, call with lists directly
            try:
                draw_all(preds_list, targets_list, w_old_list, ns_list, fi_list, str(figures_dir))
            except Exception as e2:
                print(f'调用 draw_all 保存可视化失败: {e2}')

        # save metric series in same folder
        try:
            plot_metric_series(per_sample_rel_l2, 'relative_l2_error', str(figures_dir))
            plot_metric_series(per_sample_nrmse, 'nrmse', str(figures_dir))
        except Exception as e:
            print(f'保存指标曲线失败: {e}')

        print(f'已保存最后一轮验证集可视化至: {figures_dir}，样本数: {sample_idx}')

        metrics_dir = self.exp_dir / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        epochs = list(range(1, len(self.history['train_loss']) + 1))

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(metrics_dir / 'loss_curve.png', dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history['train_r2'], label='Train R2')
        plt.plot(epochs, self.history['val_r2'], label='Validation R2')
        plt.xlabel('Epoch')
        plt.ylabel('R2 Score')
        plt.title('Training and Validation R2')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(metrics_dir / 'r2_curve.png', dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history['train_relative_l2'], label='Train Relative L2')
        plt.plot(epochs, self.history['val_relative_l2'], label='Validation Relative L2')
        plt.xlabel('Epoch')
        plt.ylabel('Relative L2 Error')
        plt.title('Training and Validation Relative L2 Error')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(metrics_dir / 'relative_l2_curve.png', dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history['train_nrmse'], label='Train NRMSE')
        plt.plot(epochs, self.history['val_nrmse'], label='Validation NRMSE')
        plt.xlabel('Epoch')
        plt.ylabel('NRMSE')
        plt.title('Training and Validation NRMSE')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(metrics_dir / 'nrmse_curve.png', dpi=300)
        plt.close()

        history_path = metrics_dir / 'history.json'
        history_data = {
            'epochs': epochs,
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'train_r2': self.history['train_r2'],
            'val_r2': self.history['val_r2'],
            'train_relative_l2': self.history['train_relative_l2'],
            'val_relative_l2': self.history['val_relative_l2'],
            'train_nrmse': self.history['train_nrmse'],
            'val_nrmse': self.history['val_nrmse'],
        }
        with open(history_path, 'w', encoding='utf-8') as history_file:
            json.dump(history_data, history_file, indent=2, ensure_ascii=False)
        print(f'训练指标已保存至: {metrics_dir}')

    @staticmethod
    def _compute_r2(ss_res: float, ss_tot: float, sample_count: int) -> float:
        if sample_count == 0:
            return float('nan')
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - (ss_res / ss_tot)

    @staticmethod
    def _compute_relative_l2_and_nrmse(targets: torch.Tensor, preds: torch.Tensor, eps: float = 1e-8) -> Tuple[float, float]:
        diff = preds - targets
        rel_l2_denom = torch.norm(targets) + eps
        rel_l2 = torch.norm(diff) / rel_l2_denom

        rmse = torch.sqrt(torch.mean(diff ** 2))
        range_val = torch.max(targets) - torch.min(targets)
        normalization = float(torch.abs(range_val).item())
        if normalization < eps:
            normalization = float(torch.norm(targets).item()) + eps
        nrmse = rmse / (normalization + eps)

        return float(rel_l2.item()), float(nrmse.item())

    @staticmethod
    def _denormalize_tensor(values: torch.Tensor, scaler: Union[dict, None], key: str) -> torch.Tensor:
        if not scaler:
            return values
        mean = scaler.get(f'{key}_mean')
        std = scaler.get(f'{key}_std')
        if mean is None or std is None:
            return values
        mean_t = torch.tensor(mean, dtype=values.dtype)
        std_t = torch.tensor(std, dtype=values.dtype)
        return values * std_t + mean_t

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    def _ensure_amp_compatibility(self, batch: Dict) -> None:
        if not self.use_amp:
            return
        # For 1D FNO, we check the last dimension (N)
        # batch['w_old'] has shape [B, N]
        if 'w_old' in batch:
            tensor = batch['w_old']
            size_last = int(tensor.shape[-1])
            if not self._is_power_of_two(size_last):
                 # Try to pad if possible, or disable AMP
                 # Here we just disable AMP for simplicity as padding logic is complex with variable N
                 reason = (
                    f"cuFFT requires power-of-two dimensions for fp16 ({size_last} received); "
                    "disabling AMP to avoid RuntimeError."
                )
                 self._disable_amp(reason)
                 return

        # Legacy check for 2D (if 'A' is used in a way that requires 2D FFT)
        # But now we use 1D FNO, so 'A' dimensions might not matter for FFT unless we do 2D FFT on it.
        # The new model uses 1D FNO on [B, C, N], so only N needs to be power of 2 for optimal performance/correctness in some implementations.
        
    def _disable_amp(self, reason: str) -> None:
        if self._amp_disabled_reason is not None:
            return
        self._amp_disabled_reason = reason
        if self.use_amp:
            print(f"AMP disabled: {reason}")
        self.use_amp = False
        if AMP_MODE == 'torch_amp':
            self.scaler = GradScaler(self.amp_device_type if self.amp_device_type else 'cuda', enabled=False)
        else:
            self.scaler = GradScaler(enabled=False)


FluidSolidTrainer = _FluidSolidTrainer