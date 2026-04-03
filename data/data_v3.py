from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from common import rebuild_ddm

try:
    import yaml
except Exception:
    yaml = None


class SamplePathDataset(Dataset):
    """Dataset that loads samples from sample directories on demand or preloads in memory."""

    def __init__(
        self,
        sample_paths: List[Union[str, Path]],
        *,
        max_n: int = 3000,
        preload: bool = False,
        load_cb: bool = False,
        load_ddm_info: bool = False,
        skip_invalid_samples: bool = False,
    ) -> None:
        self.sample_paths = [Path(p) for p in sample_paths]
        self.max_n = max_n
        self.preload = preload
        self.load_cb = load_cb
        self.load_ddm_info = load_ddm_info
        self.skip_invalid_samples = skip_invalid_samples
        self._samples: Union[List[dict], None] = None
        self.invalid_samples: List[Dict[str, str]] = []

        self.sample_paths = self._filter_invalid_samples(self.sample_paths)

        if self.preload:
            import os
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            samples: List[dict] = []
            kept_paths: List[Path] = []
            
            max_workers = min(32, (os.cpu_count() or 1) * 2)
            # 使用多线程加速预加载，维持原本的顺序
            results = [None] * len(self.sample_paths)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(self._load_sample, p): i for i, p in enumerate(self.sample_paths)}
                
                for future in tqdm(as_completed(future_to_idx), total=len(self.sample_paths), desc="Preloading samples"):
                    idx = future_to_idx[future]
                    p = self.sample_paths[idx]
                    try:
                        results[idx] = (future.result(), p)
                    except Exception as e:
                        if not self.skip_invalid_samples:
                            raise
                        print(f"[数据异常] 预加载跳过样本: {p} | 原因: {e}")
                        self.invalid_samples.append({"sample_dir": str(p), "reason": str(e)})

            # 保持原始路径顺序
            for item in results:
                if item is not None:
                    sample_dict, p = item
                    samples.append(sample_dict)
                    kept_paths.append(p)
                    
            self._samples = samples
            self.sample_paths = kept_paths

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> dict:
        if self._samples is not None:
            return self._samples[idx]
        if len(self.sample_paths) == 0:
            raise IndexError("数据集为空，无法获取样本。")
        if not self.skip_invalid_samples:
            return self._load_sample(self.sample_paths[idx])

        total = len(self.sample_paths)
        if total == 0:
            raise IndexError("数据集为空")
        for offset in range(total):
            next_idx = (idx + offset) % total
            sample_path = self.sample_paths[next_idx]
            try:
                return self._load_sample(sample_path)
            except Exception as e:
                print(f"[数据异常] 跳过样本: {sample_path} | 原因: {e}")
                self.invalid_samples.append({"sample_dir": str(sample_path), "reason": str(e)})
                continue
        raise ValueError("未找到可用样本，请检查数据集质量。")

    @staticmethod
    def _build_row_index(meta: Dict) -> Dict[str, int]:
        rows = meta.get("rows", {})
        name_to_idx = {}
        for idx_str, row in rows.items():
            name = row.get("name") if isinstance(row, dict) else None
            if name is not None:
                name_to_idx[name] = int(idx_str)
        return name_to_idx

    def _check_sample_validity(self, sample_dir: Path) -> Tuple[bool, str]:
        sample_path = sample_dir / "sample.npy"
        meta_path = sample_dir / "sample_metadata.json"

        if not sample_path.exists():
            return False, f"缺少 sample.npy: {sample_path}"
        if not meta_path.exists():
            return False, f"缺少 sample_metadata.json: {meta_path}"

        try:
            sample_arr = np.load(sample_path)
        except Exception as e:
            return False, f"读取 sample.npy 失败: {e}"

        if sample_arr.ndim != 2:
            return False, f"sample.npy 维度异常，期望2维，实际 {sample_arr.ndim} 维"

        if sample_arr.shape[0] < 8:
            return False, f"sample.npy 行数不足，期望>=8，实际 {sample_arr.shape[0]}"

        if not np.all(np.isfinite(sample_arr)):
            invalid_count = int(np.size(sample_arr) - np.count_nonzero(np.isfinite(sample_arr)))
            return False, f"sample.npy 存在 NaN/Inf，异常值数量={invalid_count}"

        max_abs = float(np.nanmax(np.abs(sample_arr)))
        if not np.isfinite(max_abs):
            return False, "sample.npy 最大值计算异常 (NaN/Inf)"
        if max_abs > np.finfo(np.float32).max:
            return False, f"sample.npy 存在超过 float32 上限的数值: max_abs={max_abs:.3e}"

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            return False, f"读取 sample_metadata.json 失败: {e}"

        n = int(meta.get("n_size") or sample_arr.shape[1])
        n = min(n, sample_arr.shape[1])  # max_n 不再截断，模型支持任意长度
        if n <= 0:
            return False, f"n_size 非法，n={n}"

        rows_meta = meta.get("rows", {})
        if isinstance(rows_meta, dict):
            for row_meta in rows_meta.values():
                if not isinstance(row_meta, dict):
                    continue
                mean = row_meta.get("mean")
                std = row_meta.get("std")
                if mean is None or std is None:
                    continue
                if not np.isfinite(mean) or not np.isfinite(std):
                    return False, "sample_metadata.json 含 NaN/Inf 的 mean/std"
                if std <= 0:
                    return False, f"sample_metadata.json std 非正数: std={std}"

        return True, ""

    def _filter_invalid_samples(self, sample_paths: List[Path]) -> List[Path]:
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        valid_paths: List[Path] = []
        skipped = 0
        
        # 使用多线程加速 (Numpy和文件读取会释放GIL，且在Windows下比多进程更稳定不会报错)
        max_workers = min(32, (os.cpu_count() or 1) * 2)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(self._check_sample_validity, p): p for p in sample_paths}
            
            for future in tqdm(as_completed(future_to_path), total=len(sample_paths), desc="Validating samples"):
                sample_dir = future_to_path[future]
                try:
                    is_valid, message = future.result()
                    if is_valid:
                        valid_paths.append(sample_dir)
                    else:
                        skipped += 1
                        print(f"[数据异常] 跳过样本: {sample_dir} | 原因: {message}")
                        self.invalid_samples.append({"sample_dir": str(sample_dir), "reason": message})
                except Exception as e:
                    skipped += 1
                    print(f"[数据异常] 并行处理错误 跳过样本: {sample_dir} | 原因: {e}")
                    self.invalid_samples.append({"sample_dir": str(sample_dir), "reason": str(e)})

        if skipped > 0:
            print(f"[数据过滤] 共跳过异常样本 {skipped} 个，保留 {len(valid_paths)} 个。")
        return valid_paths

    def _load_sample(self, sample_dir: Path) -> dict:
        sample_path = sample_dir / "sample.npy"
        meta_path = sample_dir / "sample_metadata.json"

        if not sample_path.exists():
            raise FileNotFoundError(f"sample.npy not found: {sample_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"sample_metadata.json not found: {meta_path}")

        sample_arr = np.load(sample_path)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        name_to_idx = self._build_row_index(meta)
        fallback_order = ["w_old", "b", "Ax", "diag_A", "r", "fi", "w_new", "cb"]
        row_idx = {name: name_to_idx.get(name, idx) for idx, name in enumerate(fallback_order)}

        # 动态捕捉其余自定义的元数据索引
        for name, idx in name_to_idx.items():
            if name not in row_idx:
                row_idx[name] = idx

        n = int(meta.get("n_size") or sample_arr.shape[1])
        n = min(n, sample_arr.shape[1])  # max_n 不再截断，模型支持任意长度
        if n <= 0:
            raise ValueError(f"Invalid n_size for sample: {sample_dir}")

        def row(name: str) -> np.ndarray:
            return sample_arr[row_idx[name], :n]

        def row_optional(name: str):
            idx = row_idx.get(name)
            if idx is not None and idx < sample_arr.shape[0]:
                return sample_arr[idx, :n]
            return None

        def _safe_cast_float32(values: np.ndarray, name: str) -> np.ndarray:
            if values.size == 0:
                raise ValueError(f"样本 {sample_dir} 行 {name} 为空")
            if not np.all(np.isfinite(values)):
                invalid = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
                raise ValueError(f"样本 {sample_dir} 行 {name} 含 NaN/Inf, count={invalid}")
            max_abs = float(np.nanmax(np.abs(values)))
            if max_abs > np.finfo(np.float32).max:
                raise ValueError(
                    f"样本 {sample_dir} 行 {name} 超过 float32 上限: max_abs={max_abs:.3e}"
                )
            return values.astype(np.float32)

        w_old = _safe_cast_float32(row("w_old"), "w_old")
        b = _safe_cast_float32(row("b"), "b")
        diag_A = _safe_cast_float32(row("diag_A"), "diag_A")
        r = _safe_cast_float32(row("r"), "r")
        fi = _safe_cast_float32(row("fi"), "fi")
        w_new = _safe_cast_float32(row("w_new"), "w_new")
        cb = _safe_cast_float32(row("cb"), "cb")

        # Derive scalers from metadata rows if present
        scalers = {}
        rows_meta = meta.get("rows", {})
        if isinstance(rows_meta, dict):
            for idx_str, row_meta in rows_meta.items():
                if not isinstance(row_meta, dict):
                    continue
                name = row_meta.get("name")
                if not name:
                    continue
                mean = row_meta.get("mean")
                std = row_meta.get("std")
                if mean is None or std is None:
                    continue
                scalers[f"{name}_mean"] = float(mean)
                scalers[f"{name}_std"] = float(std) + 1e-8

        # Compute pred normalization (per-sample) for training/test consistency
        w_old_mean = scalers.get("w_old_mean", 0.0)
        w_old_std = scalers.get("w_old_std", 1.0)
        if not np.isfinite(w_old_mean) or not np.isfinite(w_old_std) or w_old_std <= 0:
            raise ValueError(
                f"样本 {sample_dir} w_old 统计量异常: mean={w_old_mean}, std={w_old_std}"
            )
        w_old_raw = w_old * w_old_std + w_old_mean
        if not np.all(np.isfinite(w_old_raw)):
            raise ValueError(f"样本 {sample_dir} 反归一化 w_old 后出现 NaN/Inf")
        pred_raw = w_new - w_old_raw
        if not np.all(np.isfinite(pred_raw)):
            raise ValueError(f"样本 {sample_dir} pred_raw 含 NaN/Inf")
        pred_mean = float(np.mean(pred_raw))
        pred_std = float(np.std(pred_raw)) + 1e-8
        if not np.isfinite(pred_mean) or not np.isfinite(pred_std) or pred_std <= 0:
            raise ValueError(
                f"样本 {sample_dir} pred 统计量异常: mean={pred_mean}, std={pred_std}"
            )
        pred_norm = (pred_raw - pred_mean) / pred_std
        if not np.all(np.isfinite(pred_norm)):
            raise ValueError(f"样本 {sample_dir} pred_norm 含 NaN/Inf")
        scalers["pred_mean"] = pred_mean
        scalers["pred_std"] = pred_std

        time_stats = meta.get("time_stats", {})
        solver_time = float(time_stats.get("total_loop_ms", meta.get("coupled_time_ms", 0.0))) / 1000.0
        pure_solve_time = float(time_stats.get("matrix_solve_ms", 0.0)) / 1000.0
        prep_time = float(time_stats.get("avg_matrix_creation_ms", 0.0)) / 1000.0

        ddm_info = None
        if self.load_ddm_info:
            ddm_meta = meta.get("ddm", {})
            if ddm_meta:
                ddm_info = {
                    "ddm_idx": str(sample_dir / ddm_meta.get("ddm_idx_file", "")),
                    "tep_C": str(sample_dir / ddm_meta.get("tep_C_file", "")),
                    "triangular_value": str(sample_dir / ddm_meta.get("global", {}).get("triangular_value", "")),
                    "grids": str(sample_dir / ddm_meta.get("global", {}).get("grids", "")),
                    "ranges": ddm_meta.get("ranges", {}),
                }

        # Extract physics parameters
        params = meta.get("parameters", {})
        E = float(params.get("E", 27000.0))
        PR = float(params.get("PR", 0.22))
        Kic = float(params.get("Kic", 1.6))
        Q = float(params.get("Q", 0.1))
        fk = float(params.get("fk", 1e-8))
        Dclu = float(params.get("Dclu", 15.0))

        phys_params = np.array([
            (E - 25000.0) / 10000.0,
            PR,
            Kic,
            Q * 10.0,
            np.log10(max(fk, 1e-12)) + 8.0
        ], dtype=np.float32)

        # 读取新引入的地层分布 hf 和隔层应力差 df 
        # 当 numpy array 中缺省空间场矩阵时，根据 par.hf 取默认值扩散到所有网格点上
        hf_p = params.get("hf", [25.0, 25.0])
        df_p = params.get("df", [2.0, 2.0])
        if isinstance(hf_p, (int, float)): hf_p = [float(hf_p), float(hf_p)]
        if isinstance(df_p, (int, float)): df_p = [float(df_p), float(df_p)]

        sp_hf_0 = row_optional("spatial_hf_0")
        if sp_hf_0 is None: sp_hf_0 = np.full(n, hf_p[0], dtype=np.float32)
        else: sp_hf_0 = _safe_cast_float32(sp_hf_0, "spatial_hf_0")

        sp_hf_1 = row_optional("spatial_hf_1")
        if sp_hf_1 is None: sp_hf_1 = np.full(n, hf_p[1], dtype=np.float32)
        else: sp_hf_1 = _safe_cast_float32(sp_hf_1, "spatial_hf_1")

        sp_df_0 = row_optional("spatial_df_0")
        if sp_df_0 is None: sp_df_0 = np.full(n, df_p[0], dtype=np.float32)
        else: sp_df_0 = _safe_cast_float32(sp_df_0, "spatial_df_0")

        sp_df_1 = row_optional("spatial_df_1")
        if sp_df_1 is None: sp_df_1 = np.full(n, df_p[1], dtype=np.float32)
        else: sp_df_1 = _safe_cast_float32(sp_df_1, "spatial_df_1")

        fi_diff = np.abs(np.diff(fi))
        is_boundary = np.zeros_like(fi, dtype=bool)
        is_boundary[1:] = fi_diff > 1e-2
        cluster_idx = np.cumsum(is_boundary)
        cluster_dist = (cluster_idx * Dclu).astype(np.float32)

        sample = {
            "b": b,
            "w_old": w_old,
            "w_new": w_new,
            "pred": pred_norm.astype(np.float32),
            "fi": fi,
            "phys_params": phys_params,
            "cluster_dist": cluster_dist,
            "sp_hf_0": sp_hf_0,
            "sp_hf_1": sp_hf_1,
            "sp_df_0": sp_df_0,
            "sp_df_1": sp_df_1,
            "r": r,
            "diag_A": diag_A,
            "n": n,
            "tipn": n,
            "scalers": scalers,
            "condition": sample_dir.parent.parent.name if sample_dir.parent.name == "samples" else sample_dir.parent.name,
            "sample_dir": str(sample_dir),
            "solver_time": solver_time,
            "pure_solve_time": pure_solve_time,
            "prep_time": prep_time,
        }

        if self.load_cb:
            sample["cb"] = cb

        if ddm_info is not None:
            sample["ddm_info"] = ddm_info

        return sample


def _condition_from_sample_path(sample_path: Union[str, Path]) -> str:
    path = Path(sample_path)
    parts = path.parts
    if "samples" in parts:
        idx = parts.index("samples")
        if idx > 0:
            return parts[idx - 1]
    return path.parent.name


def filter_sample_paths(
    sample_paths: List[Union[str, Path]],
    *,
    max_samples_per_condition: Union[int, None] = None,
    max_conditions: Union[int, None] = None,
) -> List[str]:
    if not sample_paths:
        return []

    if max_samples_per_condition in (None, 0) and max_conditions in (None, 0):
        return [str(p) for p in sample_paths]

    grouped: Dict[str, List[Path]] = {}
    for p in sample_paths:
        cond = _condition_from_sample_path(p)
        grouped.setdefault(cond, []).append(Path(p))

    condition_names = sorted(grouped.keys())
    if max_conditions not in (None, 0):
        condition_names = condition_names[: int(max_conditions)]

    filtered: List[str] = []
    for cond in condition_names:
        paths = sorted(grouped[cond])
        if max_samples_per_condition not in (None, 0):
            paths = paths[: int(max_samples_per_condition)]
        filtered.extend([str(p) for p in paths])
    return filtered


class FSCdatasetV3:
    def __init__(
        self,
        max_n: int = 3000,
        file_path: str = "",
        save_path: str = "",
        rates: List[float] = None,
        seed: int = 42,
        max_conditions: int = None,
        max_samples_per_condition: Union[int, None] = None,
        skip_invalid_samples: bool = False,
        export_stats: bool = False,
        stats_format: str = "yaml",
        stats_filename: str = "dataset_stats.yaml",
    ) -> None:
        self.max_n = max_n
        self.file_path = file_path
        self.save_path = save_path
        self.rates = rates or [0.7, 0.2, 0.1]
        self.seed = seed
        self.max_conditions = max_conditions
        self.max_samples_per_condition = max_samples_per_condition
        self.skip_invalid_samples = skip_invalid_samples
        self.export_stats = export_stats
        self.stats_format = str(stats_format).lower()
        self.stats_filename = stats_filename

    @staticmethod
    def _stats_init() -> Dict[str, Union[int, float, None]]:
        return {
            "count": 0,
            "nan_count": 0,
            "inf_count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": None,
            "max": None,
        }

    @staticmethod
    def _stats_update(bucket: Dict[str, Union[int, float, None]], values: np.ndarray) -> None:
        vals = np.asarray(values)
        if vals.size == 0:
            return
        nan_count = int(np.count_nonzero(np.isnan(vals)))
        inf_count = int(np.count_nonzero(np.isinf(vals)))
        bucket["nan_count"] += nan_count
        bucket["inf_count"] += inf_count
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return
        bucket["count"] += int(finite.size)
        s = float(np.sum(finite))
        sq = float(np.sum(finite.astype(np.float64) ** 2))
        bucket["sum"] += s
        bucket["sum_sq"] += sq
        min_v = float(np.min(finite))
        max_v = float(np.max(finite))
        bucket["min"] = min_v if bucket["min"] is None else min(bucket["min"], min_v)
        bucket["max"] = max_v if bucket["max"] is None else max(bucket["max"], max_v)

    @staticmethod
    def _stats_finalize(bucket: Dict[str, Union[int, float, None]]) -> Dict[str, Union[int, float, None]]:
        count = int(bucket["count"])
        out = {
            "count": count,
            "nan_count": int(bucket["nan_count"]),
            "inf_count": int(bucket["inf_count"]),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
        if count == 0:
            return out
        mean = float(bucket["sum"]) / count
        var = max(0.0, float(bucket["sum_sq"]) / count - mean ** 2)
        out["min"] = float(bucket["min"])
        out["max"] = float(bucket["max"])
        out["mean"] = mean
        out["std"] = float(np.sqrt(var))
        return out

    @staticmethod
    def _build_row_index(meta: Dict) -> Dict[str, int]:
        rows = meta.get("rows", {})
        name_to_idx = {}
        for idx_str, row in rows.items():
            name = row.get("name") if isinstance(row, dict) else None
            if name is not None:
                name_to_idx[name] = int(idx_str)
        return name_to_idx

    def _read_sample_arrays(self, sample_dir: Path) -> Dict[str, np.ndarray]:
        sample_path = sample_dir / "sample.npy"
        meta_path = sample_dir / "sample_metadata.json"
        sample_arr = np.load(sample_path)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        name_to_idx = self._build_row_index(meta)
        fallback_order = ["w_old", "b", "Ax", "diag_A", "r", "fi", "w_new", "cb"]
        row_idx = {name: name_to_idx.get(name, idx) for idx, name in enumerate(fallback_order)}
        n = int(meta.get("n_size") or sample_arr.shape[1])
        n = min(n, sample_arr.shape[1])  # max_n 不再截断，模型支持任意长度

        def row(name: str) -> np.ndarray:
            return sample_arr[row_idx[name], :n]

        w_old = row("w_old")
        w_new = row("w_new")
        pred_raw = w_new - w_old
        time_stats = meta.get("time_stats", {})
        return {
            "condition": sample_dir.parent.parent.name if sample_dir.parent.name == "samples" else sample_dir.parent.name,
            "n": np.array([n], dtype=np.float64),
            "w_old": w_old,
            "b": row("b"),
            "r": row("r"),
            "diag_A": row("diag_A"),
            "fi": row("fi"),
            "w_new": w_new,
            "pred_raw": pred_raw,
            "solver_time": np.array([float(time_stats.get("total_loop_ms", meta.get("coupled_time_ms", 0.0))) / 1000.0], dtype=np.float64),
            "pure_solve_time": np.array([float(time_stats.get("matrix_solve_ms", 0.0)) / 1000.0], dtype=np.float64),
            "prep_time": np.array([float(time_stats.get("avg_matrix_creation_ms", 0.0)) / 1000.0], dtype=np.float64),
        }

    def _export_dataset_stats(
        self,
        train_dataset: SamplePathDataset,
        val_dataset: SamplePathDataset,
        test_dataset: SamplePathDataset,
    ) -> None:
        if not self.export_stats:
            return
        if not self.save_path:
            return

        var_keys = ["w_old", "b", "r", "diag_A", "fi", "w_new", "pred_raw", "n", "solver_time", "pure_solve_time", "prep_time"]
        global_stats = {k: self._stats_init() for k in var_keys}
        by_condition: Dict[str, Dict[str, Dict[str, Union[int, float, None]]]] = {}
        split_summary = {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        }

        invalid_samples = []
        invalid_samples.extend(getattr(train_dataset, "invalid_samples", []))
        invalid_samples.extend(getattr(val_dataset, "invalid_samples", []))
        invalid_samples.extend(getattr(test_dataset, "invalid_samples", []))

        all_paths = []
        all_paths.extend([Path(p) for p in train_dataset.sample_paths])
        all_paths.extend([Path(p) for p in val_dataset.sample_paths])
        all_paths.extend([Path(p) for p in test_dataset.sample_paths])

        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = min(32, (os.cpu_count() or 1) * 2)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_dir = {executor.submit(self._read_sample_arrays, sample_dir): sample_dir for sample_dir in all_paths}
            
            for future in tqdm(as_completed(future_to_dir), total=len(all_paths), desc="Collecting dataset stats"):
                sample_dir = future_to_dir[future]
                try:
                    rec = future.result()
                except Exception as e:
                    invalid_samples.append({"sample_dir": str(sample_dir), "reason": f"统计阶段读取失败: {e}"})
                    print(f"[统计异常] 跳过样本: {sample_dir} | 原因: {e}")
                    continue

                cond = rec["condition"]
                if cond not in by_condition:
                    by_condition[cond] = {k: self._stats_init() for k in var_keys}

                for key in var_keys:
                    self._stats_update(global_stats[key], rec[key])
                    self._stats_update(by_condition[cond][key], rec[key])

        condition_stats = {}
        for cond, cond_buckets in by_condition.items():
            condition_stats[cond] = {
                key: self._stats_finalize(bucket)
                for key, bucket in cond_buckets.items()
            }
            condition_stats[cond]["sample_count"] = condition_stats[cond]["n"]["count"]

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_version": "v3",
            "origin_path": self.file_path,
            "save_path": self.save_path,
            "max_n": self.max_n,
            "skip_invalid_samples": self.skip_invalid_samples,
            "split_samples": split_summary,
            "total_conditions": len(condition_stats),
            "total_samples": int(self._stats_finalize(global_stats["n"])["count"]),
            "invalid_samples_skipped": len(invalid_samples),
            "global_stats": {k: self._stats_finalize(v) for k, v in global_stats.items()},
            "condition_stats": condition_stats,
            "invalid_samples": invalid_samples,
        }

        os.makedirs(self.save_path, exist_ok=True)
        target_name = self.stats_filename
        if self.stats_format == "json" and not target_name.lower().endswith(".json"):
            target_name = f"{Path(target_name).stem}.json"
        if self.stats_format == "yaml" and not target_name.lower().endswith((".yaml", ".yml")):
            target_name = f"{Path(target_name).stem}.yaml"

        output_path = Path(self.save_path) / target_name
        if self.stats_format == "json" or yaml is None:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            with output_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        print(f"[统计导出] 数据统计信息已保存: {output_path}")

        gs = payload["global_stats"]
        print("[统计导出] 全局指标摘要:")
        print(
            f"  - 条件数={payload['total_conditions']}, 样本数={payload['total_samples']}, "
            f"异常样本={payload['invalid_samples_skipped']}"
        )
        print(
            f"  - n(mean±std)={gs['n']['mean']:.2f}±{gs['n']['std']:.2f}, "
            f"n[min,max]=[{gs['n']['min']:.0f}, {gs['n']['max']:.0f}]"
        )
        print(
            f"  - pred_raw(mean±std)={gs['pred_raw']['mean']:.4e}±{gs['pred_raw']['std']:.4e}"
        )
        print(
            f"  - w_new(mean±std)={gs['w_new']['mean']:.4e}±{gs['w_new']['std']:.4e}"
        )
        print(
            f"  - solver_time(mean±std)={gs['solver_time']['mean']:.6f}s±{gs['solver_time']['std']:.6f}s"
        )

    def _list_condition_dirs(self) -> List[Path]:
        root = Path(self.file_path)
        if not root.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.file_path}")
        return [p for p in root.iterdir() if p.is_dir()]

    @staticmethod
    def _list_samples(condition_dir: Path) -> List[Path]:
        samples_dir = condition_dir / "samples"
        if not samples_dir.exists():
            return []
        return sorted([p for p in samples_dir.iterdir() if p.is_dir() and (p / "sample.npy").exists()])

    def _scan_samples(self) -> Dict[str, List[Path]]:
        condition_dirs = self._list_condition_dirs()
        if self.max_conditions:
            condition_dirs = condition_dirs[: int(self.max_conditions)]
        samples_map: Dict[str, List[Path]] = {}
        for cond_dir in condition_dirs:
            sample_dirs = self._list_samples(cond_dir)
            if self.max_samples_per_condition:
                sample_dirs = sample_dirs[: int(self.max_samples_per_condition)]
            if sample_dirs:
                samples_map[cond_dir.name] = sample_dirs
        if not samples_map:
            raise ValueError("未找到任何样本目录（samples/sample_t*_k*）。")
        return samples_map

    def init_local_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        rng = np.random.RandomState(self.seed)
        samples_map = self._scan_samples()
        condition_names = list(samples_map.keys())
        rng.shuffle(condition_names)

        n_conditions = len(condition_names)
        n_train = int(n_conditions * self.rates[0])
        n_val = int(n_conditions * self.rates[1])
        train_conds = condition_names[:n_train]
        val_conds = condition_names[n_train:n_train + n_val]
        test_conds = condition_names[n_train + n_val:]

        def collect(conds: List[str]) -> List[str]:
            paths: List[str] = []
            for cond in conds:
                paths.extend([str(p) for p in samples_map[cond]])
            return paths

        train_paths = collect(train_conds)
        val_paths = collect(val_conds)
        test_paths = collect(test_conds)

        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            with open(Path(self.save_path) / "train_paths.json", "w", encoding="utf-8") as f:
                json.dump(train_paths, f, ensure_ascii=False)
            with open(Path(self.save_path) / "val_paths.json", "w", encoding="utf-8") as f:
                json.dump(val_paths, f, ensure_ascii=False)
            with open(Path(self.save_path) / "test_paths.json", "w", encoding="utf-8") as f:
                json.dump(test_paths, f, ensure_ascii=False)

        datasets = (
            SamplePathDataset(
                train_paths,
                max_n=self.max_n,
                preload=True,
                load_cb=False,
                load_ddm_info=False,
                skip_invalid_samples=self.skip_invalid_samples,
            ),
            SamplePathDataset(
                val_paths,
                max_n=self.max_n,
                preload=True,
                load_cb=False,
                load_ddm_info=False,
                skip_invalid_samples=self.skip_invalid_samples,
            ),
            SamplePathDataset(
                test_paths,
                max_n=self.max_n,
                preload=False,
                load_cb=True,
                load_ddm_info=True,
                skip_invalid_samples=self.skip_invalid_samples,
            ),
        )
        self._export_dataset_stats(*datasets)
        return datasets

    def load_data(self, file_path: str, test_pkl_path: str = None):
        train_paths_file = os.path.join(file_path, "train_paths.json")
        val_paths_file = os.path.join(file_path, "val_paths.json")
        test_paths_file = os.path.join(file_path, "test_paths.json")

        if os.path.exists(train_paths_file) and os.path.exists(val_paths_file) and os.path.exists(test_paths_file):
            print("检测到本地数据集索引 (JSON)，加载中...")
            with open(train_paths_file, "r", encoding="utf-8") as f:
                train_paths = json.load(f)
            with open(val_paths_file, "r", encoding="utf-8") as f:
                val_paths = json.load(f)
            with open(test_paths_file, "r", encoding="utf-8") as f:
                test_paths = json.load(f)
        else:
            print("本地数据集不存在，初始化 v3 数据集...")
            return self.init_local_data()

        train_paths = filter_sample_paths(
            train_paths,
            max_samples_per_condition=self.max_samples_per_condition,
            max_conditions=self.max_conditions,
        )
        val_paths = filter_sample_paths(
            val_paths,
            max_samples_per_condition=self.max_samples_per_condition,
            max_conditions=self.max_conditions,
        )
        test_paths = filter_sample_paths(
            test_paths,
            max_samples_per_condition=self.max_samples_per_condition,
            max_conditions=self.max_conditions,
        )

        if test_pkl_path:
            test_pkl_path = os.path.expanduser(test_pkl_path)
            if os.path.isfile(test_pkl_path):
                raise FileNotFoundError("V3 数据集不支持 pkl 测试集，请提供 JSON 路径索引。")
            save_dir = os.path.dirname(test_pkl_path)
            json_path = os.path.join(save_dir, "test_paths.json")
            if os.path.exists(json_path):
                print(f"指定的 pkl 不存在，但检测到 JSON 索引文件: {json_path}")
                with open(json_path, "r", encoding="utf-8") as f:
                    test_paths = json.load(f)
            else:
                raise FileNotFoundError(f"指定的测试集文件不存在: {test_pkl_path}，且未找到同目录下的 test_paths.json")

        datasets = (
            SamplePathDataset(
                train_paths,
                max_n=self.max_n,
                preload=True,
                load_cb=False,
                load_ddm_info=False,
                skip_invalid_samples=self.skip_invalid_samples,
            ),
            SamplePathDataset(
                val_paths,
                max_n=self.max_n,
                preload=True,
                load_cb=False,
                load_ddm_info=False,
                skip_invalid_samples=self.skip_invalid_samples,
            ),
            SamplePathDataset(
                test_paths,
                max_n=self.max_n,
                preload=False,
                load_cb=True,
                load_ddm_info=True,
                skip_invalid_samples=self.skip_invalid_samples,
            ),
        )
        self._export_dataset_stats(*datasets)
        return datasets

    def collate_fn(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}

        batch_size = len(batch)
        max_n = max(sample["w_old"].shape[0] for sample in batch)
        dtype = torch.float32

        b = torch.zeros((batch_size, max_n), dtype=dtype)
        w_old = torch.zeros((batch_size, max_n), dtype=dtype)
        w_new = torch.zeros((batch_size, max_n), dtype=dtype)
        pred = torch.zeros((batch_size, max_n), dtype=dtype)
        fi = torch.zeros((batch_size, max_n), dtype=dtype)
        r = torch.zeros((batch_size, max_n), dtype=dtype)
        diag_A = torch.zeros((batch_size, max_n), dtype=dtype)
        n = torch.zeros(batch_size, dtype=torch.long)
        tipn = torch.zeros(batch_size, dtype=torch.long)
        solver_time = torch.zeros(batch_size, dtype=torch.float32)
        pure_solve_time = torch.zeros(batch_size, dtype=torch.float32)
        prep_time = torch.zeros(batch_size, dtype=torch.float32)
        
        # New physics/topology fields
        has_phys = "phys_params" in batch[0]
        phys_params = torch.zeros((batch_size, 5), dtype=dtype) if has_phys else None
        cluster_dist = torch.zeros((batch_size, max_n), dtype=dtype) if has_phys else None
        sp_hf_0 = torch.zeros((batch_size, max_n), dtype=dtype) if has_phys else None
        sp_hf_1 = torch.zeros((batch_size, max_n), dtype=dtype) if has_phys else None
        sp_df_0 = torch.zeros((batch_size, max_n), dtype=dtype) if has_phys else None
        sp_df_1 = torch.zeros((batch_size, max_n), dtype=dtype) if has_phys else None

        cb = None
        if any("cb" in sample for sample in batch):
            cb = torch.zeros((batch_size, max_n), dtype=dtype)

        for idx, sample in enumerate(batch):
            n_i = sample["w_old"].shape[0]
            b[idx, :n_i] = torch.from_numpy(sample["b"])
            w_old[idx, :n_i] = torch.from_numpy(sample["w_old"])
            w_new[idx, :n_i] = torch.from_numpy(sample["w_new"])
            if "pred" in sample:
                pred[idx, :n_i] = torch.from_numpy(sample["pred"])
            fi[idx, :n_i] = torch.from_numpy(sample["fi"])
            if "r" in sample:
                r[idx, :n_i] = torch.from_numpy(sample["r"])
            if "diag_A" in sample:
                diag_A[idx, :n_i] = torch.from_numpy(sample["diag_A"])
            if cb is not None and "cb" in sample:
                cb[idx, :n_i] = torch.from_numpy(sample["cb"])
                
            if has_phys and "phys_params" in sample:
                phys_params[idx] = torch.from_numpy(sample["phys_params"])
                cluster_dist[idx, :n_i] = torch.from_numpy(sample["cluster_dist"])
                sp_hf_0[idx, :n_i] = torch.from_numpy(sample["sp_hf_0"])
                sp_hf_1[idx, :n_i] = torch.from_numpy(sample["sp_hf_1"])
                sp_df_0[idx, :n_i] = torch.from_numpy(sample["sp_df_0"])
                sp_df_1[idx, :n_i] = torch.from_numpy(sample["sp_df_1"])

            n[idx] = sample.get("n", n_i)
            tipn[idx] = sample.get("tipn", n_i)
            solver_time[idx] = sample.get("solver_time", 0.0)
            pure_solve_time[idx] = sample.get("pure_solve_time", 0.0)
            prep_time[idx] = sample.get("prep_time", 0.0)

        conditions = [sample.get("condition", "unknown") for sample in batch]
        scalers = [sample.get("scalers") for sample in batch]
        ddm_infos = [sample.get("ddm_info") for sample in batch]
        sample_dirs = [sample.get("sample_dir") for sample in batch]

        batch_dict = {
            "b": b,
            "w_old": w_old,
            "w_new": w_new,
            "pred": pred,
            "fi": fi,
            "r": r,
            "diag_A": diag_A,
            "n": n,
            "tipn": tipn,
            "max_n": torch.full((batch_size,), max_n, dtype=torch.long),
            "scalers": scalers,
            "condition": conditions,
            "sample_dir": sample_dirs,
            "solver_time": solver_time,
            "pure_solve_time": pure_solve_time,
            "prep_time": prep_time,
        }
        if cb is not None:
            batch_dict["cb"] = cb
        if any(info is not None for info in ddm_infos):
            batch_dict["ddm_info"] = ddm_infos
        if has_phys:
            batch_dict["phys_params"] = phys_params
            batch_dict["cluster_dist"] = cluster_dist
            batch_dict["sp_hf_0"] = sp_hf_0
            batch_dict["sp_hf_1"] = sp_hf_1
            batch_dict["sp_df_0"] = sp_df_0
            batch_dict["sp_df_1"] = sp_df_1

        return batch_dict


def build_ddm_coef_matrix(sample_dir: Union[str, Path], part: str = "all") -> np.ndarray:
    """Build DDM coefficient matrix C from sample directory.

    Args:
        sample_dir: path to sample directory (contains sample_metadata.json).
        part: one of "all", "ntip", "ttip", "Ccc", "Cct".
    """
    sample_dir = Path(sample_dir)
    meta = rebuild_ddm.load_metadata(sample_dir)
    ddm = meta.get("ddm", {})
    if not ddm:
        raise ValueError("sample_metadata.json does not contain ddm info")

    tri = np.load(sample_dir / ddm["global"]["triangular_value"]).astype(np.float64)
    ddm_idx = np.load(sample_dir / ddm["ddm_idx_file"]).astype(np.int64)
    ranges = ddm["ranges"]

    ntip_idx = ddm_idx[ranges["ntip"]["start"]: ranges["ntip"]["start"] + ranges["ntip"]["count"]]
    ttip_idx = ddm_idx[ranges["ttip"]["start"]: ranges["ttip"]["start"] + ranges["ttip"]["count"]]
    all_idx = ddm_idx[ranges["all"]["start"]: ranges["all"]["start"] + ranges["all"]["count"]]

    if part == "ntip":
        return rebuild_ddm.create_coef_matrix(ntip_idx, ntip_idx, tri)
    if part == "ttip":
        return rebuild_ddm.create_coef_matrix(ttip_idx, ttip_idx, tri)
    if part == "Ccc":
        return rebuild_ddm.create_coef_matrix(ntip_idx, ntip_idx, tri)
    if part == "Cct":
        return rebuild_ddm.create_coef_matrix(ntip_idx, ttip_idx, tri)
    return rebuild_ddm.create_coef_matrix(all_idx, all_idx, tri)
