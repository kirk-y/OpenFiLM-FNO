from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np


def load_metadata(sample_dir: Path) -> dict:
    meta_path = sample_dir / "sample_metadata.json"
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def triangular_value(tri: np.ndarray, i: int, j: int) -> float:
    if i > j:
        i, j = j, i
    idx = j * (j + 1) // 2 + i
    return tri[idx]


def create_coef_matrix(idx_rows: np.ndarray, idx_cols: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """
    从一维三角存储 tri 中构建系数矩阵 C[i,j] = tri[triangular_index(ii,jj)]。
    使用 NumPy 广播向量化，避免 Python 双重 for 循环，速度提升约 100-1000x。
    """
    idx_rows = idx_rows.astype(np.int64)
    idx_cols = idx_cols.astype(np.int64)
    if (idx_rows < 0).any() or (idx_cols < 0).any():
        raise ValueError("index arrays contain -1; check grid mapping/export")
    # 广播成 (m, n)，取 min/max 保证 a <= b（对称三角存储规则）
    ii = idx_rows[:, np.newaxis]   # (m, 1)
    jj = idx_cols[np.newaxis, :]   # (1, n)
    a = np.minimum(ii, jj)
    b = np.maximum(ii, jj)
    flat_idx = b * (b + 1) // 2 + a   # (m, n) 平铺索引
    return tri[flat_idx]


def validate_sample(sample_dir: Path) -> float | None:
    meta = load_metadata(sample_dir)
    ddm = meta.get("ddm", {})

    tri = np.load(sample_dir / ddm["global"]["triangular_value"]).astype(np.float64)
    ddm_idx = np.load(sample_dir / ddm["ddm_idx_file"]).astype(np.int64)

    ranges = ddm["ranges"]
    ntip_start = ranges["ntip"]["start"]
    ntip_count = ranges["ntip"]["count"]
    ttip_start = ranges["ttip"]["start"]
    ttip_count = ranges["ttip"]["count"]
    all_start = ranges["all"]["start"]
    all_count = ranges["all"]["count"]

    ntip_idx = ddm_idx[ntip_start: ntip_start + ntip_count]
    ttip_idx = ddm_idx[ttip_start: ttip_start + ttip_count]
    all_idx = ddm_idx[all_start: all_start + all_count]

    Ccc = create_coef_matrix(ntip_idx, ntip_idx, tri)
    Cct = create_coef_matrix(ntip_idx, ttip_idx, tri)
    C = create_coef_matrix(all_idx, all_idx, tri)

    print(f"[{sample_dir.name}] Ccc shape:", Ccc.shape)
    print(f"[{sample_dir.name}] Cct shape:", Cct.shape)
    print(f"[{sample_dir.name}] C shape:", C.shape)

    c_ref_path = sample_dir / ddm["tep_C_file"]
    if c_ref_path.exists():
        C_ref = np.load(c_ref_path).astype(np.float64)
        diff = np.linalg.norm(C - C_ref)
        denom = max(np.linalg.norm(C_ref), 1e-12)
        rel = diff / denom
        print(f"[{sample_dir.name}] C relative error:", rel)
        return rel
    return None


def main() -> None:
    if len(sys.argv) > 1:
        case_dir = Path(sys.argv[1]).resolve()
    else:
        case_dir = Path.cwd().resolve()

    samples_dir = case_dir / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(f"samples dir not found: {samples_dir}")

    sample_dirs = sorted([p for p in samples_dir.iterdir() if p.is_dir() and (p / "sample_metadata.json").exists()])
    if not sample_dirs:
        raise FileNotFoundError("no samples found under case_dir/samples")

    rel_errors = []
    for sample_dir in sample_dirs:
        rel = validate_sample(sample_dir)
        if rel is not None:
            rel_errors.append(rel)

    if rel_errors:
        print("total samples:", len(rel_errors))
        print("max relative error:", max(rel_errors))


if __name__ == "__main__":
    main()
