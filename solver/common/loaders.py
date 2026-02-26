# -*- coding-*- utf-8 -*-
"""
Matrix loading utilities for optimization problems.

Supports loading from:
 - Sparse snapshot .npz files (from PDHG solvers)
 - Dense .npz files
 - .mtx (Matrix Market) files
 - .npy (Numpy binary) files
"""

from __future__ import annotations
import os
import numpy as np
from typing import Optional

__all__ = ["load_matrix_any"]

def _maybe_apply_scaling(A: np.ndarray, z: dict) -> np.ndarray:
    """Applies row/column scaling vectors if found in the NPZ archive."""
    apply_r = str(os.environ.get("APPLY_ROW_SCALING", "0")).lower() in ("1", "true", "yes")
    apply_c = str(os.environ.get("APPLY_COL_SCALING", "0")).lower() in ("1", "true", "yes")
    if not (apply_r or apply_c):
        return A
    
    R = z.get("row_scale_vec")
    C = z.get("col_scale_vec")
    
    if apply_r and R is not None:
        R = np.asarray(R, dtype=float).reshape(-1)
        if R.shape[0] == A.shape[0]:
            A = (R[:, None]) * A
            
    if apply_c and C is not None:
        C = np.asarray(C, dtype=float).reshape(-1)
        if C.shape[0] == A.shape[1]:
            A = A * (C[None, :])
            
    return A

def _load_constraint_from_npz(snapshot_path: str, prefix: Optional[str] = None, to_dense: bool = True) -> np.ndarray:
    """Loads a sparse matrix from a PDHG snapshot .npz file."""
    try:
        from scipy.sparse import coo_matrix
    except ImportError as e:
        raise RuntimeError("scipy must be installed to load sparse NPZ constraints.") from e

    with np.load(snapshot_path, allow_pickle=False) as z:
        candidates = [prefix] if prefix else ["A", "G", "H"]
        chosen = None
        for p in candidates:
            required_keys = [f"{p}_data", f"{p}_row", f"{p}_col", f"{p}_shape"]
            if all(key in z for key in required_keys):
                chosen = p
                break
        
        if chosen is None:
            raise ValueError(f"No suitable prefix in {snapshot_path}. Need keys like: <prefix>_data/_row/_col/_shape.")
            
        data, row, col = z[f"{chosen}_data"], z[f"{chosen}_row"], z[f"{chosen}_col"]
        shape = tuple(int(x) for x in z[f"{chosen}_shape"].reshape(-1))
        
        A = coo_matrix((data, (row, col)), shape=shape)
        if to_dense:
            A = A.toarray()
            
        A = _maybe_apply_scaling(np.asarray(A, dtype=float), z)
        return np.asarray(A, dtype=float)

def load_matrix_any(path: str) -> np.ndarray:
    """
    Loads a matrix from a file, auto-detecting the format.
    
    Supports .npz (sparse/dense), .mtx, and .npy files.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("A valid file path must be provided.")
        
    lower_path = path.lower()
    
    if lower_path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.keys())
        is_snapshot = any("_data" in k and "_row" in k for k in keys)
        
        if is_snapshot:
            prefix = os.environ.get("NPZ_PREFIX", "A")
            to_dense = str(os.environ.get("NPZ_TO_DENSE", "1")).lower() in ("1", "true", "yes")
            return _load_constraint_from_npz(path, prefix=prefix, to_dense=to_dense)
        else:
            with np.load(path, allow_pickle=False) as z:
                key = "A" if "A" in z else (keys[0] if keys else None)
                if key is None:
                    raise ValueError(f"No arrays found in the dense NPZ file: {path}")
                A = np.asarray(z[key], dtype=float)
                print(f"[LOAD][NPZ-dense] loaded array '{key}' with shape={A.shape}", flush=True)
                return A
                
    elif lower_path.endswith(".mtx"):
        try:
            from scipy.io import mmread
        except ImportError as e:
            raise RuntimeError("scipy must be installed to load .mtx files.") from e
        M = mmread(path)
        A = M.toarray() if hasattr(M, "toarray") else np.asarray(M, dtype=float)
        print(f"[LOAD][MTX] loaded matrix with shape={A.shape}", flush=True)
        return A.astype(float)
        
    elif lower_path.endswith(".npy"):
        A = np.load(path).astype(float)
        print(f"[LOAD][NPY] loaded matrix with shape={A.shape}", flush=True)
        return A
        
    else:
        raise ValueError(f"Unsupported matrix file type: {path}. Please use .npz, .mtx, or .npy.")