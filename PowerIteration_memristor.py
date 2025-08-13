#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Power Iteration on Memristor-based Systems via MELISO (MPI)
----------------------------------------------------------------

Purpose
-------
Estimate the operator norm (‖A‖₂, the largest singular value) of matrices
used in optimization problems, by replacing ALL matrix–vector (matvec)
operations with MELISO's distributed memristor-based MVM.

MELISO/MPI conventions
----------------------
MELISO uses MPI with *the last rank as the "root"* (controller). The number of
worker ranks must match your MCA grid: size-1 == mca_rows * mca_cols (from config).
Set your device/distribution parameters via the same EXP_CONFIG_FILE YAML used in PDHG.

Requirements (environment)
--------------------------
- EXP_CONFIG_FILE: path to the YAML experiment config (required by MELISO).
- REPORT_PATH: a writable file path where MELISO appends experiment logs.
               If not set, this script will set it to reports/PowerIteration/meliso_report.log
- MELISO_SRC_PATH: (optional) if you have a non-standard src layout.

Usage examples
--------------
Single matrix:
    mpirun -n <K+1> python PowerIteration_memristor.py \
        --matrix inputs/problems/converted/relaxed_gen-ip002.npz \
        --max-iter 2000 --tol 1e-6 --correction \
        --reports-dir reports/PowerIteration --seed 0

Batch of matrices (one path per line):
    mpirun -n <K+1> python PowerIteration_memristor.py \
        --batch inputs/problems/converted/batch_list.txt \
        --max-iter 2000 --tol 1e-6 --correction \
        --reports-dir reports/PowerIteration --seed 0

Self-test (quick single MVM sanity check using MELISO's matvec; see notes):
    mpirun -n <K+1> python PowerIteration_memristor.py --selftest

Notes on "selftest":
- It performs a single A@x and compares with NumPy on the *root rank output*
  to detect obvious wiring/config issues. It requires a valid EXP_CONFIG_FILE.
- This is not a comprehensive hardware-accuracy test; it's a functional check.

Author: Vo, Huynh Quang Nguyen.
Date: 2025-08-12
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from mpi4py import MPI
from scipy.io import mmread
import scipy.sparse as sp

# --- MELISO matvec solver ---
try:
    from solver.matvec.MatVecSolver import MatVecSolver
except Exception as e:
    print("ERROR: Failed to import MatVecSolver from MELISO. "
          "Ensure PYTHONPATH includes the MELISO project root (folder containing 'solver/').",
          file=sys.stderr)
    raise


# =============================================================================
# Utilities
# =============================================================================

def _ensure_reports_dir(path: str, is_root: bool) -> None:
    if is_root:
        os.makedirs(path, exist_ok=True)


def _set_default_report_path(reports_dir: str, is_root: bool) -> None:
    """
    MELISO's RootMCA reads REPORT_PATH unconditionally. Provide a safe default.
    """
    default_path = os.path.join(reports_dir, "meliso_report.log")
    if "REPORT_PATH" not in os.environ or not os.environ["REPORT_PATH"]:
        # Set for all ranks to avoid KeyError in NonRoot/Root code paths
        os.environ["REPORT_PATH"] = default_path
        if is_root:
            print(f"[root] REPORT_PATH not set; defaulting to {default_path}")


def _require_exp_config(is_root: bool) -> None:
    if "EXP_CONFIG_FILE" not in os.environ or not os.environ["EXP_CONFIG_FILE"]:
        if is_root:
            print("ERROR: EXP_CONFIG_FILE environment variable is not set. "
                  "Point it to your MELISO YAML configuration.", file=sys.stderr)
        # Synchronize and exit cleanly for all ranks
        MPI.COMM_WORLD.Barrier()
        sys.exit(2)


def _basename_wo_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _load_matrix(path: str) -> np.ndarray:
    """
    Load matrix A from supported formats:
    - .npz with keys A_row, A_col, A_data, A_shape  (COO sparse)
    - .mtx (Matrix Market)
    - .npy (dense)
    - .csv/.txt (dense via np.loadtxt)
    Returns dense float64 ndarray (MELISO expects float64).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        req = ["A_row", "A_col", "A_data", "A_shape"]
        missing = [k for k in req if k not in data]
        if missing:
            raise KeyError(f"NPZ file missing fields: {missing}")
        A = sp.coo_matrix((data["A_data"], (data["A_row"], data["A_col"])),
                          shape=tuple(data["A_shape"])).astype(np.float64).toarray()
    elif ext == ".mtx":
        A = mmread(path)
        A = A.astype(np.float64).toarray() if sp.issparse(A) else np.array(A, dtype=np.float64)
    elif ext == ".npy":
        A = np.load(path).astype(np.float64)
    else:
        # try csv/txt dense
        A = np.loadtxt(path, delimiter=",").astype(np.float64)
    return A


def _nondegenerate_unit_rand(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Tiny random positive vector (avoids zero/degenerate starts), normalized.
    Mirrors PDHG's safer init philosophy.
    """
    v = rng.random(n) + 1e-12
    v /= np.linalg.norm(v) + 1e-18
    return v


def _read_y_from_solver(solver: MatVecSolver, fallback_file: str = "y_mem_result.csv") -> np.ndarray:
    """
    Robustly retrieve y from MELISO:
    - Prefer attribute on the underlying solver object if exposed
    - Fall back to the CSV that RootMCA writes
    """
    # The underlying object is called "solver_object" in MatVecSolver
    y = None
    try:
        # Root exposes y_mem_result after parallelMatVec()
        y = np.array(solver.solver_object.y_mem_result, dtype=np.float64).reshape(-1)
    except Exception:
        pass
    if y is None or y.size == 0:
        y = np.loadtxt(fallback_file, delimiter=",").astype(np.float64).reshape(-1)
    return y


# =============================================================================
# MatVec wrapper (thin, re-usable)
# =============================================================================

class MELISOMatVec:
    """
    A minimal, robust wrapper that calls MELISO for A@x and A^T@y.

    - Instantiated once per process
    - Each call initializes matrix & vector on the MELISO side
    - Reads result from solver (attribute if available; CSV as fallback)
    - Finalizes after each call to keep the lifecycle simple & leak-free
    """
    def __init__(self, correction: bool) -> None:
        self.solver = MatVecSolver()
        self.correction = bool(correction)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.root = self.comm.Get_size() - 1  # MELISO uses the last rank as root

    def Av(self, A: np.ndarray, v: np.ndarray) -> np.ndarray:
        # Initialize on root; other ranks no-op inside initialize_data()
        self.solver.initialize_data(A, v)
        self.solver.matvec_mul(correction=self.correction)
        y = _read_y_from_solver(self.solver)
        # Ensure all ranks see the same y
        y = self.comm.bcast(y if self.rank == self.root else None, root=self.root)
        # Wrap-up (safe to call on all ranks)
        self.solver.finalize()
        self.solver.acquire_mca_stats()
        return y

    def ATu(self, AT: np.ndarray, u: np.ndarray) -> np.ndarray:
        # identical to Av but with A^T as the matrix
        self.solver.initialize_data(AT, u)
        self.solver.matvec_mul(correction=self.correction)
        z = _read_y_from_solver(self.solver)
        z = self.comm.bcast(z if self.rank == self.root else None, root=self.root)
        self.solver.finalize()
        self.solver.acquire_mca_stats()
        return z


# =============================================================================
# Power iteration
# =============================================================================

@dataclass
class PIConfig:
    max_iter: int = 2000
    tol: float = 1e-6
    correction: bool = True
    seed: int = 0


def power_iteration_norm(A: np.ndarray, mv: MELISOMatVec, cfg: PIConfig,
                         is_root: bool) -> Tuple[float, int]:
    """
    Two-sided power iteration for ‖A‖₂ (largest singular value).
    Returns (norm_estimate, iterations_used).
    """
    m, n = A.shape
    rng = np.random.default_rng(cfg.seed)
    v = _nondegenerate_unit_rand(n, rng)

    # Iterate
    for k in range(1, cfg.max_iter + 1):
        # 1) u ← A v / ‖A v‖
        w = mv.Av(A, v)
        w_norm = np.linalg.norm(w)
        if w_norm < cfg.tol:
            if is_root:
                print(f"[root] Converged at iter {k} (‖A v‖ ≈ 0).")
            # zero-ish matrix
            return 0.0, k
        u = w / (w_norm + 1e-18)

        # 2) v_new ← Aᵀ u / ‖Aᵀ u‖
        z = mv.ATu(A.T, u)
        z_norm = np.linalg.norm(z)
        if z_norm < cfg.tol:
            if is_root:
                print(f"[root] Converged at iter {k} (‖Aᵀ u‖ ≈ 0).")
            return 0.0, k
        v_new = z / (z_norm + 1e-18)

        # 3) Check vector change
        if np.linalg.norm(v_new - v) < cfg.tol:
            v = v_new
            if is_root:
                print(f"[root] Converged at iter {k} (vector change < tol).")
            break
        v = v_new

    # One final multiply to get ‖A v‖
    norm_est = float(np.linalg.norm(mv.Av(A, v)))
    return norm_est, k


# =============================================================================
# I/O helpers
# =============================================================================

def write_result_files(reports_dir: str, matrix_path: str, norm_est: float,
                       iters: int, cfg: PIConfig, A_shape: Tuple[int, int],
                       is_root: bool) -> None:
    """
    - Per-matrix text file: <matrix_name>_norm.txt
    - Aggregated CSV: power_iteration_results.csv (append)
    """
    if not is_root:
        return
    name = _basename_wo_ext(matrix_path)
    # Per-matrix file
    out_txt = os.path.join(reports_dir, f"{name}_norm.txt")
    with open(out_txt, "w") as f:
        f.write(f"{norm_est}\n")

    # Aggregated CSV
    csv_path = os.path.join(reports_dir, "power_iteration_results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "matrix", "rows", "cols",
                        "max_iter", "tol", "correction", "seed",
                        "norm_estimate", "iterations_used"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                    os.path.relpath(matrix_path),
                    A_shape[0], A_shape[1],
                    cfg.max_iter, cfg.tol, int(cfg.correction), cfg.seed,
                    norm_est, iters])
    print(f"[root] Saved: {out_txt}")
    print(f"[root] Appended: {csv_path}")


# =============================================================================
# Self-test (optional, quick functional check)
# =============================================================================

def selftest(mv: MELISOMatVec, is_root: bool) -> int:
    """
    Perform one MELISO A@x multiply on a small dense matrix and compare to NumPy.
    This requires a valid EXP_CONFIG_FILE and REPORT_PATH.
    Returns 0 on (rough) success, non-zero otherwise.
    """
    A = np.array([[3.0, 1.0, 0.0],
                  [1.0, 2.0, 1.0],
                  [0.0, 1.0, 3.0]], dtype=np.float64)
    x = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    y_hw = mv.Av(A, x)
    y_np = A @ x
    err = np.linalg.norm(y_hw - y_np) / (np.linalg.norm(y_np) + 1e-18)
    if is_root:
        print(f"[selftest] relative L2 error ≈ {err:.3e} (expect small if hardware/scaling/correction "
              f"are configured appropriately)")
    return 0


# =============================================================================
# Main
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Power Iteration using MELISO (memristor-based MVM)")
    g_in = p.add_argument_group("Input selection")
    g_in.add_argument("--matrix", "-m", type=str, help="Path to a single matrix file (.npz/.mtx/.npy/.csv).")
    g_in.add_argument("--batch", "-b", type=str, help="Path to a text file containing a list of matrices (one per line).")

    g_run = p.add_argument_group("Run configuration")
    g_run.add_argument("--max-iter", type=int, default=2000, help="Maximum iterations for power iteration.")
    g_run.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance on vector change.")
    g_run.add_argument("--correction", action="store_true", help="Enable MELISO output correction.")
    g_run.add_argument("--seed", type=int, default=0, help="Random seed for the initial vector.")

    g_io = p.add_argument_group("I/O")
    g_io.add_argument("--reports-dir", type=str, default="reports/PowerIteration",
                      help="Where to write <matrix>_norm.txt and power_iteration_results.csv")
    g_io.add_argument("--config", "-c", type=str, help="Path to MELISO YAML config (sets EXP_CONFIG_FILE).")

    p.add_argument("--selftest", action="store_true", help="Run a quick MELISO matvec check and exit.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # MPI layout
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = size - 1  # MELISO expects the last rank to be root
    is_root = (rank == root)

    # Reports dir + REPORT_PATH default
    _ensure_reports_dir(args.reports_dir, is_root=is_root)
    _set_default_report_path(args.reports_dir, is_root=is_root)

    # EXP_CONFIG_FILE (must be set before invoking MELISO classes)
    if args.config:
        os.environ["EXP_CONFIG_FILE"] = args.config
    _require_exp_config(is_root=is_root)

    # Create matvec helper
    mv = MELISOMatVec(correction=args.correction)

    # Optional self-test
    if args.selftest:
        rc = selftest(mv, is_root=is_root)
        # Ensure all ranks exit together
        comm.Barrier()
        sys.exit(rc)

    # Build task list
    tasks: List[str] = []
    if args.batch:
        if is_root:
            with open(args.batch, "r") as f:
                tasks = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        tasks = comm.bcast(tasks if is_root else None, root=root)
    elif args.matrix:
        tasks = [args.matrix]
    else:
        if is_root:
            print("ERROR: Provide --matrix or --batch.", file=sys.stderr)
        comm.Barrier()
        sys.exit(2)

    # Run config
    cfg = PIConfig(max_iter=args.max_iter, tol=args.tol,
                   correction=args.correction, seed=args.seed)

    # Process each matrix
    for mat_path in tasks:
        # Root loads matrix then broadcasts dense array to all
        if is_root:
            print(f"\n=== Power Iteration: {mat_path} ===")
            A = _load_matrix(mat_path)
            shape = A.shape
            print(f"[root] Loaded A with shape {shape}")
        else:
            A = None
            shape = None
        shape = MPI.COMM_WORLD.bcast(shape if is_root else None, root=root)
        if not is_root:
            A = np.empty(shape, dtype=np.float64)
        MPI.COMM_WORLD.Bcast([A, MPI.DOUBLE], root=root)

        # Compute norm
        t0 = time.time()
        norm_est, iters = power_iteration_norm(A, mv, cfg, is_root=is_root)
        t1 = time.time()
        if is_root:
            print(f"[root] ‖A‖₂ ≈ {norm_est:.6e}  (iters={iters}, time={t1 - t0:.3f}s)")

        # Save outputs (root only)
        write_result_files(args.reports_dir, mat_path, norm_est, iters, cfg, A.shape, is_root=is_root)

    # Final sync + end
    MPI.COMM_WORLD.Barrier()
    if is_root:
        print("\nAll tasks complete.")
    return


if __name__ == "__main__":
    main()
