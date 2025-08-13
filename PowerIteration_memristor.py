#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Power Iteration on Memristor-based Systems via MELISO (MPI)
----------------------------------------------------------------

Usage (examples):
  mpiexec -n <K+1> python PowerIteration_memristor.py \
      --matrix inputs/problems/converted/relaxed_gen-ip002.npz \
      --max-iter 2000 --tol 1e-6 --correction \
      --reports-dir reports/PowerIteration --seed 0

  mpiexec -n <K+1> python PowerIteration_memristor.py \
      --batch inputs/problems/converted/batch_list.txt \
      --max-iter 2000 --tol 1e-6 --correction \
      --reports-dir reports/PowerIteration --seed 0

Quick sanity check:
  mpiexec -n <K+1> python PowerIteration_memristor.py --selftest

Env required:
  EXP_CONFIG_FILE (MELISO YAML). REPORT_PATH optional (we set default).
Root rank convention:
  MELISO uses the LAST MPI rank as the root (controller).
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
    default_path = os.path.join(reports_dir, "MELISO_report.log")
    if "REPORT_PATH" not in os.environ or not os.environ["REPORT_PATH"]:
        os.environ["REPORT_PATH"] = default_path
        if is_root:
            print(f"[root] REPORT_PATH not set; defaulting to {default_path}")

def _require_exp_config(is_root: bool) -> None:
    if "EXP_CONFIG_FILE" not in os.environ or not os.environ["EXP_CONFIG_FILE"]:
        if is_root:
            print("ERROR: EXP_CONFIG_FILE is not set. Point it to your MELISO YAML.", file=sys.stderr)
        MPI.COMM_WORLD.Barrier()
        sys.exit(2)

def _basename_wo_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def _load_matrix(path: str) -> np.ndarray:
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
        A = np.loadtxt(path, delimiter=",").astype(np.float64)
    return A

def _nondegenerate_unit_rand(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.random(n) + 1e-12
    v /= np.linalg.norm(v) + 1e-18
    return v

def _read_y_from_solver(solver: MatVecSolver, fallback_file: str = "y_mem_result.csv") -> np.ndarray:
    y = None
    try:
        y = np.array(solver.solver_object.y_mem_result, dtype=np.float64).reshape(-1)
    except Exception:
        pass
    if y is None or y.size == 0:
        y = np.loadtxt(fallback_file, delimiter=",").astype(np.float64).reshape(-1)
    return y

# =============================================================================
# MatVec wrapper
# =============================================================================

class MELISOMatVec:
    """Wrapper that executes A@x and A^T@y via MELISO (last rank is root)."""
    def __init__(self, correction: bool) -> None:
        self.solver = MatVecSolver()
        self.correction = bool(correction)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.root = self.comm.Get_size() - 1

    def Av(self, A: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.solver.initialize_data(A, v)
        self.solver.matvec_mul(correction=self.correction)
        y = _read_y_from_solver(self.solver)
        y = self.comm.bcast(y if self.rank == self.root else None, root=self.root)
        self.solver.finalize()
        self.solver.acquire_mca_stats()
        return y

    def ATu(self, AT: np.ndarray, u: np.ndarray) -> np.ndarray:
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

def power_iteration_norm(
    A: np.ndarray,
    mv: MELISOMatVec,
    cfg: PIConfig,
    is_root: bool,
    save_temp: bool = True,
    tmpvec_dir: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Tuple[float, int]:
    """
    Two-sided power iteration for ‖A‖₂ (largest singular value).
    Saves forward/adjoint vectors if save_temp=True (root rank only).
    """
    m, n = A.shape
    rng = np.random.default_rng(cfg.seed)
    v = _nondegenerate_unit_rand(n, rng)

    # Temp-vector saver (root only)
    def _maybe_save(arr: np.ndarray, phase: str, it: int) -> None:
        if not (save_temp and is_root):
            return
        if tmpvec_dir is None:
            return
        os.makedirs(tmpvec_dir, exist_ok=True)
        fname = f"{phase}_it{it:06d}_job{job_id if job_id else 'local'}.csv"
        fpath = os.path.join(tmpvec_dir, fname)
        np.savetxt(fpath, arr.reshape(-1), delimiter=",")

    # Iterate
    for k in range(1, cfg.max_iter + 1):
        # Forward: u ← A v / ‖A v‖
        w = mv.Av(A, v)
        _maybe_save(w, "fwd", k)
        w_norm = np.linalg.norm(w)
        if w_norm < cfg.tol:
            if is_root:
                print(f"[root] Converged at iter {k} (‖A v‖ ≈ 0).")
            return 0.0, k
        u = w / (w_norm + 1e-18)

        # Adjoint: v_new ← Aᵀ u / ‖Aᵀ u‖
        z = mv.ATu(A.T, u)
        _maybe_save(z, "adj", k)
        z_norm = np.linalg.norm(z)
        if z_norm < cfg.tol:
            if is_root:
                print(f"[root] Converged at iter {k} (‖Aᵀ u‖ ≈ 0).")
            return 0.0, k
        v_new = z / (z_norm + 1e-18)

        # Check vector change
        if np.linalg.norm(v_new - v) < cfg.tol:
            v = v_new
            if is_root:
                print(f"[root] Converged at iter {k} (vector change < tol).")
            break
        v = v_new

    # Final multiply for ‖A v‖
    norm_est = float(np.linalg.norm(mv.Av(A, v)))
    return norm_est, k

# =============================================================================
# I/O helpers
# =============================================================================

def write_result_files(
    reports_dir: str,
    matrix_path: str,
    norm_est: float,
    iters: int,
    cfg: PIConfig,
    A_shape: Tuple[int, int],
    is_root: bool,
) -> None:
    if not is_root:
        return
    name = _basename_wo_ext(matrix_path)
    out_txt = os.path.join(reports_dir, f"{name}_norm.txt")
    with open(out_txt, "w") as f:
        f.write(f"{norm_est}\n")

    csv_path = os.path.join(reports_dir, "power_iteration_results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "matrix", "rows", "cols",
                "max_iter", "tol", "correction", "seed",
                "norm_estimate", "iterations_used"
            ])
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            os.path.relpath(matrix_path),
            A_shape[0], A_shape[1],
            cfg.max_iter, cfg.tol, int(cfg.correction), cfg.seed,
            norm_est, iters
        ])
    print(f"[root] Saved: {out_txt}")
    print(f"[root] Appended: {csv_path}")

# =============================================================================
# Self-test (optional)
# =============================================================================

def selftest(mv: MELISOMatVec, is_root: bool) -> int:
    A = np.array([[3.0, 1.0, 0.0],
                  [1.0, 2.0, 1.0],
                  [0.0, 1.0, 3.0]], dtype=np.float64)
    x = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    y_hw = mv.Av(A, x)
    y_np = A @ x
    err = np.linalg.norm(y_hw - y_np) / (np.linalg.norm(y_np) + 1e-18)
    if is_root:
        print(f"[selftest] relative L2 error ≈ {err:.3e} (expect small if config is aligned)")
    return 0

# =============================================================================
# Main
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Power Iteration using MELISO (memristor-based MVM)")

    g_in = p.add_argument_group("Input selection")
    g_in.add_argument("--matrix", "-m", type=str, help="Path to a single matrix file (.npz/.mtx/.npy/.csv).")
    g_in.add_argument("--batch", "-b", type=str, help="Text file of matrix paths (one per line).")

    g_run = p.add_argument_group("Run configuration")
    g_run.add_argument("--max-iter", type=int, default=2000, help="Maximum iterations.")
    g_run.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance on vector change.")
    g_run.add_argument("--correction", action="store_true", help="Enable MELISO output correction.")
    g_run.add_argument("--seed", type=int, default=0, help="Random seed for initialization.")

    g_io = p.add_argument_group("I/O")
    g_io.add_argument("--reports-dir", type=str, default="reports/PowerIteration",
                      help="Where to write <matrix>_norm.txt and power_iteration_results.csv")
    g_io.add_argument("--config", "-c", type=str, help="Path to MELISO YAML config (sets EXP_CONFIG_FILE).")

    # Temp vectors: default on; can disable
    p.add_argument("--save-temp-vectors", dest="save_temp_vectors", action="store_true",
                   help="Save intermediate matvec outputs (CSV) with job-ID-based names (default).")
    p.add_argument("--no-save-temp-vectors", dest="save_temp_vectors", action="store_false",
                   help="Disable saving of intermediate matvec outputs.")
    p.add_argument("--tmpvec-dir", type=str, default=None,
                   help="Directory for temporary vectors (default: <reports-dir>/tmpvec_job<JOBID>).")

    p.add_argument("--selftest", action="store_true", help="Run a quick MELISO matvec check and exit.")

    # Defaults
    p.set_defaults(save_temp_vectors=True)
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = size - 1
    is_root = (rank == root)

    _ensure_reports_dir(args.reports_dir, is_root=is_root)
    _set_default_report_path(args.reports_dir, is_root=is_root)

    if args.config:
        os.environ["EXP_CONFIG_FILE"] = args.config
    _require_exp_config(is_root=is_root)

    # Job ID for unique filenames
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        job_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"

    # Temp vectors directory
    if args.tmpvec_dir:
        tmpvec_dir = args.tmpvec_dir
    else:
        tmpvec_dir = os.path.join(args.reports_dir, f"tmpvec_job{job_id}")

    mv = MELISOMatVec(correction=args.correction)

    if args.selftest:
        rc = selftest(mv, is_root=is_root)
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

    cfg = PIConfig(max_iter=args.max_iter, tol=args.tol,
                   correction=args.correction, seed=args.seed)

    for mat_path in tasks:
        if is_root:
            print(f"\n=== Power Iteration: {mat_path} ===")
            A = _load_matrix(mat_path)
            shape = A.shape
            print(f"[root] Loaded A with shape {shape}")
        else:
            A = None
            shape = None
        shape = comm.bcast(shape if is_root else None, root=root)
        if not is_root:
            A = np.empty(shape, dtype=np.float64)
        comm.Bcast([A, MPI.DOUBLE], root=root)

        t0 = time.time()
        norm_est, iters = power_iteration_norm(
            A, mv, cfg, is_root=is_root,
            save_temp=args.save_temp_vectors,
            tmpvec_dir=tmpvec_dir,
            job_id=job_id
        )
        t1 = time.time()
        if is_root:
            print(f"[root] ‖A‖₂ ≈ {norm_est:.6e}  (iters={iters}, time={t1 - t0:.3f}s)")

        write_result_files(args.reports_dir, mat_path, norm_est, iters, cfg, A.shape, is_root=is_root)

    comm.Barrier()
    if is_root:
        print("\nAll tasks complete.")

if __name__ == "__main__":
    main()
