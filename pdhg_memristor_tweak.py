
"""
PDHG on Memristor-based Systems via MELISO (MPI)

Changes in this version:
- Operator norm (step-size) estimation moved to CPU (SciPy/NumPy) to reserve MCA for PDHG.
- Added diagnostics (primal/dual residuals, KKT projected-gradient residual, duality gap, objective).
- Added early stopping with patience/min_delta and configurable monitor metric.
- Optional plots saved on root rank.

Run with mpirun/mpiexec, e.g.:
    mpirun -n 4 python pdhg_memristor.py \
        --npz problems/converted/relaxed_foo.npz \
        --max_iter 200000 --tol 1e-6 --theta 1.0 \
        --patience 5000 --monitor gap --min_delta 0.0 \
        --plot_prefix diag --plot_every 500 --save_logs logs.csv --save_solution x.npy --correction
"""

from __future__ import annotations

import os
import sys
import math
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import scipy.sparse as sp

from mpi4py import MPI

# --- Ensure UTF-8 logs even under non-UTF locales ---
import io
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


# Import MELISO distributed MVM wrapper
try:
    from solver.matvec.MatVecSolver import MatVecSolver
except Exception as e:
    print("Failed to import MatVecSolver from MELISO. Ensure PYTHONPATH includes the"
          " MELISO project root (the folder that contains 'solver/').", file=sys.stderr)
    raise


# --------------------------------------------------------------------------------------
# Utilities: file I/O
# --------------------------------------------------------------------------------------

def load_problem_data(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    required_fields = ['A_row', 'A_col', 'A_data', 'A_shape', 'b', 'c', 'lb', 'ub']
    missing = [k for k in required_fields if k not in data]
    if missing:
        raise KeyError(f"Missing fields in NPZ: {missing}")
    return {
        'A_row': data['A_row'],
        'A_col': data['A_col'],
        'A_data': data['A_data'],
        'A_shape': tuple(data['A_shape']),
        'b': data['b'],
        'c': data['c'],
        'lb': data['lb'],
        'ub': data['ub'],
    }


def create_sparse_matrices(data: Dict[str, np.ndarray]) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    A_coo = sp.coo_matrix((data['A_data'], (data['A_row'], data['A_col'])), shape=data['A_shape'])
    A = A_coo.tocsr()
    AT = A.transpose().tocsr()
    return A, AT


def validate_problem_dimensions(data: Dict[str, np.ndarray]) -> None:
    m, n = data['A_shape']
    assert data['b'].shape[0] == m, "b length mismatch"
    assert data['c'].shape[0] == n, "c length mismatch"
    assert data['lb'].shape[0] == n, "lb length mismatch"
    assert data['ub'].shape[0] == n, "ub length mismatch"
    if np.any(data['lb'] > data['ub']):
        bad = np.where(data['lb'] > data['ub'])[0][:5]
        raise ValueError(f"Infeasible bounds (lb>ub) at indices (first few): {bad}")



def _ensure_nondegenerate_vec(v: np.ndarray) -> np.ndarray:
    # Ensure the vector has a non-zero range so MELISO scaling doesn't fail.
    vv = np.asarray(v, dtype=np.float64).reshape(-1).copy()
    if not np.all(np.isfinite(vv)):
        vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)
    if vv.size == 0:
        return vv
    if (vv.max() - vv.min()) == 0.0:
        eps = 1e-12
        ramp = eps * np.arange(1, vv.size + 1, dtype=np.float64)
        vv = vv + ramp
    return vv

# --------------------------------------------------------------------------------------
# Memristor MVM wrapper
# --------------------------------------------------------------------------------------

class MemristorMatVec:
    """
    Thin wrapper around MELISO's MatVecSolver to perform y = A @ x on the MCA.
    The same instance is shared across MPI ranks; Root provides data, workers assist.
    Result vector is returned (broadcast to all ranks).
    """

    def __init__(self, comm: MPI.Comm, root_rank: Optional[int] = None):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.root = root_rank if root_rank is not None else (self.size - 1)

        # Ensure REPORT_PATH exists for MELISO BaseMCA
        if "REPORT_PATH" not in os.environ:
            os.environ["REPORT_PATH"] = os.getcwd()

        # One solver per rank
        self.solver = MatVecSolver()

    def matvec(self, A, x, correction: bool = False):
        """
        Perform a single distributed matvec on the MCA: y = A @ x.
        Must be called by all ranks in lock-step.
        """
        # Initialize on root (no-op on workers)
        self.solver.initialize_data(A, _ensure_nondegenerate_vec(x))
        # All ranks participate
        self.solver.matvec_mul(correction=correction)
        # Retrieve result only on root (then broadcast)
        if self.rank == self.root:
            y = np.copy(self.solver.solver_object.y_mem_result).reshape(-1)
        else:
            y = None
        y = self.comm.bcast(y, root=self.root)
        return y

    def finalize(self):
        try:
            self.solver.finalize()
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# Operator norm estimation on CPU (SciPy)
# --------------------------------------------------------------------------------------

def estimate_operator_norm_cpu(A: sp.csr_matrix, iters: int = 200, seed: int = 0) -> float:
    """
    Estimate ||A||_2 via power iteration on CPU (no MCA involvement):
        x_{k+1} = A^T (A x_k) / ||A^T (A x_k)||
        ||A||_2 ≈ ||A x_k||
    """
    rng = np.random.default_rng(seed)
    n = A.shape[1]
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x) + 1e-20
    for _ in range(iters):
        z = A @ x
        x = A.T @ z
        nrm = np.linalg.norm(x)
        if nrm == 0:
            return 0.0
        x /= nrm
    return float(np.linalg.norm(A @ x))


# --------------------------------------------------------------------------------------
# Projections & KKT residuals
# --------------------------------------------------------------------------------------

def project_box(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lb), ub)


def kkt_projected_gradient_norm(x: np.ndarray, grad: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    """
    KKT stationarity residual for bound constraints, measured via projected gradient:
        r_pg = x - P_[lb,ub](x - grad)
    Uses L2 norm of r_pg.
    """
    pg = x - project_box(x - grad, lb, ub)
    return float(np.linalg.norm(pg))


def kkt_complementarity_violation(x: np.ndarray, grad: np.ndarray, lb: np.ndarray, ub: np.ndarray, tol_b: float = 1e-10) -> float:
    """
    Complementarity-style violation for bound-constrained LP:
      - if x_i ~ lb_i: require grad_i >= 0  -> violation = max(0, -grad_i)
      - if x_i ~ ub_i: require grad_i <= 0  -> violation = max(0,  grad_i)
      - if lb_i < x_i < ub_i: require grad_i ~ 0 -> violation = |grad_i|
    Returns L-infinity norm of per-coordinate violations.
    """
    v = np.zeros_like(x)
    at_lb = x <= (lb + tol_b)
    at_ub = x >= (ub - tol_b)
    strict = (~at_lb) & (~at_ub)
    v[at_lb] = np.maximum(0.0, -grad[at_lb])
    v[at_ub] = np.maximum(0.0,  grad[at_ub])
    v[strict] = np.abs(grad[strict])
    return float(np.linalg.norm(v, ord=np.inf))


# --------------------------------------------------------------------------------------
# PDHG solver using memristor MVMs
# --------------------------------------------------------------------------------------

def solve_pdhg_memristor(problem: Dict[str, np.ndarray],
                         max_iterations: int = 200000,
                         tolerance: float = 1e-6,
                         theta: float = 1.0,
                         correction: bool = False,
                         norm_iters: int = 200,
                         early_stopping_patience: Optional[int] = None,
                         monitor: str = "gap",           # {"gap","primal_dual","kkt"}
                         min_delta: float = 0.0,
                         plot_prefix: Optional[str] = None,
                         plot_every: int = 0,
                         save_logs: Optional[str] = None,
                         verbose: bool = True) -> Tuple[np.ndarray, Dict[str, float], Dict[str, List[float]]]:
    """
    PDHG solver where A @ v and A^T @ v are computed on the MCA via MELISO.
    All ranks must call into this function. Only root performs updates; workers join MVM.
    Returns (x, stats, logs).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = size - 1

    # Build sparse A
    A, AT = create_sparse_matrices(problem)
    m, n = A.shape
    b = problem['b']
    c = problem['c']
    lb = problem['lb']
    ub = problem['ub']

    # Instantiate MVM helper
    mmv = MemristorMatVec(comm, root_rank=root)

    # Step sizes from CPU norm estimate
    if rank == root and verbose:
        print("\n=== Phase 1: Estimating operator norm on CPU ===")
    L = estimate_operator_norm_cpu(A, iters=norm_iters)
    L = max(L, 1e-12)
    tau = 0.95 / L
    sigma = 0.95 / L
    if rank == root and verbose:
        print(f"||A||_2 ≈ {L:.6f}  ->  τ={tau:.6e}, σ={sigma:.6e}\n")
        print("=== Phase 2: PDHG iterations on MCA (MVMs offloaded) ===")

    # Initialize variables on root
    if rank == root:
        x = np.zeros(n, dtype=np.float64)
        x_bar = x.copy()
        y = np.zeros(m, dtype=np.float64)
        best_x = x.copy()
        # logs
        logs = dict(iter=[], primal_res=[], dual_res=[], kkt_pg=[], kkt_comp=[], gap=[], obj=[])
        # early stopping state
        patience = int(early_stopping_patience) if early_stopping_patience is not None else None
        best_metric = np.inf
        since_improve = 0
    else:
        x = x_bar = y = None
        logs = dict()

    comm.barrier()

    converged = False
    final_iter = max_iterations
    primal_res = dual_res = gap = np.inf
    obj_val = np.inf
    kkt_pg = kkt_comp = np.inf

    for it in range(max_iterations):
        # --- Dual ascent: y += σ (A x̄ - b) ---
        Axbar = mmv.matvec(A, x_bar if rank == root else np.empty(A.shape[1]), correction=correction)

        if rank == root:
            residual = Axbar - b
            y = y + sigma * residual

        # --- Primal descent: x -= τ (A^T y + c); then project to [lb, ub] ---
        ATy = mmv.matvec(AT, y if rank == root else np.empty(AT.shape[1]), correction=correction)

        if rank == root:
            grad = ATy + c
            x_old = x.copy()
            x = x - tau * grad
            x = project_box(x, lb, ub)
            x_bar = x + theta * (x - x_old)

            # Diagnostics
            primal_res = np.linalg.norm(Axbar - b) / (1 + np.linalg.norm(b))
            dual_res   = np.linalg.norm(ATy + c)   / (1 + np.linalg.norm(c))
            obj_val    = float(c @ x)
            gap        = abs(obj_val + float(b @ y)) / (1 + abs(obj_val))
            kkt_pg     = kkt_projected_gradient_norm(x, grad, lb, ub)
            kkt_comp   = kkt_complementarity_violation(x, grad, lb, ub)

            logs['iter'].append(it)
            logs['primal_res'].append(primal_res)
            logs['dual_res'].append(dual_res)
            logs['gap'].append(gap)
            logs['obj'].append(obj_val)
            logs['kkt_pg'].append(kkt_pg)
            logs['kkt_comp'].append(kkt_comp)

            # Convergence check
            if max(primal_res, dual_res, gap) <= tolerance:
                converged = True
                final_iter = it + 1
                best_x = x.copy()
                break

            # Early stopping
            if early_stopping_patience is not None:
                if monitor == "gap":
                    metric = gap
                elif monitor == "primal_dual":
                    metric = max(primal_res, dual_res)
                elif monitor == "kkt":
                    metric = kkt_pg  # or combine with kkt_comp
                else:
                    metric = gap

                improved = (best_metric - metric) > min_delta
                if improved:
                    best_metric = metric
                    best_x = x.copy()
                    since_improve = 0
                else:
                    since_improve += 1
                    if since_improve >= patience:
                        final_iter = it + 1
                        break

        # Keep ranks in sync
        comm.barrier()

    # Finalize MELISO
    mmv.finalize()

    # Only root assembles results
    if rank == root:
        x_out = best_x if (early_stopping_patience is not None and not converged) else x

        stats = dict(
            iterations=final_iter,
            converged=bool(converged),
            primal_res=float(primal_res),
            dual_res=float(dual_res),
            gap=float(gap),
            obj=float(obj_val),
            kkt_pg=float(kkt_pg),
            kkt_comp=float(kkt_comp),
            L=float(L),
            tau=float(tau),
            sigma=float(sigma),
            early_stopped=bool(early_stopping_patience is not None and not converged),
            monitor=monitor,
            best_metric=float(best_metric if 'best_metric' in locals() else np.inf),
        )

        # Save logs if requested
        if save_logs:
            import csv
            with open(save_logs, 'w', newline='') as f:
                w = csv.writer(f)
                headers = ['iter','primal_res','dual_res','gap','obj','kkt_pg','kkt_comp']
                w.writerow(headers)
                for i in range(len(logs['iter'])):
                    w.writerow([logs[h][i] for h in headers])

        # Plots if requested
        if plot_prefix:
            try:
                import matplotlib.pyplot as plt

                if plot_every and plot_every > 0:
                    # Downsample logs for plotting if desired
                    idx = np.arange(0, len(logs['iter']), plot_every)
                else:
                    idx = np.arange(0, len(logs['iter']))

                iters = np.array(logs['iter'])[idx]

                # 1) Primal/Dual residuals
                plt.figure()
                plt.semilogy(iters, np.array(logs['primal_res'])[idx], label='primal_res')
                plt.semilogy(iters, np.array(logs['dual_res'])[idx], label='dual_res')
                plt.xlabel('Iteration')
                plt.ylabel('Residual (semilogy)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{plot_prefix}_primal_dual.png", dpi=150)
                plt.close()

                # 2) KKT projected gradient
                plt.figure()
                plt.semilogy(iters, np.array(logs['kkt_pg'])[idx], label='kkt_projected_grad')
                plt.xlabel('Iteration')
                plt.ylabel('KKT PG Residual')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{plot_prefix}_kkt_pg.png", dpi=150)
                plt.close()

                # 3) (Optional) Duality gap
                plt.figure()
                plt.semilogy(iters, np.array(logs['gap'])[idx], label='duality_gap')
                plt.xlabel('Iteration')
                plt.ylabel('Gap')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{plot_prefix}_gap.png", dpi=150)
                plt.close()

            except Exception as e:
                print(f"[plotting] Skipped plots due to error: {e}", file=sys.stderr)

        return x_out, stats, logs
    else:
        return np.array([]), {}, {}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="PDHG on MELISO (memristor-based MVM) with diagnostics & early stopping")
    p.add_argument("--npz", required=True, help="Path to NPZ with A (COO), b, c, lb, ub")
    p.add_argument("--max_iter", type=int, default=200000, help="Maximum PDHG iterations")
    p.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    p.add_argument("--theta", type=float, default=1.0, help="Over-relaxation parameter")
    p.add_argument("--norm_iters", type=int, default=200, help="CPU power-iteration steps for ||A||")
    p.add_argument("--correction", action="store_true", help="Enable MELISO output correction")
    # diagnostics
    p.add_argument("--plot_prefix", type=str, default=None, help="Save diagnostics plots with this prefix (root only)")
    p.add_argument("--plot_every", type=int, default=0, help="Downsample plotting every k points (0 = no downsample)")
    p.add_argument("--save_logs", type=str, default=None, help="CSV file to save iteration logs (root only)")
    # early stopping
    p.add_argument("--patience", type=int, default=None, help="Stop after this many iterations without improvement")
    p.add_argument("--monitor", type=str, default="gap", choices=["gap","primal_dual","kkt"], help="Metric to monitor")
    p.add_argument("--min_delta", type=float, default=0.0, help="Minimal improvement to reset patience")
    p.add_argument("--save_solution", type=str, default=None, help="Path to save x (npy) on root")
    return p


def main():
    args = build_argparser().parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = size - 1

    # Load problem on root, then broadcast
    if rank == root:
        try:
            problem = load_problem_data(args.npz)
            validate_problem_dimensions(problem)
        except Exception as e:
            print(f"[root] Failed to load problem: {e}", file=sys.stderr)
            err = dict(ok=False, msg=str(e))
        else:
            err = dict(ok=True, msg="")
    else:
        problem = None
        err = None

    err = comm.bcast(err, root=root)
    if not err["ok"]:
        if rank == root:
            print("[root] Aborting due to load/validation error.", file=sys.stderr)
        sys.exit(1)

    # Broadcast arrays and rebuild A on all ranks
    if rank != root:
        problem = {}
    keys = ["A_row", "A_col", "A_data", "A_shape", "b", "c", "lb", "ub"]
    for k in keys:
        val = problem[k] if rank == root else None
        val = comm.bcast(val, root=root)
        problem[k] = val

    # Solve
    x, stats, logs = solve_pdhg_memristor(
        problem,
        max_iterations=args.max_iter,
        tolerance=args.tol,
        theta=args.theta,
        norm_iters=args.norm_iters,
        correction=args.correction,
        early_stopping_patience=args.patience,
        monitor=args.monitor,
        min_delta=args.min_delta,
        plot_prefix=args.plot_prefix,
        plot_every=args.plot_every,
        save_logs=args.save_logs,
        verbose=(rank == root)
    )

    if rank == root:
        if args.save_solution:
            np.save(args.save_solution, x)
            print(f"Saved solution to: {args.save_solution}")

        print("\n=== PDHG (MELISO) Summary ===")
        for k, v in stats.items():
            print(f"{k:>16s}: {v}")
        print(f"||x||_2: {np.linalg.norm(x):.6f}")


if __name__ == "__main__":
    main()
