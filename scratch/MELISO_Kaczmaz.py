#!/usr/bin/env python3
import os
import argparse
import numpy as np

from solver.matvec.MatVecSolver import MatVecSolver
from solver.matvec.Root import Root


def output_path(filename: str) -> str:
    base = os.environ.get("TMPDIR", ".")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)


def load_array(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.endswith(".npy"):
        return np.load(path)  # allow_pickle not needed for numeric arrays

    if path.endswith(".csv"):
        return np.loadtxt(path, delimiter=",")

    if path.endswith(".txt"):
        # try comma-delimited first, then whitespace
        try:
            return np.loadtxt(path, delimiter=",")
        except Exception:
            return np.loadtxt(path)

    raise ValueError(f"Unsupported file format: {path}") 
    

def build_augmented_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = b.reshape(-1, 1)
    return np.hstack((A, b))  # M = [A | b]
    return M

def correction_vectorized(root: Root, y_scaled: np.ndarray) -> np.ndarray:
    """
    Vectorized equivalent of root.addCorrectionY(n, y, a_min, a_rng, row_sum, x_min, x_rng, x_sum).

    NOTE:
      - root.mca.mat_max is actually ptp/range (a_rng)
      - root.x_max is actually ptp/range (x_rng)
      - n must be number of columns of augmented matrix M (n+1)
    """
    n_cols = float(root.origMatCols)
    a_min = float(root.mca.mat_min)
    a_rng = float(root.mca.mat_max)
    row_sum = np.asarray(root.mca.mat_row_sum, dtype=np.float64).reshape(-1)

    x_min = float(root.x_min)
    x_rng = float(root.x_max)
    x_sum = float(root.x_sum)

    y_scaled = np.asarray(y_scaled, dtype=np.float64).reshape(-1)

    # y = y_scaled*(a_rng*x_rng) + a_min*x_sum + x_min*row_sum - n_cols*a_min*x_min
    return (a_rng * x_rng) * y_scaled + (a_min * x_sum) + (x_min * row_sum) - (n_cols * a_min * x_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--row", choices=["random", "cyclic"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_prefix", default="x_kaczmarz")
    ap.add_argument("--save", action="store_true", help="Save x to TMPDIR")
    ap.add_argument("--check", action="store_true", help="Compare corrected r vs numpy (debug)")
    args = ap.parse_args()

    A = load_array(args.A).astype(np.float64)
    b = load_array(args.b).astype(np.float64).reshape(-1)

    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError(f"Dimension mismatch: A has {m} rows but b has {b.shape[0]} entries")

    # Augmented setup: M=[A|b], z=[x;-1] => Mz = Ax - b
    M = build_augmented_matrix(A, b)

    x = np.zeros(n, dtype=np.float64)
    z0 = np.concatenate([x, np.array([-1.0], dtype=np.float64)])

    mv = MatVecSolver(xvec=z0, mat=M)
    mv.initializeMat()  # Root only

    # NonRoot ranks: enter service loop once; block until Root finalizes
    if not isinstance(mv.solverObject, Root):
        mv.matVec(correction=False)
        return

    root = mv.solverObject
    rng = np.random.default_rng(args.seed)

    row_norm2 = np.sum(A * A, axis=1)
    row_norm2[row_norm2 == 0.0] = 1.0  # avoid divide-by-zero

    res_norm = float("inf")
    k_done = 0

    for k in range(args.iters):
        k_done = k + 1
        i = (k % m) if args.row == "cyclic" else int(rng.integers(0, m))

        z = np.concatenate([x, np.array([-1.0], dtype=np.float64)])
        mv.xvec = z
        mv.initializeVec()          # Root scales z internally (stores x_min/x_max(ptp)/x_sum)
        mv.matVec(correction=False) # returns y_scaled = M_scaled @ z_scaled

        y_scaled = np.asarray(root.y_mem_result, dtype=np.float64).reshape(-1)

        # Correct back to ORIGINAL domain: r = Mz = Ax - b
        r = correction_vectorized(root, y_scaled)

        res_norm = float(np.linalg.norm(r, 2))
        print(f"[k={k:04d}] ||Ax-b||_2 = {res_norm:.3e}")

        if args.check:
            r_ref = A @ x - b
            rel = np.linalg.norm(r - r_ref) / (np.linalg.norm(r_ref) + 1e-12)
            print(f"         corr_check_rel = {rel:.3e}")

        if res_norm <= args.tol:
            break

        # Classical Kaczmarz update: x <- x - omega * (r_i / ||a_i||^2) * a_i
        x = x - args.omega * (r[i] / row_norm2[i]) * A[i, :]

    mv.finalize()

    print("[INFO] Kaczmarz run complete.")
    print(f"Final residual norm: {res_norm:.3e} after {k_done} iterations")
    print(f"Final solution x: {x}")

    if args.save:
        np.save(output_path(f"{args.save_prefix}.npy"), x)
        np.savetxt(output_path(f"{args.save_prefix}.txt"), x, delimiter=",")
        print("Saved:", output_path(f"{args.save_prefix}.npy"))


if __name__ == "__main__":
    main()


