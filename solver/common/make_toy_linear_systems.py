import os
import numpy as np

def save_case(folder, name, A, b, xstar):
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, f"{name}_A.npy"), A)
    np.save(os.path.join(folder, f"{name}_b.npy"), b)
    np.save(os.path.join(folder, f"{name}_xstar.npy"), xstar)
    print(f"Saved {name}: A {A.shape}, b {b.shape}, x* {xstar.shape}")

def main():
    out = "toy_cases"

    # Case 1
    A1 = np.array([[ 2.0, -1.0],
                   [-3.0,  4.0]])
    x1 = np.array([1.5, -2.0])
    b1 = A1 @ x1

    # Case 2
    A2 = np.array([[ 1.0, -2.0,  0.5],
                   [ 0.0,  3.0, -1.0],
                   [-4.0,  1.0,  2.0]])
    x2 = np.array([-1.0, 0.5, 2.0])
    b2 = A2 @ x2

    # Case 3 (overdetermined consistent)
    A3 = np.array([[ 1.0,   2.0],
                   [-2.0,   1.0],
                   [ 0.5,  -1.0],
                   [ 3.0,  -0.5],
                   [-1.5,  -2.0]])
    x3 = np.array([0.75, -1.25])
    b3 = A3 @ x3

    save_case(out, "case1", A1, b1, x1)
    save_case(out, "case2", A2, b2, x2)
    save_case(out, "case3", A3, b3, x3)

    # Also print for quick eyeballing
    for name, A, b, x in [("case1", A1, b1, x1), ("case2", A2, b2, x2), ("case3", A3, b3, x3)]:
        if A.shape[0] == A.shape[1]:
            x_np = np.linalg.solve(A, b)
        else:
            x_np = np.linalg.lstsq(A, b, rcond=None)[0]
        print(f"{name}: x* = {x}, numpy = {x_np}, err = {np.linalg.norm(x_np-x)}")

if __name__ == "__main__":
    main()
