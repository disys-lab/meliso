# `NonRootMCA`

**Module:** `src.core.NonRootMCA`
**File:** `src/core/NonRootMCA.py`
**Inherits:** [`BaseMCA`](BaseMCA.md)

---

## Overview

`NonRootMCA` implements the worker-process side of distributed MVM.  Each instance lives on a distinct MPI rank (0 … `size-2`) and manages one physical (or simulated) memristor crossbar array (MCA).

Responsibilities:

- Receiving a matrix sub-block from the root process via MPI.
- Initializing the C++ NeuroSim backend through the [`MelisoPy`](../../cython/MelisoPy.md) Cython wrapper.
- Loading device-specific parameters (conductance, write voltage, device variation) from a YAML file.
- Encoding matrix weights onto the hardware using an iterative write-and-verify loop.
- Receiving an input vector slice and running the local MVM on hardware.
- Optionally applying a multi-step error-correction algorithm.
- Sending the result back to the root process.

---

## Class Definition

```python
class NonRootMCA(BaseMCA):
    def __init__(self, comm, HW=-1, SC=-1, set_mat=True): ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comm` | `mpi4py.MPI.Comm` | — | MPI communicator |
| `HW` | `int` | `-1` | Hardware on/off override (`-1` → read from YAML) |
| `SC` | `int` | `-1` | Scaling on/off override (`-1` → read from YAML) |
| `set_mat` | `bool` | `True` | If `True`, immediately calls `setMat()` to receive the sub-matrix |

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `A` | `np.ndarray` | Local matrix sub-block `(cellRows, cellCols)` |
| `localx` | `np.ndarray` | Local input vector slice |
| `X` | `np.ndarray` | Tiled version of the input vector (shape `locRows × locCols`) |
| `y` | `np.ndarray` | Local output vector |
| `locRows` | `int` | Rows of the local matrix chunk |
| `locCols` | `int` | Columns of the local matrix chunk |
| `device_type` | `int` | NeuroSim device type (`DT` env var or default `1`) |
| `meliso_obj` | `MelisoPy` | Cython wrapper for the C++ Meliso simulator |
| `device_config` | `dict` | Parsed device YAML parameters |
| `ITER_LIMIT` | `int` | Max iterations for the write-and-verify loop (env `ITER_LIMIT`, default `21`) |
| `OVERRIDE` | `int` | Force iteration limit without early-stop check (env `OVERRIDE`, default `0`) |
| `PRECISION` | `float` | Convergence threshold for weight encoding (`1e-6`) |
| `RESIDUALS_TOL` | `float` | `PRECISION²` — stop threshold on residual change |
| `MAX_TOL` | `float` | Upper conductance normalisation bound (`1.0`) |
| `MIN_TOL` | `float` | Lower conductance normalisation bound (`0.0`) |
| `interpolants` | `int` | Number of interpolation points for NeuroSim (default `3`) |
| `turnOnHardware` | `int` | Hardware simulation flag |
| `turnOnScaling` | `int` | Scaling simulation flag |
| `ERR_CORR` | `int` | Error correction flag (inherited, may be overridden by `EC` env var) |
| `mcaStats` | `np.ndarray` | Shape `(8, 1)` — local hardware statistics |

---

## Methods

### `setMat()`

Receives the local matrix chunk and initialises the MCA weights.

```python
def setMat(self) -> None
```

Calls `acquireLocalA()` then `initializeMCA()`.

---

### `acquireLocalA()`

Receives the matrix sub-block from the root process.

```python
def acquireLocalA(self) -> None
```

- If `useMPI4MatDist` is `True` (default): receives via `comm.Recv` from `ROOT_PROCESS_RANK`.
- Otherwise: loads from the pre-written `.npy` file in the decomposition directory.

Sets `self.A`, `self.locRows`, `self.locCols`.

---

### `acquireLocalX(x)`

Tiles the 1-D input vector slice `x` into a 2-D matrix `self.X` of shape `(locRows, locCols)`.

```python
def acquireLocalX(self, x: np.ndarray) -> None
```

---

### `initializeMCA()`

Resets all hardware weights to the minimum conductance and then encodes the current sub-block `A`.

```python
def initializeMCA(self) -> None
```

Calls `meliso_obj.initializeWeights()` followed by `setWeightsIncremental(self.A)`.

---

### `setWeights(A)`

Directly programs matrix `A` onto the hardware (no error checking).

```python
def setWeights(self, A: np.ndarray) -> None
```

---

### `setWeightsIncremental(A)`

Iteratively encodes `A` with a write-and-verify loop.

```python
def setWeightsIncremental(self, A: np.ndarray) -> tuple[int, float]
```

**Algorithm:**

```
j = 0
while j < ITER_LIMIT:
    meliso_obj.setWeightsIncremental(A, PRECISION)
    actualWeights = meliso_obj.getWeights()
    current_residuals = ||actualWeights - A||
    if OVERRIDE == 0:
        if |residuals - current_residuals| < RESIDUALS_TOL and j > 0:
            break  # converged
    residuals = current_residuals
    j += 1
```

**Returns:** `(iterations, final_residuals)`

---

### `getDeviceConfig()`

Looks up the device YAML config file for this rank using the `assignment` map in the experiment config.

```python
def getDeviceConfig(self) -> None
```

Assignment rule: `-1` means all ranks use this config. Otherwise, a list of rank-range pairs is checked.

---

### `readDeviceConfigFile(deviceConfigFile)`

Opens `<device_config.root>/<deviceConfigFile>` and populates `self.device_config`.

```python
def readDeviceConfigFile(self, deviceConfigFile: str) -> None
```

---

### `setConductanceProperties()`

Reads conductance parameters from `device_config["ConductanceProperties"]` and calls `meliso_obj.setConductanceProperties(...)`.

```python
def setConductanceProperties(self) -> None
```

| Device YAML key | Description |
|-----------------|-------------|
| `maxConductance` | Maximum cell conductance |
| `minConductance` | Minimum cell conductance |
| `avgMaxConductance` | Average maximum conductance |
| `avgMinConductance` | Average minimum conductance |
| `conductance` | Initial conductance |
| `conductancePrev` | Previous conductance |

---

### `setWriteProperties()`

Reads write parameters from `device_config["WriteProperties"]` and calls `meliso_obj.setWriteProperties(...)`.

```python
def setWriteProperties(self) -> None
```

| Device YAML key | Description |
|-----------------|-------------|
| `writeVoltageLTP` | Write voltage for LTP (potentiation) |
| `writeVoltageLTD` | Write voltage for LTD (depression) |
| `writePulseWidthLTP` | Pulse width for LTP |
| `writePulseWidthLTD` | Pulse width for LTD |
| `maxNumLevelLTP` | Max discrete levels for LTP |
| `maxNumLevelLTD` | Max discrete levels for LTD |

---

### `setDeviceVariation()`

Reads variation parameters from `device_config["DeviceVariation"]` and calls `meliso_obj.setDeviceVariation(...)`.

```python
def setDeviceVariation(self) -> None
```

| Device YAML key | Description |
|-----------------|-------------|
| `NL_LTP` | Non-linearity for LTP |
| `NL_LTD` | Non-linearity for LTD |
| `sigmaDtoD` | Device-to-device variation |
| `sigmaCtoC` | Cycle-to-cycle variation |

---

### `localMatVec(x)`

Runs a single MVM on the hardware for input vector `x`.

```python
def localMatVec(self, x: np.ndarray) -> np.ndarray
```

Steps:

1. `meliso_obj.loadInput(x)`
2. `meliso_obj.matVec()`
3. `y = meliso_obj.getResults()`

**Returns:** Output vector `y` of shape `(locRows, 1)`.

---

### `parallelMatVec()`

Executes the full worker-side MVM protocol and sends results to the root.

```python
def parallelMatVec(self) -> np.ndarray
```

**Flow:**

- If `ERR_CORR == 1`: calls `errorCorrection()` and times it.
- Else: receives `x` from root, calls `setWeights(A)`, then `localMatVec(x)`.
- Sends `self.y` to `ROOT_PROCESS_RANK`.

---

### `errorCorrection()`

Applies the MELISO+ multi-step error-correction algorithm.

```python
def errorCorrection(self) -> np.ndarray
```

**Algorithm summary:**

1. Receive `x` from root; tile into `X`.
2. Encode `X_tilde` (tiled vector on hardware), get `U_tilde = A @ X_tilde` row-by-row.
3. Compute `V_tilde_x` similarly.
4. Re-encode `A_tilde`; compute `V_tilde_a = A_tilde @ X_tilde` row-by-row.
5. Compute direct estimate `y_tilde = A_hardware @ x`, then denoise.
6. Correct:
   ```
   y_a[i] = y_tilde[i] - V_tilde_a[i] + U_tilde[i]
   y_x[i] = U_tilde[i] - V_tilde_x[i] + y_tilde[i]
   ```
7. Apply denoising again: `y_corr = denoiseLeastSquare(y_a + y_x)`.

**Returns:** Corrected output vector `y_corr`.

---

### `denoiseLeastSquare(w, l_dn=1e-12)`

Applies a regularised least-squares denoising filter.

```python
def denoiseLeastSquare(self, w: np.ndarray, l_dn: float = 1e-12) -> np.ndarray
```

Solves:

```
y = (I + l_dn * L^T L)^{-1} w
```

where `L` is a first-difference matrix (tridiagonal with `1` on diagonal, `-1` on super-diagonal).

---

### `getMCAStats()`

Collects hardware statistics from `meliso_obj` and gathers them at the root via `comm.Gather`.

```python
def getMCAStats(self) -> None
```

---

## Device Configuration YAML Format

```yaml
Device:
  ConductanceProperties:
    maxConductance: 3.0e-6
    minConductance: 0.5e-6
    avgMaxConductance: maxConductance
    avgMinConductance: minConductance

  WriteProperties:
    writeVoltageLTP: 2.0
    writeVoltageLTD: 2.0
    writePulseWidthLTP: 1.0e-9
    writePulseWidthLTD: 1.0e-9
    maxNumLevelLTP: 15.0
    maxNumLevelLTD: 15.0

  DeviceVariation:
    NL_LTP: 2.4
    NL_LTD: -4.09
    sigmaDtoD: 0.0
    sigmaCtoC: 0.0
```

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `DT` | Device type (`0`–`6`). Default `1`. |
| `ITER_LIMIT` | Max write-and-verify iterations. Default `21`. |
| `OVERRIDE` | `1` = always run `ITER_LIMIT` iterations; `0` = early-stop. |
| `EC` | Override error correction flag. |

---

## See Also

- [`BaseMCA`](BaseMCA.md) — Parent class.
- [`RootMCA`](RootMCA.md) — Counterpart coordinator.
- [`MelisoPy`](../../cython/MelisoPy.md) — Cython wrapper used by `NonRootMCA`.
- [`NonRoot`](../../solver/matvec/NonRoot.md) — Higher-level wrapper around `NonRootMCA`.
