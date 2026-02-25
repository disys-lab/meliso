# `MelisoPy`

**Module:** `meliso` (compiled Cython extension)
**File:** `src/cython/mel.pyx`
**Language:** Cython (`.pyx`)

---

## Overview

`MelisoPy` is a Cython extension type that wraps the C++ `meliso::Meliso` class (declared in [`Meliso.pxd`](Meliso_interface.md)).
It is the boundary between Python/NumPy and the NeuroSim hardware simulator.

The class manages:

- Memory allocation of raw C arrays for the matrix (`A_matrix`), input vector (`x`), and output vector (`y`).
- Construction of the C++ `Meliso` object with device and grid parameters.
- Forwarding all Python calls to the corresponding C++ methods.
- Converting NumPy arrays to/from C pointer arrays for hardware communication.

After the Cython module is compiled and placed in `build/`, it is imported as:

```python
import meliso
obj = meliso.MelisoPy(device_type, rows, columns, MAX_TOL, TOL, turnOnHardware, turnOnScaling)
```

---

## Class Definition

```cython
cdef class MelisoPy:
    cdef Meliso melisoObj
    cdef int m, n
    cdef double TOL, MAX_TOL
    cdef int device_type
    cdef double *A_matrix
    cdef double *x
    cdef double *y
```

---

## Constructor

```python
MelisoPy(device_type, rows, columns, MAX_TOL, TOL, turnOnHardware, turnOnScaling)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `device_type` | `int` | NeuroSim device model (0–6; see [device type table](../../index.md#supported-device-types)) |
| `rows` | `int` | Number of rows in the weight matrix (= `cellRows`) |
| `columns` | `int` | Number of columns in the weight matrix (= `cellCols`) |
| `MAX_TOL` | `float` | Upper conductance normalisation bound (typically `1.0`) |
| `TOL` | `float` | Lower conductance normalisation bound (typically `0.0`) |
| `turnOnHardware` | `int` | `1` = enable hardware simulation; `0` = bypass |
| `turnOnScaling` | `int` | `1` = enable internal scaling; `0` = disable |

**Side effects:** Allocates `m*n*sizeof(double)` bytes for `A_matrix`, `n*sizeof(double)` for `x`, and `m*sizeof(double)` for `y` via C `malloc`.

---

## Methods

### `setHardwareOn(turnOnHardware)`

Enable or disable hardware simulation at runtime.

```python
def setHardwareOn(self, turnOnHardware: int) -> None
```

---

### `setScalingOn(turnOnScaling)`

Enable or disable internal conductance scaling at runtime.

```python
def setScalingOn(self, turnOnScaling: int) -> None
```

---

### `setInterpolants(interpolants)`

Set the number of interpolation points used by NeuroSim for device modelling.

```python
def setInterpolants(self, interpolants: int) -> None
```

---

### `initializeWeights()`

Resets all hardware weights to the minimum conductance state.

```python
def initializeWeights(self) -> None
```

Must be called before encoding a new weight matrix.

---

### `getWeights()`

Reads the currently encoded weights from the hardware and returns them as a NumPy array.

```python
def getWeights(self) -> np.ndarray  # shape (m, n), dtype float64
```

Internally copies `melisoObj.actualWeights[i*n + j]` into a 2-D array.

---

### `setWeightsIncremental(np_A_matrix, precision)`

Performs one step of the iterative write-and-verify encoding.

```python
def setWeightsIncremental(self, np_A_matrix: np.ndarray, precision: float) -> None
```

| Parameter | Description |
|-----------|-------------|
| `np_A_matrix` | Target weight matrix, shape `(m, n)` |
| `precision` | Convergence tolerance passed to C++ |

Copies the NumPy matrix into the C array `A_matrix` (row-major), then calls `melisoObj.setWeightsIncremental`.

---

### `setWeights(np_A_matrix)`

Directly programs the full weight matrix (single-shot, no verification).

```python
def setWeights(self, np_A_matrix: np.ndarray) -> None
```

---

### `loadInput(np_x)`

Loads a 1-D input vector into the hardware input register.

```python
def loadInput(self, np_x: np.ndarray) -> None
```

Copies `np_x[j]` into `self.x[j]` then calls `melisoObj.loadInput(self.x)`.

---

### `matVec()`

Triggers the matrix-vector multiplication on the hardware (or simulator).

```python
def matVec(self) -> None
```

Results are stored internally in `melisoObj.y`; retrieve them with `getResults()`.

---

### `getResults()`

Reads the output vector from the hardware and returns it as a NumPy array.

```python
def getResults(self) -> np.ndarray  # shape (m, 1), dtype float64
```

---

### `setConductanceProperties(maxConductance, minConductance, avgMaxConductance, avgMinConductance, conductance, conductancePrev)`

Configure the six conductance parameters.

```python
def setConductanceProperties(
    self,
    maxConductance: float,
    minConductance: float,
    avgMaxConductance: float,
    avgMinConductance: float,
    conductance: float,
    conductancePrev: float
) -> None
```

---

### `getConductanceProperties(x, y)`

Retrieve the six conductance properties as a NumPy array.

```python
def getConductanceProperties(self, x: int, y: int) -> np.ndarray  # shape (6, 1)
```

---

### `setWriteProperties(writeVoltageLTP, writeVoltageLTD, writePulseWidthLTP, writePulseWidthLTD, maxNumLevelLTP, maxNumLevelLTD)`

Configure six write parameters for LTP and LTD pulses.

```python
def setWriteProperties(
    self,
    writeVoltageLTP: float,
    writeVoltageLTD: float,
    writePulseWidthLTP: float,
    writePulseWidthLTD: float,
    maxNumLevelLTP: float,
    maxNumLevelLTD: float
) -> None
```

---

### `getWriteProperties(x, y)`

Retrieve the six write properties as a NumPy array.

```python
def getWriteProperties(self, x: int, y: int) -> np.ndarray  # shape (6, 1)
```

---

### `setDeviceVariation(NL_LTP, NL_LTD, sigmaDtoD, sigmaCtoC)`

Configure device variation parameters.

```python
def setDeviceVariation(
    self,
    NL_LTP: float,
    NL_LTD: float,
    sigmaDtoD: float,
    sigmaCtoC: float
) -> None
```

| Parameter | Description |
|-----------|-------------|
| `NL_LTP` | Non-linearity for long-term potentiation |
| `NL_LTD` | Non-linearity for long-term depression |
| `sigmaDtoD` | Device-to-device conductance variation (σ) |
| `sigmaCtoC` | Cycle-to-cycle conductance variation (σ) |

---

### `getDeviceVariation(x, y)`

Retrieve the four variation parameters as a NumPy array.

```python
def getDeviceVariation(self, x: int, y: int) -> np.ndarray  # shape (4, 1)
```

---

### `getMCAStats(num_mca_stats)`

Reads `num_mca_stats` hardware performance statistics from `melisoObj.mcaStats`.

```python
def getMCAStats(self, num_mca_stats: int) -> np.ndarray  # shape (num_mca_stats, 1)
```

Standard usage passes `num_mca_stats = 8`. See [`RootMCA.getMCAStats()`](../core/RootMCA.md#getmcastats) for the index-to-statistic mapping.

---

## Memory Management

The three C arrays (`A_matrix`, `x`, `y`) are allocated in `__cinit__` via `malloc` and are **not freed** by a `__dealloc__`. This is safe for the lifetime of a typical experiment run but should be noted if `MelisoPy` objects are created and destroyed in large numbers.

---

## Example Usage

```python
import meliso

# Create a 66×66 RealDevice MCA with hardware simulation on
mca = meliso.MelisoPy(1, 66, 66, 1.0, 0.0, 1, 0)

# Configure device properties (required for device_type == 1)
mca.setConductanceProperties(3e-6, 0.5e-6, 3e-6, 0.5e-6, 0.5e-6, 0.5e-6)
mca.setWriteProperties(2.0, 2.0, 1e-9, 1e-9, 15.0, 15.0)
mca.setDeviceVariation(2.4, -4.09, 0.0, 0.0)

# Encode a weight matrix
import numpy as np
A = np.random.rand(66, 66)
mca.initializeWeights()
mca.setWeightsIncremental(A, 1e-6)

# Run MVM
x = np.random.rand(66)
mca.loadInput(x)
mca.matVec()
y = mca.getResults()  # shape (66, 1)
```

---

## See Also

- [`Meliso_interface`](Meliso_interface.md) — C++ class declaration (`.pxd`).
- [`NonRootMCA`](../core/NonRootMCA.md) — Python class that owns and drives `MelisoPy`.
