# C++ Interface: `meliso::Meliso`

**File:** `src/cython/Meliso.pxd`
**Language:** Cython declaration file (`.pxd`)

---

## Overview

`Meliso.pxd` is the Cython declaration file that exposes the C++ `meliso::Meliso` class to Cython code.
It does **not** contain any Python-callable logic; it only defines the C++ interface so that `mel.pyx` can call the C++ class directly.

The actual C++ implementation lives in:

- `src/cython/Meliso.h` — class header
- `src/cython/Meliso.cpp` — implementation (linked via `cdef extern from "Meliso.cpp"`)
- `src/cython/NewtonDividedDifference.cpp` — numerical helper for device interpolation

---

## Cython Declaration

```cython
cdef extern from "Meliso.h" namespace "meliso":
    cdef cppclass Meliso:

        # ── Scalar members ─────────────────────────────────────────────────────
        int device_type

        double totalSubArrayArea
        double totalNeuronAreaIH
        double heightNeuronIH
        double widthNeuronIH
        double leakageNeuronIH

        # ── Pointer members (C arrays) ──────────────────────────────────────────
        double *y                    # Output vector (length m)
        double *actualWeights        # Encoded weight matrix (length m*n, row-major)
        double *delta                # Weight delta buffer
        double *y_min                # Minimum output values

        double *conductanceProperties  # 6 conductance parameters
        double *writeProperties        # 6 write parameters
        double *deviceVariation        # 4 variation parameters
        double *mcaStats               # 8 performance statistics

        int *sign                      # Sign buffer

        # ── Methods ─────────────────────────────────────────────────────────────
        void setHardwareOn(int)
        void setScalingOn(int)
        void setInterpolants(int)
        void loadInput(double *)
        void initializeWeights()
        void setWeights(double *)
        void setWeightsIncremental(double *, double)
        void getWeights()
        void matVec()
        void getResults()
        void setConductanceProperties(double, double, double, double, double, double)
        void getConductanceProperties(int, int)
        void setWriteProperties(double, double, double, double, double, double)
        void getWriteProperties(int, int)
        void setDeviceVariation(double, double, double, double)
        void getDeviceVariation(int, int)
```

---

## C++ Constructors

```cpp
// Declared in Meliso.pxd
Meliso()
Meliso(int device_type, int rows, int cols, double MAX_TOL, double TOL,
       int turnOnHardware, int turnOnScaling)
```

Both constructors are declared with `except +` in Cython, meaning C++ exceptions are translated to Python exceptions.

---

## Member Reference

### Scalar Members

| Member | Type | Description |
|--------|------|-------------|
| `device_type` | `int` | Active device model (0–6) |
| `totalSubArrayArea` | `double` | Total sub-array area [µm²] |
| `totalNeuronAreaIH` | `double` | Total neuron area in the IH layer [µm²] |
| `heightNeuronIH` | `double` | Height of the IH neuron circuit [µm] |
| `widthNeuronIH` | `double` | Width of the IH neuron circuit [µm] |
| `leakageNeuronIH` | `double` | Leakage power of IH neurons [W] |

### Pointer Members

| Member | Length | Description |
|--------|--------|-------------|
| `y` | `m` | Output vector after `getResults()` |
| `actualWeights` | `m * n` | Row-major encoded weight matrix after `getWeights()` |
| `delta` | `m * n` | Weight error delta buffer |
| `y_min` | `m` | Element-wise minimum outputs |
| `conductanceProperties` | `6` | `[maxG, minG, avgMaxG, avgMinG, G, G_prev]` |
| `writeProperties` | `6` | `[V_LTP, V_LTD, PW_LTP, PW_LTD, levels_LTP, levels_LTD]` |
| `deviceVariation` | `4` | `[NL_LTP, NL_LTD, sigmaDtoD, sigmaCtoC]` |
| `mcaStats` | `8` | Performance statistics (see below) |
| `sign` | — | Sign encoding buffer |

### `mcaStats` Index Map

| Index | Statistic |
|-------|-----------|
| 0 | `totalSubArrayArea` |
| 1 | `totalNeuronAreaIH` |
| 2 | `subArrayIH→leakage` |
| 3 | `leakageNeuronIH` |
| 4 | `subArrayIH→writeLatency` [s] |
| 5 | `arrayIH→writeEnergy + subArrayIH→writeDynamicEnergy` [J] |
| 6 | `subArrayIH→readLatency` [s] |
| 7 | `arrayIH→readEnergy + subArrayIH→readDynamicEnergy` [J] |

---

## Method Reference

| Method | C++ Signature | Description |
|--------|---------------|-------------|
| `setHardwareOn` | `void(int)` | Toggle hardware simulation |
| `setScalingOn` | `void(int)` | Toggle internal conductance scaling |
| `setInterpolants` | `void(int)` | Set number of interpolation points |
| `loadInput` | `void(double*)` | Load input vector into hardware |
| `initializeWeights` | `void()` | Reset all cells to min conductance |
| `setWeights` | `void(double*)` | Program weights (no verification) |
| `setWeightsIncremental` | `void(double*, double)` | One write-and-verify iteration |
| `getWeights` | `void()` | Copy weights into `actualWeights` |
| `matVec` | `void()` | Execute hardware MVM |
| `getResults` | `void()` | Copy output into `y` |
| `setConductanceProperties` | `void(double×6)` | Set conductance parameters |
| `getConductanceProperties` | `void(int,int)` | Retrieve into `conductanceProperties` |
| `setWriteProperties` | `void(double×6)` | Set write parameters |
| `getWriteProperties` | `void(int,int)` | Retrieve into `writeProperties` |
| `setDeviceVariation` | `void(double×4)` | Set variation parameters |
| `getDeviceVariation` | `void(int,int)` | Retrieve into `deviceVariation` |

---

## Build Integration

The `.pxd` file is used exclusively at Cython compile time.
The `setup.py` at the project root compiles `mel.pyx` into a shared library (`meliso.*.so`) placed in `build/`, linking against `libmlp.so` (the compiled NeuroSim library).

```python
# setup.py (excerpt)
Extension(
    "build.meliso",
    sources=["src/cython/mel.pyx"],
    include_dirs=["src/mlp_neurosim/", "src/cython/"],
    library_dirs=["build/"],
    libraries=["mlp"],
    extra_compile_args=["-fopenmp", "-O3", "-std=c++0x", "-fPIC"],
    language="c++",
)
```

---

## See Also

- [`MelisoPy`](MelisoPy.md) — Python-facing Cython class that uses this declaration.
- [`NonRootMCA`](../core/NonRootMCA.md) — Creates and drives `MelisoPy` instances.
