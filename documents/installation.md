# Installation

This page covers everything you need to build and run MELISO+ from scratch.
Two paths are provided: **native Linux** (recommended for HPC/clusters) and **Docker** (recommended for local development on any OS).

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| OS | Linux (Ubuntu 20.04+) | macOS is untested; Windows requires Docker |
| CPU | Any x86-64 | Multi-core recommended for MPI |
| RAM | 8 GB | More required for large matrices |
| Disk | 2 GB free | For NeuroSim source + build artifacts |

### Required System Libraries (Linux)

These must be present before building. Install on Ubuntu/Debian with:

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    openmpi-bin \
    libopenmpi-dev \
    git
```

| Library | Purpose |
|---------|---------|
| `gcc` / `g++` (7.5+) | Compiles the NeuroSim C++ backend |
| OpenMP (`libgomp`) | Parallelism inside the C++ simulator |
| OpenMPI | Required by `mpi4py` for distributed execution |
| `git` | Cloning the NeuroSim backend during build |

### Required Python Packages

| Package | Purpose |
|---------|---------|
| `python` 3.8+ | Core language |
| `cython` | Compiles `src/cython/mel.pyx` |
| `mpi4py` | Python MPI bindings |
| `numpy` | Matrix/vector operations |
| `scipy` | Sparse matrix loading (`.mtx` / `.npz`) |
| `pyyaml` | Parsing experiment and device config files |

---

## Option A: Native Linux Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/disys-lab/meliso.git
cd meliso
```

### Step 2 — Create a Conda Environment

The shell scripts in this repository use a conda environment named `mpienv38`.
You can recreate it manually:

```bash
conda create -n mpienv38 python=3.8 -y
conda activate mpienv38
pip install cython mpi4py numpy scipy pyyaml matplotlib
```

> **Note:** Install `mpi4py` via `pip` (not `conda`) to ensure it links against the system OpenMPI installation. Installing via conda may create an incompatible bundled MPI.

### Step 3 — Set Environment Variables

These two variables must be set in every shell session where you run MELISO+.
Add them to your `~/.bashrc` or `~/.bash_profile` to make them permanent:

```bash
export PYTHONPATH="${PYTHONPATH}:./build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:./build/"
```

Or set them inline for a single session:

```bash
cd meliso
export PYTHONPATH="${PYTHONPATH}:$(pwd)/build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/build/"
```

### Step 4 — Build

`make all` performs four steps automatically:

| Step | Make Target | What It Does |
|------|------------|--------------|
| 1 | `get-immv` | Clones the [InMemMVM](https://github.com/disys-lab/InMemMVM) NeuroSim backend into a temp directory and copies its source files into `src/mlp_neurosim/` |
| 2 | `create-build` | Creates the `build/` directory tree |
| 3 | `neurosim` | Compiles all NeuroSim `.cpp` files into a shared library `build/libmlp.so` |
| 4 | `meliso` | Compiles `src/cython/mel.pyx` into `build/meliso.*.so` via Cython + `setup.py` |

```bash
make all
```

Expected output (truncated):

```
Cloning or updating InMemMVM...
Copying files into src/mlp_neurosim/...
Done.
g++ -static-libstdc++ -fopenmp -O3 -std=c++0x -w -fPIC ...
...
running build_ext
...
```

> **Rebuilding after changes:** If you modify the C++ backend, run:
> ```bash
> make clean
> make all
> ```

### Step 5 — Verify the Build

```bash
ls build/
```

You should see at least:

```
libmlp.so
meliso.cpython-38-x86_64-linux-gnu.so   # (name varies by Python version)
```

Run a quick smoke test (2 MPI processes = 1 worker + 1 root):

```bash
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
XVEC_PATH=inputs/vectors/input_x.txt \
  mpiexec -n 2 python3 DistributedMatVec.py
```

If the build is correct, you will see output lines like:

```
INFO: Device type set to 1 on rank 0
[INFO]ROOT: begin virtualParallelMatVec at MCA 0,0
Computing MVM at Device Rank 0...
[INFO] MELISO+ Result (without normalization reversal): ...
```

---

## Option B: Docker Installation

Docker is the fastest way to get a working environment on macOS, Windows, or any Linux system without manually installing system dependencies.

### Step 1 — Install Docker

Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your OS.

### Step 2 — Clone the Repository

```bash
git clone https://github.com/disys-lab/meliso.git
cd meliso
```

### Step 3 — Build the Docker Image

The `Dockerfile` at the root of the repository installs all system dependencies, Python packages, and builds the NeuroSim backend automatically.

```bash
docker build -t meliso:latest .
```

This may take several minutes on first run.

Alternatively, pull a pre-built image from DockerHub:

```bash
docker pull pramanan3/meliso:v0.0.3
```

### Step 4 — Run the Container

Mount your local repository directory into the container so that your config files and input data are accessible:

```bash
docker run -v "$(pwd)":/meliso_edit \
           --name meliso \
           -d meliso:latest \
           /bin/sh -c "tail -f /dev/null"
```

### Step 5 — Open a Shell Inside the Container

```bash
docker exec -it meliso bash
```

You are now inside the container at `/meliso`, with the build already complete and the environment variables already set.

### Step 6 — Run an Experiment Inside the Container

The `--allow-run-as-root` and `--oversubscribe` flags are required inside Docker:

```bash
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
XVEC_PATH=inputs/vectors/input_x.txt \
  mpiexec --allow-run-as-root --oversubscribe \
  -n 2 python3 DistributedMatVec.py
```

---

## Makefile Reference

| Target | Command | Description |
|--------|---------|-------------|
| `all` | `make all` | Full build: get-immv + create-build + neurosim + meliso |
| `get-immv` | `make get-immv` | Clone/update NeuroSim backend from InMemMVM repo |
| `create-build` | `make create-build` | Create `build/` directory structure |
| `neurosim` | `make neurosim` | Compile NeuroSim C++ into `build/libmlp.so` |
| `meliso` | `make meliso` | Compile Cython extension into `build/meliso.*.so` |
| `clean` | `make clean` | Remove entire `build/` directory |
| `clean-neurosim` | `make clean-neurosim` | Remove only NeuroSim compiled objects |
| `clean-meliso` | `make clean-meliso` | Remove only the Cython compiled extension |
| `clean-immv` | `make clean-immv` | Remove the temporary InMemMVM clone directory |
| `delete-immv` | `make delete-immv` | Remove the `src/mlp_neurosim/` source directory |

---

## Common Build Errors

### `make: g++: Command not found`
Install `build-essential`:
```bash
sudo apt-get install build-essential
```

### `mpi4py` import fails / `libmpi.so` not found
Ensure OpenMPI is installed and that `mpi4py` was installed via `pip`, not `conda`:
```bash
pip install --force-reinstall mpi4py
```

### `ModuleNotFoundError: No module named 'meliso'`
The `build/` directory is not on your Python path. Ensure the following is set:
```bash
export PYTHONPATH="${PYTHONPATH}:./build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:./build/"
```

### `ExperimentConfigFileError: MCA grid size != mpi processes`
The number of MPI processes must equal `mca_rows × mca_cols + 1`.
See the [Configuration Reference](configuration.md) for details.

### Cython compilation fails with `fatal error: Meliso.h: No such file or directory`
The NeuroSim backend has not been fetched yet. Run:
```bash
make get-immv
make meliso
```
