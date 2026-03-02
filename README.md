# MELISO+: Memristor Linear Solver for Optimization

**Full documentation:** [https://disys-lab.github.io/meliso/](https://disys-lab.github.io/meliso/)

MELISO+ is a distributed matrix-vector multiplication (MVM) framework for memristive crossbar array (MCA) hardware, built on MPI and backed by the NeuroSim C++ simulator.

---

## Quick Links

| | |
|---|---|
| Installation | [docs/installation](https://disys-lab.github.io/meliso/installation/) |
| Configuration reference | [docs/configuration](https://disys-lab.github.io/meliso/configuration/) |
| Quick-start tutorial | [docs/tutorials/quickstart_local](https://disys-lab.github.io/meliso/tutorials/quickstart_local/) |
| Distributed MVM on SLURM | [docs/tutorials/run_distributed_MVM](https://disys-lab.github.io/meliso/tutorials/run_distributed_MVM/) |
| API reference | [docs/api](https://disys-lab.github.io/meliso/api/) |

---

## Repository Layout

| Path | Contents |
|------|----------|
| `src/mlp_neurosim/` | NeuroSim C++ simulator |
| `src/cython/` | Python/Cython interface to NeuroSim |
| `src/core/` | MPI coordinator classes (`RootMCA`, `NonRootMCA`, `BaseMCA`) |
| `solver/matvec/` | High-level MVM interface (`MatVecSolver`, `Root`, `NonRoot`) |
| `solver/mlp/` | MLP inference helper (`MLP`) |
| `solver/common/` | Matrix loaders and test-system generators |
| `config_files/` | Experiment and device YAML configuration files |
| `documents/` | MkDocs documentation source |
| `DistributedMatVec.py` | Driver: standalone distributed MVM |
| `mlpInference.py` | Driver: MLP inference accelerated by MELISO+ |
| `pdhg.py` | Driver: PDHG optimization accelerated by MELISO+ |

---

## Getting Started

```bash
git clone https://github.com/disys-lab/meliso.git
cd meliso
mkdir build
export PYTHON_PATH=$PYTHON_PATH:./build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/
make all
```

For detailed prerequisites, Docker instructions, and troubleshooting see the [Installation guide](https://disys-lab.github.io/meliso/installation/).

To run a minimal MVM experiment:

```bash
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
  mpiexec -n 2 python3 DistributedMatVec.py
```

See the [Quick-start tutorial](https://disys-lab.github.io/meliso/tutorials/quickstart_local/) for a step-by-step walkthrough.
