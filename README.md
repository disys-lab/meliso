# MELISO: Memristor Linear Solver for Optimization

## File lists
1. Neurosim files are inside `src/mlp_neurosim`
2. Python interfacing files (Cython) are in `src/cython`
3. Nonlinearity-to-A table: `Documents/Nonlinearity-NormA.htm`
4. NeuroSim Manual: `Documents/Manual.pdf`

## Installation steps (Linux)
1. Get the tool from GitHub
```
git clone https://github.com/paritoshpr/meliso.git
```

2. Run the following from the home directory i.e
```
export PYTHONPATH=$PYTHONPATH:./build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/
cd meliso && make all
```

3. Clean using
```
make clean
make clean-neurosim
make clean-meliso
```
