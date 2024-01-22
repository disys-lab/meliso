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
cd meliso
export PYTHONPATH=$PYTHONPATH:./build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/
make all
```

3. Clean using
```
make clean
make clean-neurosim
make clean-meliso
```

Run through the tutorial in `MelisoDriver.py`

## Docker build
1. After git clone run `cd  meliso && docker build -t pramanan3/meliso:v0.0.3 . `
2. Run the built container using `docker run -v /path/to/meliso:/meliso_edit/ --name meliso -d pramanan3/meliso:v0.0.3 /bin/sh -c "tail -f /dev/null"`

Pull from dockerhub using
`docker pull pramanan3/meliso:v0.0.3`

## Tutorial 
Run through the tutorial in `MelisoDriver.py`. The tutorial presents a simple example of using Meliso to perform MatVecs.

