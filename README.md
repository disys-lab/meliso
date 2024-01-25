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

## Using Docker container for development and testing
1. Pull the image `pramanan3/meliso:v0.0.4` from DockerHub either using the GUI (Docker Desktop) or through the command line.
2. Run the container and map the directory of your working copy to the directory `/meliso_edit/` on the container. 
You can do this by either mapping on the GUI or through the command line using the following commmand
`docker run -v /path/to/meliso/on/host:/neurosim_edit/ --name neurosim -d pramanan3/meliso:v0.0.4 /bin/sh -c "tail -f /dev/null" `
3. You can then log into the running container by using the command `docker exec -it neurosim bash`

## Using MPI on the Docker container
- Use the following command to run simulations with MPI enabled.
- `mpiexec --allow-run-as-root --oversubscribe -n <num_procs> python3 <filename.py>`
- For example to run `MelisoDriver.py` on 4 separate processes, execute the following command inside the container command prompt.
`mpiexec --allow-run-as-root --oversubscribe -n 4 python3 MelisoDriver.py`

