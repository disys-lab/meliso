# MELISO: Memristor Linear Solver for Optimization

## File Structure
- **NeuroSim Files (in C++)**: Located in `src/mlp_neurosim`
- **Python Interfacing Files**: Found in `src/cython`
- **Python Distributed Computation Scheme Files**: Found in `src/core`
- **NeuroSim Manual**: Accessible in `documents/NeuroSimV3.0_user_manual.pdf`
- **MELISO Manual**: Accessible in `documents/MELISO_user_manual.pdf`

## Installation Steps
### For Linux-based OS

1. Clone the Repository
```
git clone https://github.com/paritoshpr/MELISO.git
```

2. Build and Setup
```
cd MELISO
mkdir build
export PYTHONPATH=$PYTHONPATH:./build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/
make all
```

3. Execute the tutorial script located in `MELISODriver.py`
```
python3 MELISODriver.py
```

4. To run the framework after modifying **NeuroSim Files**, first clean built artifacts then rebuild them
```
make clean
make clean-neurosim
make clean-MELISO
mkdir build
export PYTHONPATH=$PYTHONPATH:./build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/
make all
```
## For Windows OS (via Docker build)
1. Clone the Repository
```
git clone https://github.com/paritoshpr/MELISO.git
```

2. Build Docker Image
```
cd MELISO
docker build -t pramanan3/MELISO:v0.0.3 
```

3. Run Docker Container
```
docker run -v /path/to/MELISO:/MELISO_edit/ --name MELISO -d pramanan3/MELISO:v0.0.3 /bin/sh -c "tail -f /dev/null"`
```

4. Pull from DockerHub
```
docker pull pramanan3/MELISO:v0.0.3
```

## Tutorials
### Running MELISO for ICONS24
1. Follow the script `MELISODriver.py` for a simple example of using MELISO to perform matrix-vector multiplication (MVM). To perform MVM with different device types in `MELISODriver.py`, specify the first argument accordingly:
    * `0`: IdealDevice
    * `1`: RealDevice
    * `2`: MeasuredDevice
    * `3`: SRAM
    * `4`: DigitalNVM
    * `5`: HybridCell
    * `6`: _2T1F

2. Follow the script `MELISODriver_RealDevices.py` for an example of using MELISO to perform MVM with different memristor materials (Ag-aSi, AlOx/HfO2, TaOx/HfOx, and EpiRAM).

### Running MELISO for Distributed Computation
1. Ensure that the experiment configurations are properly defined in `config_files` folder. The experiment configurations are contained in `exp_<user_specific>.yaml` and `device_<user_specific>.yaml`. Follow these two files `exp_example.yaml` and `device_generic.yaml` for an example of supplying the configurations.

2. To run the experiments:
    * On [Pete Supercomputer](https://hpcc.okstate.edu/pete-supercomputer.html) (assuming you are authorized to do so AND familiar with the system), use the following command
`EXP_CONFIG_FILE=/path/to/exp/config/file mpiexec -n <numMCAs+1> python3 DistributedMatVec.py`
    * On Docker container, add `--allow-run-as-root --oversubscribe` tags

3. For the experiments to run smoothly, make sure that: 
    * The matrices must be located in `/path/to/MELISO/matrices` and the decomposition directory must be specified in the parameter 
`distributed: decomposition_dir:` in the `exp_<user_specific>.yaml` file (the default path is `/tmp/`).
    * The default device type is `1`, standing for RealDevice. Nevertheless, you can change to other device types using the following command: `DT=<device_type> EXP_CONFIG_FILE=/path/to/exp/config/file mpiexec -n <numMCAs+1> python3 DistributedMatVec.py`. For instance, use the following command to switch to IdealDevice: `DT=0 EXP_CONFIG_FILE=/path/to/exp/config/file mpiexec -n <numMCAs+1> python3 DistributedMatVec.py`

## Miscellaneous
### Using Docker Container for Development and Testing
1. **Pull the Docker Image**
    - Pull the image `pramanan3/MELISO:v0.0.4` from DockerHub using either the Docker Desktop GUI or the command line:
      ```sh
      docker pull pramanan3/MELISO:v0.0.4
      ```

2. **Run the Docker Container**
    - Run the container and map your working directory on the host to the `/MELISO_edit/` directory inside the container. This can be done using the Docker Desktop GUI or through the command line:
      ```sh
      docker run -v /path/to/MELISO/on/host:/MELISO_edit/ --name neurosim -d pramanan3/MELISO:v0.0.4 /bin/sh -c "tail -f /dev/null"
      ```

3. **Access the Running Container**
    - Log into the running container using the following command:
      ```sh
      docker exec -it neurosim bash
      ```

## Using MPI on the Docker Container
- To run simulations with MPI enabled, use the following command:
  ```
  sh mpiexec --allow-run-as-root --oversubscribe -n <num_procs> python3 <filename.py>
  ```
- For example, to run MELISODriver.py on 4 separate processes, execute the following command inside the container command prompt:
    ```
    mpiexec --allow-run-as-root --oversubscribe -n 4 python3 MELISODriver.py
    ```
- Note that the process count must be one greater than the number of memristor tiles you wish to simulate.

