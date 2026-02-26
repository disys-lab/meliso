# Tutorials for MELISO+

Created by: Huynh Quang Nguyen Vo

Last updated time: February 23, 2026 1:28 PM

# Distributed Matrix-Vector Multiplication (distributedMVM)

## Installation

<aside>

The following installation steps are introduced under the assumption that you are running the experiments on supercomputers with [SLURM](https://slurm.schedmd.com/documentation.html) (e.g., OSU Pete Supercomputer).

</aside>

1. Clone the repository from GitHub:
    
    ```bash
    git clone https://github.com/disys-lab/meliso.git
    ```
    
2. Run this command from the home directory:
    
    ```bash
    cd meliso
    ```
    
    1. Install the required dependencies:
        
        ```bash
        pip install -r requirements.txt
        ```
        
    2. Since MELISO+ is constructed as a Python wrapper built upon a C++ backend (please refer) , run these commands to compile and build the backend:
        
        ```bash
        mkdir build
        export PYTHON_PATH=$PYTHON_PATH:./build
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/
        make all
        ```
        
    3. If there are changes made in the C++ backend, run these commands to recompile and build an updated backend:
        
        ```bash
        make clean
        make clean-neurosim
        make clean-meliso
        mkdir build
        export PYTHON_PATH=$PYTHON_PATH:./build
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/
        make all
        ```
        
3. Run the demo for matrix-vector multiplication with RRAMs.

## Run the distributedMVM with different device materials

Suppose we have a system of four RRAM crossbar arrays, and we want to run experiments for distributedMVM given an input matrix, says `bcsstk06.mtx` from [SuiteSparse Matrix Collection](https://sparse.tamu.edu/), and a given input vector. We select $\text{TaO}_x/\text{HfO}_x$ as our device material of interest.

### Setting up the experiment

1. Create a YAML file (let’s name it as `exp1.yaml`) with the following attributes:
    
    ```yaml
    exp_params:
      turnOnHardware: 1
      turnOnScaling: 0
      matrix_name: "bcsstk02"
      matrix_file: "inputs/matrices/bcsstk02.mtx"
      distributed:
        decomposition_dir: "/tmp/"
        mca_rows: 1
        mca_cols: 1
    
    device_config:
      root: "config_files/"
      cell_rows: 66
      cell_cols: 66
      assignment:
        device_TaOx-HfOx.yaml: [[-1]]
    ```
    
    - **Explanation for important attributes in the YAML file**
        - Experiment parameters (`exp_params`):
            1. `matrix_name` and `matrix_file`: Name of the input matrix and its path.
            2. `distributed`: How the distributed computation is handled. In distributed computation, a matrix is broken down into many smaller blocks.
                1. `decomposition_dir`: where the block’s information (position, etc.) is stored.
                2. The number of blocks a matrix is broken down into is defined by the product `mca_rows` $\times$`mca_cols`. For instance,
                3. When `mca_rows: 1` and `mca_cols: 1` , there is only one block. Thus, the block has the same dimensions as the original matrix.
                4. When `mca_rows: 3` and`mca_cols: 3` , the matrix is divided into nine blocks in total, each of which has smaller dimensions compared to those of the original matrix.
        - Device configuration (`device_config`):
            1. The number of cells in a specific crossbar array is defined by the product `cell_rows` $\times$`cell_cols`. For instance, when `cell_rows: 66` and `cell_cols: 66`, the total number of cells in the array is $66\times66 = 4356$.
            2. The `assignment` represents whether the system is homogeneous (all arrays are made from the same material) or heterogenous (array $i$ is made from Material $i$, while array $j$ is made from Material $j$, and so on). For instance,
                1. `device_EpiRAM.yaml: [[-1]]` means all arrays are made from EpiRAM.
                2. `device_EpiRAM.yaml: [[0, 2]]` means the first and third arrays are made from EpiRAM.
                    
                    Below is an example of a heterogeneous system having $2\times2$ arrays:
                    
                    ```yaml
                    <other attributes>
                      assignment:
                        device_Ag-aSi.yaml: [[1]]
                        device_AlOx-HfO2: [[2]]
                        device_EpiRAM.yaml: [[3]]
                        device_TaOx-HfOx.yaml: [[4]]
                    ```
                    
2. Once `exp1.yaml` has been created, 
    1. In the home directory of the repository, we run these commands to recompile and build an updated backend:
        
        ```bash
        mkdir build
        make all
        ```
        
    2. If the backend has already compiled and built, no further action is need.

### Run the experiment

<aside>

The following steps are introduced under the assumption that you are running the experiments on supercomputers with [SLURM](https://slurm.schedmd.com/documentation.html) (e.g., OSU Pete Supercomputer).

</aside>

1. To run the experiment, you need to prepare a shell script which contains a sequence of commands used to interact with a computer. For our experiment, a shell script called [quickstart.sh](http://quickstart.sh) has already been prepared for us.
    - **Explanation of the shell script template for running experiments on SLURM-based supercomputers.**
        
        The shell script should contain these sections:
        
        1. Script direction (as a commented section):
            
            ```bash
            #===================================================================================================
            # SCRIPT DESCRIPTION
            #===================================================================================================
            # This script automates running a series of computational experiments using the
            # 'DistributedMatVec.py' Python script. It is designed to be submitted to a Slurm workload
            # manager.
            #
            # The script iterates through a predefined set of materials, experiment IDs, and
            # iteration limits. For each unique combination, it runs the simulation multiple times
            # (replications) and saves the output to a unique report file.
            #
            # Key operations:
            #   1. Sets up the necessary software environment (Anaconda, GCC).
            #   2. Defines experiment parameters in a dedicated configuration section.
            #   3. Loops through all parameter combinations.
            #   4. For each run, it creates a unique output directory and report file.
            #   5. Executes the Python script in parallel using 'mpiexec'.
            #===================================================================================================
            ```
            
        2. SLURM Directives
            
            ```bash
            #===================================================================================================
            # SLURM DIRECTIVES -- Job scheduler settings
            #===================================================================================================
            #SBATCH --partition batch                         # Partition (queue) to submit the job to
            #SBATCH --time 1:00:00                            # Maximum runtime in HH:MM:SS format
            #SBATCH --nodes=1                                 # Number of nodes requested
            #SBATCH --ntasks=2                                # Total MPI ranks
            #SBATCH --ntasks-per-node=2                       # MPI ranks per node
            #SBATCH --mem=32GB                                # Total memory required for the job
            #SBATCH --mail-user=lucius.vo@okstate.edu         # Email address for job notifications
            #SBATCH --mail-type=END                           # Send an email when the job finishes
            #SBATCH --output=logs/DistributedMatVec_%j.out
            #SBATCH --job-name=DistributedMatVec              # Job name for easier identification
            #===================================================================================================
            ```
            
        3. Environmental Setup:
            
            ```bash
            #===================================================================================================
            # ENVIRONMENT SETUP
            #===================================================================================================
            echo "[INFO] Setting up the environment..."
            
            # Load required modules
            # Please contact with your respective supercomputers' system admins for available modules pertaining
            # to loading/running Python scripts and compiling C++ programs.
            module load anaconda3/2022.10
            module load gcc/7.5.0
            
            # Activate the specific conda environment
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate mpienv38
            
            # Add the local 'build' directory to Python's search path and the dynamic linker's path
            export PYTHONPATH="${PYTHONPATH:-}:./build"
            export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:./build/"
            
            echo "[INFO] Environment setup complete."
            #===================================================================================================
            ```
            
        4. Error Handling:
            
            ```bash
            #===================================================================================================
            # SCRIPT BEHAVIOR & ERROR HANDLING
            #===================================================================================================
            # 'set -e' will cause the script to exit immediately if any command fails.
            # 'set -u' will treat unset variables as an error, preventing unexpected behavior.
            # 'set -o pipefail' ensures that a pipeline's exit code is the status of the last command to exit
            # with a non-zero status, or zero if no command failed.
            set -euo pipefail
            #===================================================================================================
            ```
            
        5. Experiment Configuration:
            
            ```bash
            #===================================================================================================
            # EXPERIMENT CONFIGURATION
            #===================================================================================================
            EXPERIMENT_NAME="quickstart"        # A name for this set of experiments, used in report paths.
            NUM_PROCESSES=2                     # Total number of MPI processes to use for each run.
            DEVICE_TYPE=1                       # Device type to use for the experiments.
            NUM_REPLICATIONS=1                  # Number of repetitions for each experiment configuration
            EXPERIMENT_IDS=("1")                # A list of experiment IDs to run. Example: ("1" "2" "3")
            ITERATION_LIMITS=(21)               # A list of iteration limits for the write-and-verify. Example: (1 21)
            ENABLE_OVERRIDE=0                   # Whether to enable the override feature for iteration limits (1 for true, 0 for false).
            INPUT_VECTOR_PATH="inputs/vectors/input_x.txt" # The file path for the common input vector.
            
            # Define the materials to be tested and the paths to their configuration directories.
            declare -A MATERIAL_CONFIGS=(
                ["TaOx-HfOx"]="config_files/quickstart/TaOx-HfOx"
                )
            ```
            
            - The inputs of the parameter `NUM_PROCESSES` depend on the number of crossbar arrays mentioned in our experiment YAML files. For instance, if in our `exp1.yaml`, we are using one array, then `NUM_PROCESSES=2` (one for the Root process, the other for Non-Root processes). Therefore, the number of MPI processes is always $N_{\text{crossbar arrays}} + 1$.
            - The inputs of the parameter `DEVICE_TYPE` depend on what device types we are considering.
                - If we are considering RRAM devices, then `DEVICE_TYPE=1`.
                - If we are considering SRAM devices, then`DEVICE_TYPE=1`.
                - For other devices, please refer to the README file.
            - The inputs of the parameter `EXPERIMENT_IDS` depend on the names of our experiment YAML files. For instance, if we have three YAML files namely `exp1.yaml`, `exp2.yaml`, and `exp3.yaml`, then the appropriate command will be:
                
                ```bash
                EXPERIMENT_IDS=("1" "2" "3") 
                ```
                
            - The inputs of the parameter `ITERATION_LIMITS` depend on the number of cases for the write-and-verify scheme. For instance, if we want to test the write-and-verify scheme in three different iteration limits—e.g., $k = 1$, $k = 11$, and $k=21$—then the appropriate command will be:
                
                ```bash
                ITERATION_LIMITS=(1 11 21) 
                ```
                
            - The inputs of the parameter `ENABLE_OVERRIDE` depend on whether we want the write-and-verify scheme to run at the iteration limits.
                - When `ENABLE_OVERRIDE=1` and, for example,  `ITERATION_LIMITS=21`, then the write-and-verify scheme will run till $k=21$.
                - When `ENABLE_OVERRIDE=0` and, then the write-and-verify scheme will run until the synaptic weights is no longer needed to be updated. For instance, at `ENABLE_OVERRIDE=0`, it takes approx. $k^\star = 5$ for $\text{EpiRAM}$ devices.
            - The inputs of the parameter MATERIAL_CONFIGS depend on the number of devices of interests. For instance, if we have four different device materials namely $\text{Ag-aSi}$, $\text{AlO}_x\text{-HfO}_2$, $\text{EpiRAM}$, and $\text{TaO}_x\text{-HfO}_x$, then the appropriate command will be:
                
                ```bash
                declare -A MATERIAL_CONFIGS=(
                    ["Ag-aSi"]="config_files/quickstart/Ag-aSi"
                    ["AlOx-HfO2"]="config_files/quickstart/AlOx-HfO2"
                    ["EpiRAM"]="config_files/quickstart/EpiRAM"
                    ["TaOx-HfO2"]="config_files/quickstart/TaOx-HfO2"
                    )
                ```
                
        6. Experiment execution:
            
            ```bash
            #===================================================================================================
            # EXPERIMENT EXECUTION LOOP
            #===================================================================================================
            echo "[INFO] Starting experiment execution..."
            
            # Loop over each material name defined in the MATERIAL_CONFIGS array.
            for material in "${!MATERIAL_CONFIGS[@]}"; do
                CONFIG_PATH="${MATERIAL_CONFIGS[$material]}"
            
                # Loop over each experiment ID.
                for expid in "${EXPERIMENT_IDS[@]}"; do
                    EXP_CONFIG_FILE="${CONFIG_PATH}/exp${expid}.yaml"
            
                    # Loop over each specified iteration limit.
                    for iter_limit in "${ITERATION_LIMITS[@]}"; do
            
                        # Run the same experiment configuration multiple times for statistical robustness.
                        for ((rep=1; rep<=NUM_REPLICATIONS; rep++)); do
            
                            # --- PREPARE FOR THIS RUN ---
                            echo "[INFO] RUNNING: Device_Type='${DEVICE_TYPE}', Experiment_name='${EXPERIMENT_NAME}', Override='${ENABLE_OVERRIDE}'"
                            echo "[INFO] RUNNING: Material='${material}', Exp_ID='${expid}', Iter_Limit='${iter_limit}', Repetition='${rep}'"
            
                            # Define the output path for the report file for this specific run.
                            REPORT_PATH="reports/${EXPERIMENT_NAME}/${material}/exp${expid}_iter_${iter_limit}_rep_${rep}.txt"
            
                            # Create the directory structure for the report file if it does not already exist.
                            mkdir -p "$(dirname "$REPORT_PATH")"
            
                            # For a clean run, remove the report file from a previous run if it exists.
                            if [ -f "$REPORT_PATH" ]; then
                                echo "[INFO] WARNING: Removing old report file: $REPORT_PATH"
                                rm "$REPORT_PATH"
                            fi
            
                            # --- EXECUTE THE SIMULATION ---
                            echo "[INFO] Executing 'mpiexec' ..."
                            echo "[INFO]   - Config File: ${EXP_CONFIG_FILE}"
                            echo "[INFO]   - Report File: ${REPORT_PATH}"
            
                            # Set environment variables for the Python script and run it with mpiexec.
                            # These variables are only set for the duration of this single command.
                            DT=${DEVICE_TYPE} OVERRIDE=${ENABLE_OVERRIDE} \
                            ITER_LIMIT="$iter_limit" XVEC_PATH="$INPUT_VECTOR_PATH" \
                            EXP_CONFIG_FILE="$EXP_CONFIG_FILE" REPORT_PATH="$REPORT_PATH" \
                            mpiexec -n 2 python3 distributedMatVec.py
            
                            echo "[INFO] DONE: Repetition ${rep} finished."
                            echo -e "\n"
            
                        done # End of replications loop
                    done # End of iteration limits loop
                done # End of experiment IDs loop
            done # End of materials loop
            
            echo "[INFO] All experiments completed successfully!"
            ```
            
2. Once a shell script is created, we can run the experiment using this command from the home directory:
    
    ```bash
    sbatch run_quickstart.sh
    ```
    
    1. To avoid potential bugs when running the experiment, please ensure that the YAML files (device materials and experiment details) are placed in appropriate directories. Below is an example of a correct directory for YAML files.
        
        ```bash
        meliso
        ├── config_files/
        │   ├── quickstart/
        │   │   ├── EpiRAM/
        │   │   │   └── exp1.yaml
        │   ├── device_Ag-aSi.yaml
        │   ├── device_AlOx-HfO2.yaml
        │   ├── device_EpiRAM.yaml
        │   └── device_TaOx-HfOx.yaml
        └── <other files and folders>
        ```
        

## Run the distributedMVM for Nature Communications

<aside>

The following steps are introduced under the assumption that you are running the experiments on supercomputers with [SLURM](https://slurm.schedmd.com/documentation.html) (e.g., OSU Pete Supercomputer).

</aside>

1. Ensure that all YAML files are in the appropriate directories.
    - **How the appropriated directories appear for all YAML files**
        
        ```bash
        meliso
        ├── config_files/
        │   ├── iterations/
        │   │   ├── Ag-aSi/
        │   │   │   ├── exp1.yaml
        │   │   │   ├── exp2.yaml
        │   │   │   ├── exp3.yaml
        │   │   │   └── exp4.yaml
        │   │   ├── AlOx-HfO2/
        │   │   │   ├── exp1.yaml
        │   │   │   ├── exp2.yaml
        │   │   │   ├── exp3.yaml
        │   │   │   └── exp4.yaml
        │   │   ├── EpiRAM/
        │   │   │   ├── exp1.yaml
        │   │   │   ├── exp2.yaml
        │   │   │   ├── exp3.yaml
        │   │   │   └── exp4.yaml
        │   │   └── TaOx-HfOx/
        │   │       ├── exp1.yaml
        │   │       ├── exp2.yaml
        │   │       ├── exp3.yaml
        │   │       └── exp4.yaml
        │   ├── virtualization/
        │   │   ├── strongScaling/
        │   │   │   ├── Ag-aSi/
        │   │   │   │   ├── exp1.yaml
        │   │   │   │   ├── exp2.yaml
        │   │   │   │   ├── exp3.yaml
        │   │   │   │   ├── exp4.yaml
        │   │   │   │   ├── exp5.yaml
        │   │   │   │   └── exp6.yaml
        │   │   │   ├── AlOx-HfO2/
        │   │   │   │   ├── exp1.yaml
        │   │   │   │   ├── exp2.yaml
        │   │   │   │   ├── exp3.yaml
        │   │   │   │   ├── exp4.yaml
        │   │   │   │   ├── exp5.yaml
        │   │   │   │   └── exp6.yaml
        │   │   │   ├── EpiRAM/
        │   │   │   │   ├── exp1.yaml
        │   │   │   │   ├── exp2.yaml
        │   │   │   │   ├── exp3.yaml
        │   │   │   │   ├── exp4.yaml
        │   │   │   │   ├── exp5.yaml
        │   │   │   │   └── exp6.yaml
        │   │   │   └── TaOx-HfOx/
        │   │   │       ├── exp1.yaml
        │   │   │       ├── exp2.yaml
        │   │   │       ├── exp3.yaml
        │   │   │       ├── exp4.yaml
        │   │   │       ├── exp5.yaml
        │   │   │       └── exp6.yaml
        │   │   └── weakScaling/
        │   │       ├── Ag-aSi/
        │   │       │   ├── exp1.yaml
        │   │       │   ├── exp2.yaml
        │   │       │   ├── exp3.yaml
        │   │       │   ├── exp4.yaml
        │   │       │   ├── exp5.yaml
        │   │       │   └── exp6.yaml
        │   │       ├── AlOx-HfO2/
        │   │       │   ├── exp1.yaml
        │   │       │   ├── exp2.yaml
        │   │       │   ├── exp3.yaml
        │   │       │   ├── exp4.yaml
        │   │       │   ├── exp5.yaml
        │   │       │   └── exp6.yaml
        │   │       ├── EpiRAM/
        │   │       │   ├── exp1.yaml
        │   │       │   ├── exp2.yaml
        │   │       │   ├── exp3.yaml
        │   │       │   ├── exp4.yaml
        │   │       │   ├── exp5.yaml
        │   │       │   └── exp6.yaml
        │   │       └── TaOx-HfOx/
        │   │           ├── exp1.yaml
        │   │           ├── exp2.yaml
        │   │           ├── exp3.yaml
        │   │           ├── exp4.yaml
        │   │           ├── exp5.yaml
        │   │           └── exp6.yaml
        │   ├── device_Ag-aSi.yaml
        │   ├── device_AlOx-HfO2.yaml
        │   ├── device_EpiRAM.yaml
        │   └── device_TaOx-HfOx.yaml
        └── <other files and folders>
        ```
        
2. Ensure that shell scripts, each corresponding to one experiment type, are created following the templates. Most importantly, ensure that the appropriate number of MPI processes are defined in these scripts.
    - **Examples of the shell scripts for all experiments in Nature Communication**
        
        Here, only the modified sections of the script are shown, the rest remained them same.
        
        1. Different iteration limits
            
            ```bash
            #===================================================================================================
            # SLURM DIRECTIVES -- Job scheduler settings
            #===================================================================================================
            #SBATCH --partition batch                         # Partition (queue) to submit the job to
            #SBATCH --time 12:00:00                           # Maximum runtime in HH:MM:SS format
            #SBATCH --nodes=1                                 # Number of nodes requested
            #SBATCH --ntasks=2                                # Total MPI ranks
            #SBATCH --ntasks-per-node=2                       # MPI ranks per node
            #SBATCH --mem=32GB                                # Total memory required for the job
            #SBATCH --mail-user=lucius.vo@okstate.edu         # Email address for job notifications
            #SBATCH --mail-type=END                           # Send an email when the job finishes
            #SBATCH --output=logs/VariedIterationsMatVec%j.out
            #SBATCH --job-name=VariedIterationsMatVec         # Job name for easier identification
            #===================================================================================================
            
            ...
            
            #===================================================================================================
            # EXPERIMENT CONFIGURATION
            #===================================================================================================
            EXPERIMENT_NAME="iterations"        # A name for this set of experiments, used in report paths.
            NUM_PROCESSES=2                     # Total number of MPI processes to use for each run.
            DEVICE_TYPE=1                       # Device type to use for the experiments.
            NUM_REPLICATIONS=100                # Number of repetitions for each experiment configuration
            EXPERIMENT_IDS=("1" "2" "3" "4")    # A list of experiment IDs to run.
            ITERATION_LIMITS=(21)               # A list of iteration limits for the write-and-verify.
            ENABLE_OVERRIDE=1                   # Whether to enable the override feature for iteration limits
            INPUT_VECTOR_PATH="inputs/vectors/input_x.txt" # The file path for the common input vector.
            
            # Define the materials to be tested and the paths to their configuration directories.
            declare -A MATERIAL_CONFIGS=(
                ["Ag-aSi"]="config_files/${EXPERIMENT_NAME}/Ag-aSi"
                ["AlOx-HfO2"]="config_files/${EXPERIMENT_NAME}/AlOx-HfO2"
                ["EpiRAM"]="config_files/${EXPERIMENT_NAME}/EpiRAM"
                ["TaOx-HfO2"]="config_files/${EXPERIMENT_NAME}/TaOx-HfO2"
                )
            
            ...
            ```
            
        2. Virtualization (weak-scaling)
            
            ```bash
            #===================================================================================================
            # SLURM DIRECTIVES -- Job scheduler settings
            #===================================================================================================
            #SBATCH --partition long                          # Partition (queue) to submit the job to
            #SBATCH --time 500:00:00                          # Maximum runtime in HH:MM:SS format
            #SBATCH --ntasks=65                               # Total MPI ranks
            #SBATCH --mem=72GB                                # Total memory required for the job
            #SBATCH --mail-user=lucius.vo@okstate.edu         # Email address for job notifications
            #SBATCH --mail-type=END                           # Send an email when the job finishes
            #SBATCH --job-name=weakScalingMatVec              # Job name for easier identification
            #===================================================================================================
            
            ...
            
            #===================================================================================================
            # EXPERIMENT CONFIGURATION
            #===================================================================================================
            EXPERIMENT_NAME="weakScaling"           # A name for this set of experiments, used in report paths.
            NUM_PROCESSES=2                         # Total number of MPI processes to use for each run.
            DEVICE_TYPE=1                           # Device type to use for the experiments.
            NUM_REPLICATIONS=100                    # Number of repetitions for each experiment configuration
            EXPERIMENT_IDS=("1" "2" "3" "4" "5" "6")    # A list of experiment IDs to run.
            ITERATION_LIMITS=(21)                       # A list of iteration limits for the write-and-verify.
            ENABLE_OVERRIDE=0                           # Whether to enable the override feature for iteration limits
            INPUT_VECTOR_PATH="inputs/vectors/input_x.txt" # The file path for the common input vector.
            
            # Define the materials to be tested and the paths to their configuration directories.
            declare -A MATERIAL_CONFIGS=(
                ["Ag-aSi"]="config_files/${EXPERIMENT_NAME}/Ag-aSi"
                ["AlOx-HfO2"]="config_files/${EXPERIMENT_NAME}/AlOx-HfO2"
                ["EpiRAM"]="config_files/${EXPERIMENT_NAME}/EpiRAM"
                ["TaOx-HfO2"]="config_files/${EXPERIMENT_NAME}/TaOx-HfO2"
                )
                
            ...
            ```
            
        3. Virtualization (strong-scaling)
            
            ```bash
            #===================================================================================================
            # SLURM DIRECTIVES -- Job scheduler settings
            #===================================================================================================
            #SBATCH --partition long                          # Partition (queue) to submit the job to
            #SBATCH --time 500:00:00                          # Maximum runtime in HH:MM:SS format
            #SBATCH --ntasks=65                               # Total MPI ranks
            #SBATCH --mem=72GB                                # Total memory required for the job
            #SBATCH --mail-user=lucius.vo@okstate.edu         # Email address for job notifications
            #SBATCH --mail-type=END                           # Send an email when the job finishes
            #SBATCH --job-name=strongScalingMatVec              # Job name for easier identification
            #===================================================================================================
            
            ...
            
            #===================================================================================================
            # EXPERIMENT CONFIGURATION
            #===================================================================================================
            EXPERIMENT_NAME="strongScaling"           # A name for this set of experiments, used in report paths.
            NUM_PROCESSES=65                        # Total number of MPI processes to use for each run.
            DEVICE_TYPE=1                           # Device type to use for the experiments.
            NUM_REPLICATIONS=100                    # Number of repetitions for each experiment configuration
            EXPERIMENT_IDS=("1" "2" "3" "4" "5" "6")    # A list of experiment IDs to run.
            ITERATION_LIMITS=(21)                       # A list of iteration limits for the write-and-verify.
            ENABLE_OVERRIDE=0                           # Whether to enable the override feature for iteration limits
            INPUT_VECTOR_PATH="inputs/vectors/input_x.txt" # The file path for the common input vector.
            
            # Define the materials to be tested and the paths to their configuration directories.
            declare -A MATERIAL_CONFIGS=(
                ["Ag-aSi"]="config_files/${EXPERIMENT_NAME}/Ag-aSi"
                ["AlOx-HfO2"]="config_files/${EXPERIMENT_NAME}/AlOx-HfO2"
                ["EpiRAM"]="config_files/${EXPERIMENT_NAME}/EpiRAM"
                ["TaOx-HfO2"]="config_files/${EXPERIMENT_NAME}/TaOx-HfO2"
                )
            
            ...
            ```
            
3. Include the matrices mentioned in the paper in the appropriate directories
    - **How the appropriated directories appear for all input matrices and vector(s)**
        
        ```bash
        meliso
        ├── inputs/
        │   ├── matrices/
        │   │   ├── bcsstk02.mtx
        │   │   ├── Ipertubed.mtx
        │   │   └── <other matrices>
        │   └── vectors/
        │       └── input_x.txt
        └── <other files and folders>
        ```
        
4. Run the experiment using this command from the home directory:
    1. Distributed MVM with different iteration limits enforced to the write-and-verify scheme:
        
        ```bash
        sbatch run_varied_iters.sh
        ```
        
    2. Virtualization with weak-scaling (the input matrix is fixed, the number of crossbar arrays is fixed, only the number of cells per array is changing):
        
        ```bash
        sbatch run_weakScaling.sh
        ```
        
    3. Virtualization with strong-scaling (the number of crossbar arrays is fixed, the number of cells per array is fixed, only the input matrices are changing):
        
        ```bash
        sbatch run_strongScaling.sh
        ```
