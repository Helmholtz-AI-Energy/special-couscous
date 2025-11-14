#!/bin/bash
#SBATCH --partition=cpuonly
#SBATCH --time=30
#SBATCH --cpus-per-task=76
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Unload all currently loaded modules and load required modules.
ml purge
ml load compiler/llvm
ml load mpi/openmpi/4.1

# Activate venv
source venv/bin/activate
srun python -u scripts/examples/rf_training_on_dataset.py --dataset_name higgs --n_trees 640
