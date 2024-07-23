#!/bin/bash
#SBATCH --job-name=n8_m2_base  # Job name
#SBATCH --partition=cpuonly    # Queue for resource allocation
#SBATCH --mem=501600mb         # Memory required per node
#SBATCH --time=3-00:00:00      # Wall-clock time limit
#SBATCH --cpus-per-task=76     # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL        # Notify user by email when certain event types occur.

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-special-couscous/}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # Set number of threads to number of CPUs per task as provided by SLURM.
export PYDIR=${BASE_DIR}/special-couscous/specialcouscous  # Set path to Python package directory.

ml purge              # Unload all currently loaded modules.
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi

source "${BASE_DIR}"/special-couscous-venv/bin/activate  # Activate virtual environment.

# Set hyperparameters of synthetic dataset and random forest model.
# 100 trees were not trainable in serial in 3 days. Choose a number evenly divisible by 64 (scaling experiments).
# Training one tree takes approx. 1 h.
N_SAMPLES=100000000
N_FEATURES=100
N_TREES=64

SCRIPT="scripts/examples/rf_serial_synthetic.py"  # Set script name.

# Create directory to save results to.
RESDIR=${BASE_DIR}/results/single_node_experiments/n8_m2_baseline_${SLURM_JOB_ID}/
mkdir "${RESDIR}"
cd "${RESDIR}" || exit

# Run script.
python -u "${PYDIR}"/${SCRIPT} --n_samples ${N_SAMPLES} --n_features ${N_FEATURES} --n_trees ${N_TREES}
