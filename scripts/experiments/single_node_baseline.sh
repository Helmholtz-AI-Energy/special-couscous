#!/bin/bash
#SBATCH --job-name=base      # Job name
#SBATCH --partition=cpuonly  # Queue for resource allocation
#SBATCH --mem=501600mb       # Memory required per node
#SBATCH --time=3-00:00:00    # Wall-clock time limit
#SBATCH --cpus-per-task=76   # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL      # Notify user by email when certain event types occur.

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-special-couscous/}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  # Set number of threads to number of CPUs per task as provided by SLURM.
export PYDIR=${BASE_DIR}/special-couscous/specialcouscous  # Set path to Python package directory.

ml purge              # Unload all currently loaded modules.
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi

source "${BASE_DIR}"/special-couscous-venv/bin/activate  # Activate virtual environment.

# Set hyperparameters of synthetic dataset and random forest model for baseline runs.
# Choose a number of trees t evenly divisible by 64 to enable easy scaling.
# We considered three different datasets with varying number of samples n and features m, resulting in three baselines:
# n6m4: (n, m, t) = (10^6, 10^4, 1600)
# n7m3: (n, m, t) = (10^7, 10^3, 448)
# n8m2: (n, m, t) = (10^8, 10^2, 64)
#
# This single-node baseline script can also be used as a baseline for our strong and weak scaling experiments (also see
# ``script_content`` in respective job script generation Python scripts).
# Serial baselines for strong scaling: n6mw (n, m, t) = (10^6, 10^4, 1600); n7m3 (n, m, t) = (10^7, 10^3, 448)
# Serial baselines for weak scaling: n6mw (n, m, t) = (10^6, 10^4, 800); n7m3 (n, m, t) = (10^7, 10^3, 224)
#
# ADAPT NUMBERS BELOW AS NEEDED:
N_SAMPLES=100000000
N_FEATURES=100
N_TREES=64

SCRIPT="scripts/examples/rf_serial_synthetic.py"  # Set script name.

# Create directory to save results to.
RESDIR=${BASE_DIR}/results/single_node_experiments/${SLURM_JOB_ID}/
mkdir "${RESDIR}"
cd "${RESDIR}" || exit

# Run script.
# SET COMMAND-LINE ARGUMENTS AS NEEDED:
python -u "${PYDIR}"/${SCRIPT} \
    --n_samples ${N_SAMPLES} \
    --n_features ${N_FEATURES} \
    --n_trees ${N_TREES} \
    --detailed_evaluation \
    --save_model \
    --output_dir "${RESDIR}" \
    --output_label "${SLURM_JOB_ID}" \
    --log_path "${RESDIR}"
