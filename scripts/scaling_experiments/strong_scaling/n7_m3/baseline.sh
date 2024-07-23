#!/bin/bash
#SBATCH --job-name=n7_m3_base  # Job name
#SBATCH --partition=cpuonly    # Queue for resource allocation
#SBATCH --mem=501600mb         # Memory requested per node
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
# # We estimated 450 trees to be trainable in serial in 3 days
# and chose the next smaller number evenly divisible by 64 (scaling experiments).
N_SAMPLES=10000000
N_FEATURES=1000
N_TREES=448

SCRIPT="scripts/examples/rf_serial_synthetic.py"  # Set script name.

# Create directory to save results to.
RESDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/results/single_node_experiments/job_${SLURM_JOB_ID}_n6_m4_baseline/
mkdir "${RESDIR}"
cd "${RESDIR}" || exit

# Run script.
python -u "${PYDIR}"/${SCRIPT} --n_samples ${N_SAMPLES} --n_features ${N_FEATURES} --n_trees ${N_TREES} --output_dir "${RESDIR}" --output_label "${SLURM_JOB_ID}" --detailed_evaluation --save_model
