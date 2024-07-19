#!/bin/bash

#SBATCH --job-name=RF1                # job name
#SBATCH --partition=dev_single            # queue for resource allocation
#SBATCH --mem=180000mb			          # max. single: 180000mb, max. fat: 3000000mb
#SBATCH --time=9:00:00                # wall-clock time limit
#SBATCH --cpus-per-task=76            # number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous/specialcouscous

ml purge                          # Unload all currently loaded modules.
ml load compiler/gnu              # Load required modules.
ml load mpi/openmpi
source /hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous-venv/bin/activate  # Activate venv.

N_SAMPLES=1000000
N_FEATURES=100
N_TREES=100

SCRIPT="RF_serial_synthetic.py"

RESDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/results/single_node_experiments/job_${SLURM_JOB_ID}/
mkdir ${RESDIR}
cd ${RESDIR}

python -u ${PYDIR}/${SCRIPT} --n_samples ${N_SAMPLES} --n_features ${N_FEATURES} --n_trees ${N_TREES}