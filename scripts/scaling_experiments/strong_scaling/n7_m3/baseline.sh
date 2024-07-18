#!/bin/bash
#SBATCH --job-name=n7_m3_base         # job name
#SBATCH --partition=cpuonly           # queue for resource allocation
#SBATCH --mem=501600mb
#SBATCH --time=3-00:00:00             # wall-clock time limit
#SBATCH --cpus-per-task=76            # number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --account=hk-project-test-aihero2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous/py

ml purge                          # Unload all currently loaded modules.
ml load compiler/gnu              # Load required modules.
ml load mpi/openmpi
source /hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous-venv/bin/activate  # Activate venv.

N_SAMPLES=10000000
N_FEATURES=1000
N_TREES=448  # We estimated 450 trees to be trainable in serial in 3 days and chose the next smaller number evenly divisible by 64 (scaling experiments).

SCRIPT="RF_serial_synthetic.py"

RESDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/results/single_node_experiments/job_${SLURM_JOB_ID}_n7_m3_t2/
mkdir ${RESDIR}
cd ${RESDIR}

python -u ${PYDIR}/${SCRIPT} --n_samples ${N_SAMPLES} --n_features ${N_FEATURES} --n_trees ${N_TREES}
