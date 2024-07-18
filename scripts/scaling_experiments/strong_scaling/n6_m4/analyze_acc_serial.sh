#!/bin/bash
#SBATCH --job-name=n6_m4_test_acc     # job name
#SBATCH --partition=cpuonly           # queue for resource allocation
#SBATCH --mem=501600mb
#SBATCH --time=8:00:00                # wall-clock time limit
#SBATCH --cpus-per-task=76            # number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --account=hk-project-test-aihero2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous/specialcouscous

ml purge                          # Unload all currently loaded modules.
ml load compiler/gnu              # Load required modules.
ml load mpi/openmpi
source /hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous-venv/bin/activate  # Activate venv.

N_SAMPLES=1000000
N_FEATURES=10000
N_TREES=76  # We estimated 1500 trees should be trainable in serial in 3 d and chose the next smaller number evenly divisble by 64 (scaling exps)

SCRIPT="rf_serial_synthetic.py"

RESDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/results/strong_scaling/job_${SLURM_JOB_ID}_n6_m4_analyze_acc/
mkdir ${RESDIR}
cd ${RESDIR}

python -u ${PYDIR}/${SCRIPT} --n_samples ${N_SAMPLES} --n_features ${N_FEATURES} --n_trees ${N_TREES}
