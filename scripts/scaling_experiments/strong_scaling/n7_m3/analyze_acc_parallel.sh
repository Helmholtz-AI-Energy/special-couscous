#!/bin/bash
#SBATCH --job-name=n7_m3_strong_2          # Job name
#SBATCH --partition=cpuonly            # Queue for resource allocation
#SBATCH --time=60                 # Wall-clock time limit
#SBATCH --cpus-per-task=1            # Number of CPUs required per (MPI) task
#SBATCH --mem=501600mb
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --account=hk-project-test-aihero2
#SBATCH --nodes=1           # Number of nodes
#SBATCH --ntasks-per-node=76           # One MPI rank per node.


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous/specialcouscous

ml purge                          # Unload all currently loaded modules.
ml load compiler/gnu              # Load required modules.
ml load mpi/openmpi
source /hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous-venv/bin/activate  # Activate venv.

N_SAMPLES=10000000
N_FEATURES=1000
N_TREES=76  # We estimated 450 trees should be trainable in serial in 3 d and chose the closest number evenly divisble by 64 as a baseline for scaling exps.

SCRIPT="rf_scaling_synthetic.py"

RESDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/results/strong_scaling/n7_m3_${SLURM_JOB_ID}_test_acc/
mkdir ${RESDIR}
cd ${RESDIR}

srun python -u ${PYDIR}/${SCRIPT} --n_samples ${N_SAMPLES} --n_features ${N_FEATURES} --n_trees ${N_TREES} --output_dir ${RESDIR} --output_label ${SLURM_JOB_ID} --detailed_evaluation
