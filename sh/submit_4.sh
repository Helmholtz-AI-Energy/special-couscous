#!/bin/bash

#SBATCH --job-name=RF4Chunks            # job name
#SBATCH --partition=sdil        	# queue for the resource allocation
#SBATCH --nodes=4
#SBATCH --time=12:00:00                	# wall-clock time limit  
#SBATCH --mem=90000                     # memory per node
#SBATCH --ntasks-per-node=1             # maximum count of tasks per node
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=ALL                 # Notify user by email when certain event types occur.

export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module purge                          # Unload all currently loaded modules.
module restore RF                     # Load required modules.  
source ~/.venvs/sklearn/bin/activate  # Activate your virtual environment.

RESDIR=../res/job_${SLURM_JOB_ID}/
mkdir ${RESDIR}
cd ${RESDIR}
export PYDIR=/pfs/work7/workspace/scratch/ku4408-RandomForest/special-couscous/py

mpirun --mca mpi_warn_on_fork 0 python -u $PYDIR/main.py --dataloader root_wo_replace --random_state_data 9 --n_trees 101 --random_state_forest 17 --n_classes 2
