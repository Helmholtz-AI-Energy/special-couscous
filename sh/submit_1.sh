#!/bin/bash

#SBATCH --job-name=RF1                  # job name
#SBATCH --partition=single              # queue for the resource allocation
#SBATCH --mem=180000mb			# max. single: 180000mb, max. fat: 3000000mb
#SBATCH --time=9:00:00                  # wall-clock time limit  
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=ALL                 # Notify user by email when certain event types occur.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module purge                          # Unload all currently loaded modules.
module restore RF                     # Load required modules.  
source ~/.venvs/sklearn/bin/activate  # Activate your virtual environment.

RESDIR=../res/job_${SLURM_JOB_ID}/
mkdir ${RESDIR}
cd ${RESDIR}
export PYDIR=/pfs/work7/workspace/scratch/ku4408-RandomForest/special-couscous/py
python -u $PYDIR/RF_serial.py --random_state_split 9 --n_trees 1000 --random_state_forest 17 
