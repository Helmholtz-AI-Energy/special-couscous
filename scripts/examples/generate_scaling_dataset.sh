#!/bin/bash
#SBATCH --job-name=generate_scaling_dataset    # Job name
#SBATCH --partition=cpuonly                    # Queue for resource allocation
#SBATCH --time=60                              # Wall-clock time limit (1h)
#SBATCH --mem=500gb                            # Main memory -> use high memory nodes
#SBATCH --cpus-per-task=76                     # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL                        # Notify user by email when certain event types occur.
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --account=hk-project-p0022229

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${{BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-SpecialCouscous}}
DATA_DIR="${BASE_DIR}/datasets"


ml purge              # Unload all currently loaded modules.
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi/4.1
source "${{BASE_DIR}}"/special-couscous-venv-openmpi4/bin/activate  # Activate venv.

SCRIPT="${BASE_DIR}/special-couscous/specialcouscous/scaling_dataset.py"

# NOTE: pass #samples and #features as arguments to this bash script, e.g. as
#       sbatch generate_scaling_dataset.sh --n_samples 64e6 --n_features 1e4              (for n6m4 baseline)
# or    sbatch generate_scaling_dataset.sh --n_samples 64e7 --n_features 1e3              (for n7m3 baseline)
# All arguments to the bash script are passed through directly to scaling_dataset.py, can also be used for other parameters
python specialcouscous/scaling_dataset.py --n_classes 10 --random_state 0 --n_train_splits 64 "$@"
