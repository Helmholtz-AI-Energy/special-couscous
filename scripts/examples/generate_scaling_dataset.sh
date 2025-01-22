#!/bin/bash
#SBATCH --job-name=generate_scaling_dataset    # Job name
#SBATCH --partition=large                      # Queue for resource allocation
#SBATCH --time=60                              # Wall-clock time limit (1h)
#SBATCH --mem=500gb                            # Main memory -> use high memory nodes
#SBATCH --cpus-per-task=76                     # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL                        # Notify user by email when certain event types occur.
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --account=hk-project-p0022229

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-SpecialCouscous}
SCRIPT_DIR=${SCRIPT_DIR:-BASE_DIR}
DATA_DIR="${BASE_DIR}/datasets"
VENV=${VENV:-${BASE_DIR}"/special-couscous-venv-openmpi4"}

echo "BASE_DIR: ${BASE_DIR}"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "DATA_DIR: ${DATA_DIR}"
echo "VENV: ${VENV}"


ml purge              # Unload all currently loaded modules.
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi/4.1
source "${VENV}"/bin/activate  # Activate venv.

SCRIPT="${SCRIPT_DIR}/specialcouscous/scaling_dataset.py"

# NOTE: pass #ranks, #samples, and #features as arguments to this bash script, e.g. as
#     sbatch generate_scaling_dataset.sh --n_train_splits 64 --n_samples 64000000 --n_features 10000   (for n6m4)
# or  sbatch generate_scaling_dataset.sh --n_train_splits 64 --n_samples 640000000 --n_features 1000   (for n7m3)
# All arguments to the bash script are passed through directly to scaling_dataset.py, can also be used for other parameters
python "${SCRIPT}" --data_root_path "${DATA_DIR}" --n_classes 10 --random_state 0 "$@"
