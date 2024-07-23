import subprocess

limit = 60 * 24 * 3  # HoreKa wall-clock time limit in minutes
nodes = [2, 4, 8, 16, 32, 64]  # Number of nodes for scaling exps


def main() -> None:
    """Generate the job scripts for the n6_m4 strong scaling experiments."""
    for num_nodes in nodes:
        time = int(limit / num_nodes * 1.2)

        print(
            f"Current config uses {num_nodes} nodes. Wall-clock time is {time / 60} h."
        )
        job_name = f"n6_m4_strong_{num_nodes}"
        job_script_name = f"{job_name}.sh"
        scriptcontent = f"""#!/bin/bash
#SBATCH --job-name={job_name}  # Job name
#SBATCH --partition=cpuonly    # Queue for resource allocation
#SBATCH --time={time}          # Wall-clock time limit
#SBATCH --cpus-per-task=76     # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL        # Notify user by email when certain event types occur.
#SBATCH --nodes={num_nodes}    # Number of nodes
#SBATCH --ntasks-per-node=1    # One MPI rank per node

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${{BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-special-couscous/}}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}  # Set number of threads to number of CPUs per task as provided by SLURM.
export PYDIR=${{BASE_DIR}}/special-couscous/specialcouscous  # Set path to Python package directory.

ml purge              # Unload all currently loaded modules.
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi

source "${{BASE_DIR}}"/special-couscous-venv/bin/activate  # Activate virtual environment.

# Set hyperparameters of synthetic dataset and random forest model.
# We estimated 1500 trees should be trainable in serial in 3 days
# and chose the closest number evenly divisible by 64 as a baseline for scaling exps.
N_SAMPLES=1000000
N_FEATURES=10000
N_TREES=1600

SCRIPT="scripts/examples/rf_parallel_synthetic.py"

RESDIR=${{BASE_DIR}}/results/strong_scaling/n6_m4_nodes_${{SLURM_NPROCS}}_job_${{SLURM_JOB_ID}}/
mkdir "${{RESDIR}}"
cd "${{RESDIR}}" || exit

# Run script
srun python -u ${{PYDIR}}/${{SCRIPT}} --n_samples ${{N_SAMPLES}} --n_features ${{N_FEATURES}} --n_trees ${{N_TREES}} --output_dir ${{RESDIR}} --output_label ${{SLURM_JOB_ID}} --detailed_evaluation --save_model
                                """

        with open(job_script_name, "wt") as f:
            f.write(scriptcontent)

        subprocess.run(f"sbatch {job_script_name}", shell=True)


if __name__ == "__main__":
    main()
