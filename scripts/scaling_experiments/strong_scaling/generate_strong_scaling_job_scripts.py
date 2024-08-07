import argparse
import subprocess

limit = 60 * 24 * 3  # HoreKa wall-clock time limit in minutes
nodes = [2, 4, 8, 16, 32, 64]  # Number of nodes for scaling exps


def generate_strong_scaling_job_scripts(
    log_n_samples: int, log_n_features: int, n_trees: int, submit: bool = False
) -> None:
    """
    Generate the job scripts for the strong scaling experiments.

    NOTE: Our n6_m4 strong scaling experiments use 1600 trees, the n7_m3 experiments use 448 trees.

    Parameters
    ----------
    log_n_samples : int
        The common logarithm of the number of samples to use.
    log_n_features : int
        The common logarithm of the number of features to use.
    n_trees : int
        The number of trees to use.
    submit : bool, optional
        Whether to submit jobs to the cluster. Default is False.
    """
    for n_nodes in nodes:
        time = int(limit / n_nodes * 1.2)

        print(f"Current config uses {n_nodes} nodes. Wall-clock time is {time / 60} h.")
        job_name = f"n{log_n_samples}_m{log_n_features}_strong_{n_nodes}"
        job_script_name = f"{job_name}.sh"
        scriptcontent = f"""#!/bin/bash
#SBATCH --job-name={job_name}  # Job name
#SBATCH --partition=cpuonly    # Queue for resource allocation
#SBATCH --time={time}          # Wall-clock time limit
#SBATCH --cpus-per-task=76     # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL        # Notify user by email when certain event types occur.
#SBATCH --nodes={n_nodes}    # Number of nodes
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
# We estimated 1500 and 450 trees should be trainable in serial in 3 days for 1M samples with 10k features and 10M
# samples with 1k features, respectively, and chose the closest number evenly divisible by 64 as a baseline.
N_SAMPLES={10**log_n_samples}
N_FEATURES={10**log_n_features}
N_TREES={n_trees}

SCRIPT="scripts/examples/rf_parallel_synthetic.py"

RESDIR=${{BASE_DIR}}/results/strong_scaling/n6_m4_nodes_${{SLURM_NPROCS}}_job_${{SLURM_JOB_ID}}/
mkdir "${{RESDIR}}"
cd "${{RESDIR}}" || exit

# Run script
srun python -u ${{PYDIR}}/${{SCRIPT}} --n_samples ${{N_SAMPLES}} --n_features ${{N_FEATURES}} --n_trees ${{N_TREES}} --output_dir ${{RESDIR}} --output_label ${{SLURM_JOB_ID}} --detailed_evaluation --save_model
                                """
        # Write script content to file.
        with open(job_script_name, "wt") as f:
            f.write(scriptcontent)
        # Possibly submit script to cluster.
        if submit:
            subprocess.run(f"sbatch {job_script_name}", shell=True)


if __name__ == "__main__":
    # Parse command-line argument.
    parser = argparse.ArgumentParser(
        prog="Random Forest",
        description="Generate synthetic classification data and classify with (distributed) random forest.",
    )
    parser.add_argument(
        "--submit", action="store_true", help="Whether to submit jobs to the cluster."
    )
    parser.add_argument(
        "--log_n_samples",
        type=int,
        help="The common logarithm of the number of samples to use.",
    )
    parser.add_argument(
        "--log_n_features",
        type=int,
        help="The common logarithm of the number of features to use.",
    )
    parser.add_argument("--n_trees", type=int, help="The number of trees to use.")
    args = parser.parse_args()
    # Generate job scripts and possibly submit them to the cluster.
    generate_strong_scaling_job_scripts(
        args.log_n_samples, args.log_n_features, args.n_trees, args.submit
    )
