import argparse
import subprocess

nodes = [2, 4, 8, 16, 32, 64]  # Number of nodes for scaling exps


def generate_weak_scaling_job_scripts(
    log_n_samples: int, log_n_features: int, n_trees_base: int, submit: bool = False
) -> None:
    """
    Generate the job scripts for the weak scaling experiments.

    NOTE: Our n6m4 weak scaling experiments use 800 trees, the n7m3 experiments use 224 trees as a baseline.

    Parameters
    ----------
    log_n_samples : int
        The common logarithm of the number of samples to use.
    log_n_features : int
        The common logarithm of the number of features to use.
    n_trees_base : int
        The number of trees to use as a baseline.
    submit : bool, optional
        Whether to submit jobs to the cluster. Default is False.
    """
    for n_nodes in nodes:
        n_trees = n_trees_base * n_nodes
        print(f"Current config uses {n_nodes} nodes and {n_trees} trees.")
        job_name = f"n{log_n_samples}_m{log_n_features}_weak_{n_nodes}"
        job_script_name = f"{job_name}.sh"
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}         # Job name
#SBATCH --partition=cpuonly           # Queue for resource allocation
#SBATCH --time=2-12:00:00             # Wall-clock time limit
#SBATCH --mem=243200mb		          # Main memory (use standard nodes)
#SBATCH --cpus-per-task=76            # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --nodes={n_nodes}           # Number of nodes
#SBATCH --ntasks-per-node=1           # One MPI rank per node

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${{BASE_DIR:-/hkfs/work/workspace/scratch/ku4408-special-couscous/}}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export PYDIR=${{BASE_DIR}}/special-couscous/specialcouscous

ml purge              # Unload all currently loaded modules.
ml load compiler/gnu  # Load required modules.
ml load mpi/openmpi
source "${{BASE_DIR}}"/special-couscous-venv/bin/activate  # Activate venv.

N_SAMPLES={10**log_n_samples}
N_FEATURES={10**log_n_features}
N_TREES={n_trees}

SCRIPT="scripts/examples/rf_parallel_synthetic.py"

RESDIR="${{BASE_DIR}}"/results/weak_scaling/n6_m4_nodes_${{SLURM_NPROCS}}_${{SLURM_JOB_ID}}/
mkdir "${{RESDIR}}"
cd "${{RESDIR}}" || exit

srun python -u ${{PYDIR}}/${{SCRIPT}} --n_samples ${{N_SAMPLES}} --n_features ${{N_FEATURES}} --n_trees ${{N_TREES}} --output_dir ${{RESDIR}} --output_label ${{SLURM_JOB_ID}} --detailed_evaluation --save_model
                                """

        with open(job_script_name, "wt") as f:
            f.write(script_content)
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
    parser.add_argument(
        "--n_trees_base", type=int, help="The number of trees to use as a baseline."
    )
    args = parser.parse_args()
    # Generate job scripts and possibly submit them to the cluster.
    generate_weak_scaling_job_scripts(
        args.log_n_samples, args.log_n_features, args.n_trees_base, args.submit
    )
