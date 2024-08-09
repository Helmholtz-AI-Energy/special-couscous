import argparse
import pathlib
import subprocess

limit = 60 * 24 * 3  # HoreKa wall-clock time limit in minutes
nodes = [2, 4, 8, 16, 32, 64]  # Number of nodes for scaling exps


def generate_scaling_job_scripts(
    log_n_samples: int,
    log_n_features: int,
    n_trees: int,
    scaling_type: str,
    output_path: pathlib.Path,
    submit: bool = False,
) -> None:
    """
    Generate the job scripts for the scaling experiments.

    NOTE: We estimated 1500 and 450 trees to be trainable in serial in 3 days for 1M samples with 10k features and 10M
    samples with 1k features, respectively, and chose the closest number evenly divisible by 64 as a baseline.
    With number of samples n, number of features m, and number of trees t:

    Strong scaling:
    n6m4 baseline (n, m, t) = (10^6, 10^4, 1600) and n7m3 baseline: (n, m, t) = (10^7, 10^3, 448)

    Weak scaling:
    n6m4 baseline (n, m, t) = (10^6, 10^4, 800) and n7m3 baseline: (n, m, t) = (10^7, 10^3, 224)

    NOTE: All strong-scaling experiments used high-memory nodes, i.e., #SBATCH --mem=486400mb, except for the 64-node
    experiment which used the normal nodes. This is due to the fact that HoreKa has only 32 high-memory nodes. However,
    as the problem size per node decreases with increasing number of nodes in strong scaling, this was not a problem here
    but only for weak scaling. That is why the base problem size of weak scaling is only half the base problem size of
    strong scaling.

    Parameters
    ----------
    log_n_samples : int
        The common logarithm of the number of samples to use.
    log_n_features : int
        The common logarithm of the number of features to use.
    n_trees : int
        The number of trees to use. For weak scaling, this number serves as a baseline and is scaled up with the number
        of nodes.
    scaling_type : str
        The scaling type. Either 'strong' or 'weak'.
    output_path : pathlib.Path
        The path to save the generated job scripts.
    submit : bool, optional
        Whether to submit jobs to the cluster. Default is False.
    """
    for n_nodes in nodes:
        # ---- Weak scaling ----
        if scaling_type == "weak":
            n_trees = (
                n_trees * n_nodes
            )  # Number of trees is scaled with number of nodes.
            time = 3600  # All experiments should take approx. the same time (in min).
            mem = 243200  # Use standard nodes.
        # ---- Strong scaling ----
        elif scaling_type == "strong":
            n_trees = n_trees  # Number of trees stays the same.
            time = int(
                limit / n_nodes * 1.2
            )  # Run time should decrease with increasing number of nodes.
            mem = (
                486400 if n_nodes != 64 else 243200
            )  # Use high-memory nodes (except for 64-node experiment).
        else:
            raise ValueError(
                f"Unknown scaling type: {scaling_type}. Can only be 'weak' or 'strong'."
            )

        print(
            f"Current config uses {n_nodes} nodes and {n_trees} trees. Wall-clock time is {time / 60}h."
        )

        job_name = f"n{log_n_samples}_m{log_n_features}_{scaling_type}_{n_nodes}"
        job_script_name = f"{job_name}.sh"
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}  # Job name
#SBATCH --partition=cpuonly    # Queue for resource allocation
#SBATCH --time={time}          # Wall-clock time limit
#SBATCH --mem={mem}mb                # Main memory (use standard nodes)
#SBATCH --cpus-per-task=76            # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --nodes={n_nodes}             # Number of nodes
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

RESDIR="${{BASE_DIR}}"/results/{scaling_type}_scaling/n6_m4_nodes_${{SLURM_NPROCS}}_${{SLURM_JOB_ID}}/
mkdir "${{RESDIR}}"
cd "${{RESDIR}}" || exit

srun python -u ${{PYDIR}}/${{SCRIPT}} \\
    --n_samples ${{N_SAMPLES}} \\
    --n_features ${{N_FEATURES}} \\
    --n_trees ${{N_TREES}} \\
    --output_dir ${{RESDIR}} \\
    --output_label ${{SLURM_JOB_ID}} \\
    --detailed_evaluation \\
    --save_model
                                """

        with open(output_path / job_script_name, "wt") as f:
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
        "--n_trees",
        type=int,
        help="The number of trees to use. Overall number for strong scaling and baseline number for weak scaling.",
    )

    parser.add_argument(
        "--scaling_type",
        type=str,
        choices=["strong", "weak"],
        help="The scaling type. Either 'strong' or 'weak'.",
    )
    parser.add_argument(
        "--output_path",
        type=pathlib.Path,
        default=pathlib.Path("./"),
        help="The path to save the generated scripts.",
    )
    args = parser.parse_args()
    # Generate job scripts and possibly submit them to the cluster.
    generate_scaling_job_scripts(
        args.log_n_samples,
        args.log_n_features,
        args.n_trees,
        args.scaling_type,
        args.output_path,
        args.submit,
    )
