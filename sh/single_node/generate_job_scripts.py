import argparse
import subprocess

import numpy as np

# Grid search space for initial single-node capacity experiments
n_samples = [1000000, 10000000, 100000000, 1000000000]  # Number of samples in datasets
n_features = [100, 1000, 10000, 100000]  # Number of features in datasets
n_trees = [
    100,
    1000,
    10000,
    100000,
]  # Number of trees in random forest classifiers to train

limit = 60 * 24 * 3  # 3 day wall-clock time limit of local supercomputer


def get_problem_size(n: int, m: int, t: int) -> float:
    """
    Get problem size of random forest training in terms of time complexity.

    Parameters
    ----------
    n : int
        Number of samples in training dataset
    m : int
        Number of features in training dataset
    t : int
        Number of trees to train

    Returns
    -------
    float
        Time complexity for training random forest on this dataset
    """
    return n * np.log2(n) * np.sqrt(m) * t


def main():
    parser = argparse.ArgumentParser(
        prog="Generate Job Scripts for Single-Node Capacity Experiments"
    )

    parser.add_argument(
        "--workspace_dir",
        type=str,
        help="The absolute path to the workspace directory.",
    )

    args = parser.parse_args()

    base_problem_size = get_problem_size(n_samples[0], n_features[0], n_trees[0])
    base_time = 30

    for n in n_samples:
        for m in n_features:
            for t in n_trees:
                time = int(
                    base_time * get_problem_size(n, m, t) / base_problem_size + 10
                )

                if time > limit:
                    time = limit
                print(f"Current config: (n, m, t) = ({n}, {m}, {t})")
                job_name = (
                    f"n{int(np.log10(n))}_m{int(np.log10(m))}_t{int(np.log10(t))}"
                )
                job_script_name = f"{job_name}.sh"
                scriptcontent = f"""#!/bin/bash
#SBATCH --job-name={job_name}          # job name
#SBATCH --partition=cpuonly            # queue for resource allocation
#SBATCH --mem=501600mb                 
#SBATCH --time={time}                 # wall-clock time limit
#SBATCH --cpus-per-task=76            # number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --account=hk-project-test-aihero2

# Overwrite base directory by running export BASE_DIR="/some/alternative/path/here" before submitting the job.
BASE_DIR=${{BASE_DIR:-{args.workspace_dir}/}}

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export PYDIR=${{BASE_DIR}}/special-couscous/specialcouscous

ml purge                          # Unload all currently loaded modules.
ml load compiler/gnu              # Load required modules.
ml load mpi/openmpi
source "${{BASE_DIR}}"/special-couscous-venv/bin/activate  # Activate venv.

N_SAMPLES={n}
N_FEATURES={m}
N_TREES={t}

SCRIPT="RF_serial_synthetic.py"

RESDIR=${{BASE_DIR}}/results/single_node_experiments/{job_name}_${{SLURM_JOB_ID}}/
mkdir "${{RESDIR}}"
cd "${{RESDIR}}" || exit

python -u "${{PYDIR}}"/"${{SCRIPT}}" --n_samples ${{N_SAMPLES}} --n_features ${{N_FEATURES}} --n_trees ${{N_TREES}}
                                """

                with open(job_script_name, "wt") as f:
                    f.write(scriptcontent)

                subprocess.run(f"sbatch {job_script_name}", shell=True)


if __name__ == "__main__":
    main()
