import subprocess

import numpy as np

from specialcouscous.utils import get_problem_size

n_samples = [1000000, 10000000, 100000000, 1000000000]
n_features = [100, 1000, 10000, 100000]
n_trees = [100, 1000, 10000, 100000]

limit = 60 * 24 * 3


def main() -> None:
    """Generate single-node job scripts."""
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

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export PYDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous/specialcouscous

ml purge                          # Unload all currently loaded modules.
ml load compiler/gnu              # Load required modules.
ml load mpi/openmpi
source /hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous-venv/bin/activate  # Activate venv.

N_SAMPLES={n}
N_FEATURES={m}
N_TREES={t}

SCRIPT="RF_serial_synthetic.py"

RESDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/results/single_node_experiments/job_${{SLURM_JOB_ID}}_{job_name}/
mkdir ${{RESDIR}}
cd ${{RESDIR}}

python -u ${{PYDIR}}/${{SCRIPT}} --n_samples ${{N_SAMPLES}} --n_features ${{N_FEATURES}} --n_trees ${{N_TREES}}
                                """

                with open(job_script_name, "wt") as f:
                    f.write(scriptcontent)

                subprocess.run(f"sbatch {job_script_name}", shell=True)


if __name__ == "__main__":
    main()
