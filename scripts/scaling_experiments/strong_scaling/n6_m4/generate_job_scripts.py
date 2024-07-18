import subprocess

limit = 60 * 24 * 3  # HoreKa wall-clock time limit in minutes
nodes = [2, 4, 8, 16, 32, 64]  # number of nodes for scaling exps


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
#SBATCH --job-name={job_name}          # Job name
#SBATCH --partition=cpuonly            # Queue for resource allocation
#SBATCH --time={time}                 # Wall-clock time limit
#SBATCH --cpus-per-task=76            # Number of CPUs required per (MPI) task
#SBATCH --mail-type=ALL               # Notify user by email when certain event types occur.
#SBATCH --account=hk-project-test-aihero2
#SBATCH --nodes={num_nodes}           # Number of nodes
#SBATCH --ntasks-per-node=1           # One MPI rank per node.


export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export PYDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous/specialcouscous

ml purge                          # Unload all currently loaded modules.
ml load compiler/gnu              # Load required modules.
ml load mpi/openmpi
source /hkfs/work/workspace/scratch/ku4408-special-couscous/special-couscous-venv/bin/activate  # Activate venv.

N_SAMPLES=1000000
N_FEATURES=10000
N_TREES=1600  # We estimated 1500 trees should be trainable in serial in 3 d and chose the closest number evenly divisble by 64 as a baseline for scaling exps.

SCRIPT="rf_scaling_synthetic.py"

RESDIR=/hkfs/work/workspace/scratch/ku4408-special-couscous/results/strong_scaling/n6_m4_nodes_${{SLURM_NPROCS}}_job_${{SLURM_JOB_ID}}/
mkdir ${{RESDIR}}
cd ${{RESDIR}}

srun python -u ${{PYDIR}}/${{SCRIPT}} --n_samples ${{N_SAMPLES}} --n_features ${{N_FEATURES}} --n_trees ${{N_TREES}} --output_dir ${{RESDIR}} --output_label ${{SLURM_JOB_ID}} --detailed_evaluation
                                """

        with open(job_script_name, "wt") as f:
            f.write(scriptcontent)

        subprocess.run(f"sbatch {job_script_name}", shell=True)


if __name__ == "__main__":
    main()
