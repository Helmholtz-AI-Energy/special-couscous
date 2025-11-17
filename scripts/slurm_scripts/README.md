![Logo](../../docs/special_couscous.png)
# Example Slurm Batch Scripts
This directory contains [slurm](https://slurm.schedmd.com/) batch scripts to submit random forest jobs to the [HoreKa](https://www.scc.kit.edu/en/services/horeka.php) computing system.
They are unlikely to transfer directly to other systems but are provided here as an example to adjust to your system.
- `example_batch_script_horeka.sh` is an example for the 16 node strong scaling run on the HIGGS dataset. It expects to be submitted from the `special-couscous` root directory.
- `generate_batch_scripts_horeka.py` generates the batch scripts used for the experiments in our paper.
