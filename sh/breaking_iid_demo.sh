#!/bin/bash

N_RANKS=4

N_CLASSES=25
N_SAMPLES=100000
MU_DATA=5
MU_PART=5
EXPERIMENT_ID="breaking_iid_demo"
SCRIPT="../py/rf_training_breaking_iid.py"

BASE_SCRIPT="mpirun -n ${N_RANKS} python ${SCRIPT} -c ${N_CLASSES} --num_samples ${N_SAMPLES} --peak $((N_CLASSES / 2)) --experiment_id ${EXPERIMENT_ID}"

GLOBAL_IMB="--globally_imbalanced --mu_data ${MU_DATA}"
LOCAL_IMB="--locally_imbalanced --mu_partition ${MU_PART}"


echo "${BASE_SCRIPT} --output_label global-bal__local-bal"
eval "${BASE_SCRIPT} --output_label global-bal__local-bal"
echo "${BASE_SCRIPT} --output_label global-imb__local-bal ${GLOBAL_IMB}"
eval "${BASE_SCRIPT} --output_label global-imb__local-bal ${GLOBAL_IMB}"
echo "${BASE_SCRIPT} --output_label global-bal__local-imb ${LOCAL_IMB}"
eval "${BASE_SCRIPT} --output_label global-bal__local-imb ${LOCAL_IMB}"
echo "${BASE_SCRIPT} --output_label global-imb__local-imb ${GLOBAL_IMB} ${LOCAL_IMB}"
eval "${BASE_SCRIPT} --output_label global-imb__local-imb ${GLOBAL_IMB} ${LOCAL_IMB}"
