#!/bin/bash

N_RANKS=4  # Number of ranks

N_CLASSES=25  # Number of classes
N_SAMPLES=100000  # Number of samples
N_FEATURES=1000  # Number of features
MU_DATA=5  # Spread of Skellam distribution for global imbalance
MU_PART=5  # Spread of Skellam distribution for local imbalance

N_TREES=1000  # Number of trees

EXPERIMENT_ID="breaking_iid_demo"
SCRIPT="../py/rf_training_breaking_iid.py"

BASE_SCRIPT="mpirun -n ${N_RANKS} python ${SCRIPT} --n_classes ${N_CLASSES} --n_samples ${N_SAMPLES} --n_features ${N_FEATURES} --peak $((N_CLASSES / 2)) --n_trees ${N_TREES} --experiment_id ${EXPERIMENT_ID}"

GLOBAL_IMB="--globally_imbalanced --mu_data ${MU_DATA}"  # Option for globally imbalanced dataset
LOCAL_IMB="--locally_imbalanced --mu_partition ${MU_PART}"  # Option for locally imbalanced dataset

# Globally and locally balanced
echo "${BASE_SCRIPT} --output_label global-bal__local-bal"
eval "${BASE_SCRIPT} --output_label global-bal__local-bal"
# Globally imbalanced and locally balanced
echo "${BASE_SCRIPT} --output_label global-imb__local-bal ${GLOBAL_IMB}"
eval "${BASE_SCRIPT} --output_label global-imb__local-bal ${GLOBAL_IMB}"
# Globally balanced and locally imbalanced
echo "${BASE_SCRIPT} --output_label global-bal__local-imb ${LOCAL_IMB}"
eval "${BASE_SCRIPT} --output_label global-bal__local-imb ${LOCAL_IMB}"
# Globally and locally imbalanced
echo "${BASE_SCRIPT} --output_label global-imb__local-imb ${GLOBAL_IMB} ${LOCAL_IMB}"
eval "${BASE_SCRIPT} --output_label global-imb__local-imb ${GLOBAL_IMB} ${LOCAL_IMB}"
