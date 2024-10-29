#!/bin/sh
# ACC. DROP
# Robust seeds
python acc_drop.py ../../../../results/acc_drop/robust_seeds/n5_m3
python acc_drop.py ../../../../results/acc_drop/robust_seeds/n6_m2
# Rank-based seeds
python acc_drop.py ../../../../results/acc_drop/rank-based_seeds/n5_m3
python acc_drop.py ../../../../results/acc_drop/rank-based_seeds/n6_m2

# INFERENCE FLAVOR COMPARISON
# Wall times + energy comparison
python inference_flavor.py ../../../../results/inference_flavor/no_shared_global_model/n6_m2 \
../../../../results/inference_flavor/shared_global_model/n6_m2
python inference_flavor.py ../../../../results/inference_flavor/no_shared_global_model/n5_m3 \
../../../../results/inference_flavor/shared_global_model/n5_m3
# Weak scaling comparison
python weak_scaling_inference_flavor.py ../../../../results/inference_flavor/no_shared_global_model/n5_m3 \
../../../../results/inference_flavor/shared_global_model/n5_m3
python weak_scaling_inference_flavor.py ../../../../results/inference_flavor/no_shared_global_model/n6_m2 \
../../../../results/inference_flavor/shared_global_model/n6_m2
# Individual weak scaling
python weak_scaling.py ../../../../results/inference_flavor/no_shared_global_model/n6_m2
python weak_scaling.py ../../../../results/inference_flavor/no_shared_global_model/n5_m3
python weak_scaling.py ../../../../results/inference_flavor/shared_global_model/n6_m2
python weak_scaling.py ../../../../results/inference_flavor/shared_global_model/n5_m3

# STRONG SCALING
python strong_scaling.py ../../../../results/strong_scaling/n6_m4
python strong_scaling.py ../../../../results/strong_scaling/n7_m3

# WEAK SCALING
python weak_scaling_from_checkpoints.py ../../../../results/weak_scaling/n6_m4
python weak_scaling_from_checkpoints.py ../../../../results/weak_scaling/n7_m3

# CHUNKING
python chunking.py ../../../../results/chunking/n6_m4
python chunking.py ../../../../results/chunking/n7_m3
