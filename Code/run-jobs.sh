#!/bin/bash

# Specify hyperparameters
runs=15
methods=("qIT" "qRM") # ("cRBF" "qIT" "qRM" "qVS" "qDISC" "qBBF")
pcs=20
qIT_shots=1000
qRM_shots=8000
qRM_settings=16
qVS_subsamples=10
for method in ${methods[@]}; do
    for seed in $(seq 0 $(($runs-1))); do
        for pc in $(seq 3 $pcs); do
            sbatch --job-name="run-$method-$pc-$seed" job.sh $method $pc $seed $qIT_shots $qRM_shots $qRM_settings $qVS_subsamples
        done
    done
done