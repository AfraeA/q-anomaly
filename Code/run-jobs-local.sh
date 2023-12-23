#!/bin/bash

# Specify hyperparameters
runs=15
methods=("cRBF" "qIT" "qRM" "qVS")
pcs=20
qIT_shots=1000
qRM_shots=9000
qRM_settings=30
qVS_subsamples=5
qVS_maxsize=100
for method in ${methods[@]}; do
    for seed in $(seq 0 $(($runs-1))); do
        for pc in $(seq 1 $pcs); do
            ./Code/job-local.sh $method $pc $seed $qIT_shots $qRM_shots $qRM_settings $qVS_subsamples $qVS_maxsize
        done
    done
done