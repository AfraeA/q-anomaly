#!/bin/bash

# Specify hyperparameters
runs=5
methods=("qIT") # ("cRBF" "qIT" "qRM" "qVS" "qDISC" "qBBF")
pcs=6
qIT_shots=1000
qRM_shots=9000
qRM_settings=30
qVS_subsamples=2
qVS_maxsize=100
for method in ${methods[@]}; do
    for seed in $(seq 3 $(($runs-1))); do
        for pc in $(seq 6 $pcs); do
            ./Code/job-local.sh $method $pc $seed $qIT_shots $qRM_shots $qRM_settings $qVS_subsamples $qVS_maxsize
        done
    done
done