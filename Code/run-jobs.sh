#!/bin/bash

# Specify hyperparameters
runs=15
methods=("qIT" "qRM") # ("cRBF" "qIT" "qRM" "qVS" "qDISC" "qBBF")
pcs=20
for method in ${methods[@]}; do
    for seed in $(seq 0 $(($runs-1))); do
        for pc in $(seq 1 $pcs); do
            if [$method=="qRM"] then (n_shots=8000 n_settings=8) else (n_shots=1000) fi
            sbatch --job-name="run-$method-$pc-$seed" job.sh --kmethod=$method --n_pc=$pc --seed=$seed --n_shots=$n_shots --n_rotations=$n_rotations
        done
    done
done