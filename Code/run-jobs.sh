#!/bin/zsh

# Specify hyperparameters
runs=15
methods=("qIT")
pcs=5
qIT_shots=1000
qRM_shots=16000
qRM_settings=16
qVS_subsamples=5
qVS_maxsize=100

# Check if sbatch is available
if command -v sbatch > /dev/null; then
  sbatch_cmd="sbatch"
else
  echo "\033[1;31mYou are not in a slurm environment. Executing experiments sequentially!\033[0m"
  sbatch_cmd=""
fi

for method in ${methods[@]}; do
    for seed in $(seq 10 10); do
        for pc in $(seq 4 $pcs); do
            if [ -z "$sbatch_cmd" ]; then
                echo "\033[1;32mExecuting job with environment variables:\033[0m $method $pc $seed"
                ./Code/job.sh $method $pc $seed $qIT_shots $qRM_shots $qRM_settings $qVS_subsamples
            else
                $sbatch_cmd --job-name="run-$method-$pc-$seed" ./Code/job.sh $method $pc $seed $qIT_shots $qRM_shots $qRM_settings $qVS_subsamples $qVS_maxsize
            fi
        done
    done
done