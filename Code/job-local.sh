#!/bin/bash 

echo "Executing job method=$1 n_pc=$2 seed=$3"
py ./Code/main.py --kmethod=$1 --n_pc=$2 --seed=$3 --qIT_shots=$4 --qRM_shots=$5 --qRM_settings=$6 --qVS_subsamples=$7