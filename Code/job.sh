#!/bin/bash 

#SBATCH --mail-user=ahouzi.afrae@ifi.lmu.de
#SBATCH --mail-type=All
#SBATCH --partition=All
#SBATCH --export=NONE

# Make sure that your virtual environment has all required packages installed
pip install -r ./requirements.txt
echo "Activating Python Virtual Environment"
source ../env/bin/activate

echo "Executing job method=$1 n_pc=$2 seed=$3"
srun python3 ./Code/main.py $@