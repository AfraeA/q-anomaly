# Efficient Quantum Anomaly Detection

The code provided was used to generate the results of "Towards Efficient Quantum Anomaly Detection: One-Class SVMs using Variable Subsampling and Randomized Measurements" https://arxiv.org/abs/2312.09174


## Setup Instructions

Clone this repository
```bash
$ git clone https://github.com/AfraeA/q-anomaly.git
```

Create a venv environment in the main directory:
```bash
$ python -m venv env
```
Activate the environment (Linux):
```bash
$ source ./env/bin/activate
```
Or on Windows:
```bash
$ env\Scripts\activate.bat
```

Install environment:
```bash
$ pip install -r requirements.txt
```

## Structure

```
.
├── README.md                           - This file
├── Code
│   ├── main.py                         - Trains and tests a model with specific params
│   ├── Kernel_calculation.py           - Calculates gram matrices w/ quantum circuits
│   ├── Preprocessing.py                - Preprocesses Credit Card data (CC)
│   ├── Testing.py                      - Tests model & returns its performance metrics
│   ├── Training.py                     - Function for training and saving models
│   ├── run-jobs-local.sh               - Runs all CC data experiments w/ SLURM
│   ├── run-jobs.sh                     - Runs all CC data experiments locally
│   ├── job-local.sh                    - Called by run-jobs-local.sh
│   ├── job.sh                          - Called by run-jobs.sh
|   └── Visualization.ipynb             - Functions used for visualizing results
├── Data                                - Created during experiments
├── Models                              - Models trained on the CC data
├── Models_synthetic                    - Models trained on the synthetic data
├── InterimResults                      - Intermediary kernel matrices and run times
├── Results                             - Results for the CC data
├── Results_synthetic                   - Results for the synthetic data
├── Plots                               - Created after running "Visualization.ipynb"
└── requirements.txt                    - Required pip packages

```

## Reproducing experiments
To locally reproduce the experiments (by features) with CC data from base directory:
```bash
$ ./Code/run-jobs-local.sh
```
The models using the quantum feature embedding might take multiple days to run. To speed that up, the InterimResults folder can be requested and used to obtain results.
