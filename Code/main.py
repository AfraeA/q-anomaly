import argparse
import numpy as np

from Preprocessing import import_data, split_dataset, preprocess_data
from Training import train_model, save_model
from Testing import test_model, save_results


parser = argparse.ArgumentParser(description='Processing experiment specification.')
parser.add_argument('--kmethod', choices=["cRBF", "qIT", "qRM", "qVS", "qDISC", "qBBF"], required=True, help='Method to calculate the kernel matrix.')
parser.add_argument('--n_pc', type=int, required=True, help='Number of principal components or qubits.')
parser.add_argument('--seed', type=int, required=True, help='Seed value to initialize random number generation.')
parser.add_argument('--size', type=int, default=500, help='Size of the dataset sample.')
parser.add_argument('--n_shots', type=int, default=1000, help='Number of shots when measuring the quantum circuit.')
parser.add_argument('--n_settings', type=int, help='Number of basis rotations to be used for randomized measurements.')
parser.add_argument('--n_subsamples', type=int, help='Number of dataset subsamples for variable subsampling.')
args = parser.parse_args()

np.random.seed(args.seed)

quantum = args.kmethod.startswith('q')

# Download data from OpenML
data_url = 'https://www.openml.org/data/download/21756045/dataset'
data_file = './Data/dataset.csv'
data = import_data(data_url, data_file)

# Splitting and preprocessing data
train_data, test_data = split_dataset(data, args.size)
X_train, y_train, pca = preprocess_data(train_data, args.n_pc, train_split=True)
X_test, y_test, _ = preprocess_data(test_data, args.n_pc, pca=pca)

# Training and collecting metrics on test set
ocsvm, train_time = train_model(X_train, y_train, kmethod=args.kmethod, n_settings=args.n_settings, \
                n_shots=args.n_shots, n_subsamples=args.n_subsamples)
avgPrecision, precision, recall, f1_score, auroc, test_time = test_model(ocsvm, X_test, y_test, kmethod=args.kmethod)

# Saving the model
save_model(ocsvm, args.kmethod, args.seed, args.size, args.n_pc, \
                n_settings=args.n_settings, n_shots=args.n_shots, n_subsamples=args.n_subsamples)
# Saving the results of model
save_results(args.kmethod, args.seed, args.size, args.n_pc, \
                avgPrecision, precision, recall, f1_score, \
                auroc, train_time, test_time, args.n_settings, \
                args.n_shots, args.n_subsamples)