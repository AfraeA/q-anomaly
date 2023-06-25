import argparse
import numpy as np

from Preprocessing import import_data, split_dataset, preprocess_data
from Training import train_model, save_model
from Testing import test_model, save_results, check_tested

parser = argparse.ArgumentParser(description='Processing experiment specification.')
parser.add_argument('--kmethod', choices=["cRBF", "qIT", "qRM", "qVS", "qBBF"], required=True, help='Method to calculate the kernel matrix.')
parser.add_argument('--n_pc', type=int, required=True, help='Number of principal components or qubits.')
parser.add_argument('--seed', type=int, required=True, help='Seed value to initialize random number generation.')
parser.add_argument('--train_size', type=int, default=500, help='Size of the training dataset sample.')
parser.add_argument('--test_size', type=int, default=125, help='Size of the testing dataset sample.')
parser.add_argument('--anomaly_ratio', type=float, default=0.05, help='Ratio of anomalies in the testing dataset.')
parser.add_argument('--qIT_shots', default=1000, type=int, help='Number of shots when measuring the quantum inversion test circuit.')
parser.add_argument('--qRM_shots', type=int, default=8000, help='Number of shots when measuring the quantum randomized measurements circuit.')
parser.add_argument('--qRM_settings', type=int, default=8, help='Number of basis rotations to be used for randomized measurements.')
parser.add_argument('--qVS_subsamples', type=int, help='Number of dataset subsamples for variable subsampling.')
parser.add_argument('--qVS_maxsize', type=float, help='Maximum number of data points per subsample for variable subsampling.')
args = parser.parse_args()

np.random.seed(args.seed)

# Check if model was already trained and tested:
if not check_tested(args.kmethod, args.seed, args.train_size, args.n_pc, args.qIT_shots, \
                args.qRM_shots, args.qRM_settings, args.qVS_subsamples):
    
    # Download data from OpenML
    data_url = 'https://www.openml.org/data/download/21756045/dataset'
    data = import_data(data_url)

    # Splitting and preprocessing data
    train_data, test_data = split_dataset(data, args.train_size, args.test_size, args.anomaly_ratio)
    X_train, y_train, pca_scaler, pca, scaler = preprocess_data(train_data, args.kmethod, args.seed, args.n_pc, train_split=True)
    X_test, y_test, _, _, _ = preprocess_data(test_data, args.kmethod, args.seed, args.n_pc, pca_sc=pca_scaler, pca=pca, sc=scaler)

    assert np.count_nonzero(y_test), "Test data contains only genuine data points"
    assert X_train.shape[1] == args.n_pc and X_test.shape[1] == args.n_pc, "Dataset was preprocessed incorrectly"

    # Training and collecting metrics on test set
    ocsvm, train_time = train_model(X_train, y_train, seed=args.seed, kmethod=args.kmethod, qIT_shots=args.qIT_shots, \
                    qRM_shots=args.qRM_shots, qRM_settings=args.qRM_settings, \
                    qVS_subsamples=args.qVS_subsamples, qVS_maxsize=args.qVS_maxsize)

    # Saving the model
    save_model(ocsvm, args.kmethod, args.seed, args.train_size, args.n_pc, qIT_shots=args.qIT_shots, \
                    qRM_shots=args.qRM_shots, qRM_settings=args.qRM_settings,  \
                    qVS_subsamples=args.qVS_subsamples, qVS_maxsize=args.qVS_maxsize)

    # Gathering metrics
    avgPrecision, precision, recall, f1_score, auroc, test_time = test_model(ocsvm, X_test, y_test, seed=args.seed, \
                    kmethod=args.kmethod, qIT_shots=args.qIT_shots, qRM_shots=args.qRM_shots, \
                    qRM_settings=args.qRM_settings, qVS_subsamples=args.qVS_subsamples, qVS_maxsize=args.qVS_maxsize)

    # Saving the results of model
    save_results(args.kmethod, args.seed, args.train_size, args.n_pc, \
                    avgPrecision, precision, recall, f1_score, \
                    auroc, train_time, test_time, args.qIT_shots, args.qRM_shots, \
                    args.qRM_settings, args.qVS_subsamples, args.qVS_maxsize)