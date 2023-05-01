import time
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, f1_score
# impot numpy as np
# from scipy.special import kl_div

ROOT_DIR = os.path.dirname(os.path.abspath('./__file__'))
# TODO: Add quantum related metrics to train_model and test_model
'''
def get_circuit_probability(circuit):
    # TODO: Read papers and github repos
    # https://github.com/bagmk/Quantum_Machine_Learning_Express
    # https://github.com/Pratha-Me/Expressivity-Quantum-Data-encoding-Fourier-Series
    # https://github.com/XanaduAI/expressive_power_of_quantum_models
    # We sample thetas and inputs uniformly
    thetas = np.random.uniform(-np.pi, np.pi, 1000)
    inputs = np.random.uniform(-1,1, 1000)
    # We use a histogram with discretization of 100 bins
    probability = np.historgram(circuit(theta, input))
    return circuit_probability
def get_sim_expressivity(ocsvm, d, F):
    p_model = get_circuit_probability(circuit)
    p_Haar = (d-1) * (1-F)**(d-2)
    sim_expressivity = kl_div (p_model, p_Haar)
    return sim_expressivity
def get_fourier_expressivity():
    return fourier_frequencies, fourier_coeffs
def get_entanglement_capacity():
    return entanglement_capacity
'''
def test_model(ocsvm, X_test, y_test, kmethod):
    print('Gathering performance metrics...')
    if kmethod.startswith('q'):
        X_test = 0.1 * X_test # Rescale quantum data to be < 2pi
    start_test_time = time.time()
    predictions = ocsvm.predict(X_test)
    end_test_time = time.time()
    testDuration = end_test_time - start_test_time
    
    predictions = pd.Series(predictions).replace([-1,1],[1,0])
    avgPrecision = average_precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1score = f1_score(y_test, predictions)
    auroc = roc_auc_score(y_test, predictions)
    
    return avgPrecision, precision, recall, f1score, auroc, testDuration

def save_results(kmethod, seed, size, n_pc, avgPrecision, precision, \
                recall, f1_score, auroc, train_time, test_time, \
                n_rotations=None, n_shots=None, n_subsamples=None):
    print('Saving results to Results folder...')
    resultsFolderName = f'{ROOT_DIR}/Results/'
    if not os.path.exists(resultsFolderName):
        os.mkdir(resultsFolderName)
    kmethodFolderName = resultsFolderName + f'{kmethod}/'
    if not os.path.exists(kmethodFolderName):
        os.mkdir(kmethodFolderName)
    resultFileName = kmethodFolderName + f'dsize_{size}'
    if kmethod.startswith('q'):
        resultFileName += f'_n_shots_{n_shots}'
        if kmethod == 'qRM':
            resultFileName += f'_n_rotations_{n_rotations}'
        elif kmethod == 'qVS':
            resultFileName += f'_n_subsamples_{n_subsamples}'
        else:
            #TODO: add handling for hyperparameters of qDISC and qBBF
            pass
    resultFileName += '.csv'
    # TODO: Seed has to eventually be a result dataset attribute, to simplify plotting
    if not os.path.exists(resultFileName):
        with open(resultFileName, 'w+') as resultFile:
            header = 'seed, n_pc, avgPrecision, precision, recall, f1_score, auroc, train_time, test_time\n'
            resultFile.write(header)
    with open(resultFileName, 'a+') as resultFile:
        resultFormat = '{0}, {1}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}, {6:.4f}, {7:.4f}, {8:.4f}\n'
        resultFile.write(resultFormat.format(seed, n_pc, avgPrecision, precision, \
                                             recall, f1_score, auroc, train_time, test_time))