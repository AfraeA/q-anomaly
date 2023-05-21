import time
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, f1_score
from Kernel_calculation import retrieve_interim_kernel_calculation_time

ROOT_DIR = os.path.dirname(os.path.abspath('./__file__'))

def test_model(ocsvm, X_test, y_test, seed, kmethod, qIT_shots=None, \
                qRM_shots=None, qRM_settings=None, qVS_subsamples=None):
    print('Gathering performance metrics...')
    if kmethod.startswith('q'):
        X_test = 0.1 * X_test # Rescale quantum data to be < 2pi
    previous_t = retrieve_interim_kernel_calculation_time(kmethod, len(X_test), 'test', seed, \
                                    len(X_test[0]), qIT_shots, qRM_shots, qRM_settings, qVS_subsamples)
    start_test_time = time.time()
    predictions = ocsvm.predict(X_test)
    end_test_time = time.time()
    testDuration = end_test_time - start_test_time + previous_t
    
    predictions = pd.Series(predictions).replace([-1,1],[1,0])
    avgPrecision = average_precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1score = f1_score(y_test, predictions)
    auroc = roc_auc_score(y_test, predictions)
    
    return avgPrecision, precision, recall, f1score, auroc, testDuration

def save_results(kmethod, seed, size, n_pc, avgPrecision, precision, \
                recall, f1_score, auroc, train_time, test_time, qIT_shots=None, \
                qRM_shots=None, qRM_settings=None, n_subsamples=None):
    print('Saving results to Results folder...')
    resultsFolderName = f'{ROOT_DIR}/Results/'
    if not os.path.exists(resultsFolderName):
        os.mkdir(resultsFolderName)
    kmethodFolderName = resultsFolderName + f'{kmethod}/'
    if not os.path.exists(kmethodFolderName):
        os.mkdir(kmethodFolderName)
    resultFileName = kmethodFolderName + f'dsize_{size}'
    if kmethod == 'qIT':
        resultFileName += f'_n_shots_{qIT_shots}'
    elif kmethod == 'qRM':
        resultFileName += f'_n_shots_{qRM_shots}_n_settings_{qRM_settings}'
    elif kmethod == 'qVS':
            resultFileName += f'_n_subsamples_{n_subsamples}'
    else:
        #TODO: add handling for hyperparameters of qDISC and qBBF
        pass
    resultFileName += '.csv'
    if not os.path.exists(resultFileName):
        with open(resultFileName, 'w+') as resultFile:
            header = 'seed, n_pc, avgPrecision, precision, recall, f1_score, auroc, train_time, test_time\n'
            resultFile.write(header)
    with open(resultFileName, 'a+') as resultFile:
        resultFormat = '{0}, {1}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}, {6:.4f}, {7:.4f}, {8:.4f}\n'
        resultFile.write(resultFormat.format(seed, n_pc, avgPrecision, precision, \
                                             recall, f1_score, auroc, train_time, test_time))