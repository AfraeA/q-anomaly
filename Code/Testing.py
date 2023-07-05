import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, f1_score
from Kernel_calculation import retrieve_interim_kernel_calculation_time

ROOT_DIR = os.path.dirname(os.path.abspath('./__file__'))

def get_qVS_predictions(model, X_test):
     # We get the decision function for datapoint in the test set using all the components
    predictions_per_component = np.vstack([component.decision_function(X_test) for component in model])
    # We normalise the outlier scores and average them, then extract the label using the sign function
    predictions = np.sign(((predictions_per_component - predictions_per_component.mean())/predictions_per_component.std()).max(axis=0))
    return predictions

def test_model(model, X_test, y_test, seed, kmethod, qIT_shots=None, \
                qRM_shots=None, qRM_settings=None, qVS_subsamples=None, qVS_maxsize=None):
    print('Gathering performance metrics...')
    previous_t = retrieve_interim_kernel_calculation_time(kmethod, len(X_test), 'test', seed, len(X_test[0]), \
                                    qIT_shots=qIT_shots, qRM_shots=qRM_shots, qRM_settings=qRM_settings, \
                                    qVS_subsamples=qVS_subsamples, qVS_maxsize=qVS_maxsize)
    start_test_time = time.time()
    if kmethod == 'qVS':
       predictions = get_qVS_predictions(model, X_test)
    else:
        predictions = model.predict(X_test)
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
                qRM_shots=None, qRM_settings=None, qVS_subsamples=None, qVS_maxsize=None):
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
            resultFileName += f'_n_subsamples_{qVS_subsamples}_maxsize_{qVS_maxsize}'
    resultFileName += '.csv'
    if not os.path.exists(resultFileName):
        with open(resultFileName, 'w+') as resultFile:
            header = 'seed,n_pc,avgPrecision,precision,recall,f1_score,auroc,train_time,test_time\n'
            resultFile.write(header)
    with open(resultFileName, 'a+') as resultFile:
        resultFormat = '{0},{1},{2:.4f},{3:.4f},{4:.4f},{5:.4f},{6:.4f},{7:.4f},{8:.4f}\n'
        resultFile.write(resultFormat.format(seed, n_pc, avgPrecision, precision, \
                                             recall, f1_score, auroc, train_time, test_time))
        
def check_tested(kmethod, seed, size, n_pc, qIT_shots=None, \
                qRM_shots=None, qRM_settings=None, qVS_subsamples=None, qVS_maxsize=None):
    '''
    Checks if model was already trained and tested.
    '''
    resultsFolderName = f'{ROOT_DIR}/Results/'
    kmethodFolderName = resultsFolderName + f'{kmethod}/'
    resultFileName = kmethodFolderName + f'dsize_{size}'
    if kmethod == 'qIT':
        resultFileName += f'_n_shots_{qIT_shots}'
    elif kmethod == 'qRM':
        resultFileName += f'_n_shots_{qRM_shots}_n_settings_{qRM_settings}'
    elif kmethod == 'qVS':
            resultFileName += f'_n_subsamples_{qVS_subsamples}_maxsize_{qVS_maxsize}'
    else:
        #TODO: add handling for hyperparameters of qDISC and qBBF
        pass
    resultFileName += '.csv'
    if not os.path.exists(resultFileName):
        return False
    for _ in range(20):
        try:
            result_df = pd.read_csv(resultFileName, on_bad_lines='warn')
        except (KeyError, pd.errors.EmptyDataError):
            time.sleep(np.random.uniform(0,30))
            continue
        break
    if result_df[(result_df.seed==seed) & (result_df.n_pc==n_pc)].empty:
        return False
    else:
        return True
