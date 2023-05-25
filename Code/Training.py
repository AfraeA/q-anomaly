import time
import os
from sklearn.svm import OneClassSVM
from Kernel_calculation import get_kernel_matrix_qIT, \
    get_kernel_matrix_qRM, retrieve_interim_kernel_calculation_time
from joblib import dump
from functools import partial

ROOT_DIR = os.path.dirname(os.path.abspath('./__file__'))

def train_model(X_train, y_train, seed, kmethod, qIT_shots, qRM_shots, qRM_settings, qVS_subsamples):
    print('Training OCSVM...')
    if kmethod == 'cRBF':
        ocsvm = OneClassSVM(gamma='scale', nu=0.1, cache_size=2000)
    elif kmethod == 'qIT':
        #X_train = 0.1 * X_train # Rescale quantum data to be < 2pi
        get_kernel_matrix = partial(get_kernel_matrix_qIT, seed=seed, n_shots=qIT_shots)
        ocsvm = OneClassSVM(kernel=get_kernel_matrix, nu=0.1, cache_size=2000)
    elif kmethod == 'qRM':
        #X_train = 0.1 * X_train
        get_kernel_matrix = partial(get_kernel_matrix_qRM, seed=seed, n_shots=qRM_shots, n_settings=qRM_settings)
        ocsvm = OneClassSVM(kernel=get_kernel_matrix, nu=0.1, cache_size=2000)
    elif kmethod == 'qVS':
        #X_train = 0.1 * X_train
        raise(NotImplementedError, "Code for training with qVS still has not been implemented.")
    previous_t = retrieve_interim_kernel_calculation_time(kmethod, len(X_train), 'train', seed, \
                                    len(X_train[0]), qIT_shots, qRM_shots, qRM_settings, qVS_subsamples)
    start_train_time = time.time()
    ocsvm = ocsvm.fit(X_train)
    end_train_time = time.time()
    train_duration = end_train_time - start_train_time + previous_t
    return ocsvm, train_duration

def save_model(ocsvm, kmethod, seed, size, n_pc, qIT_shots=None, \
                qRM_shots=None, qRM_settings=None, qVS_subsamples=None):
    '''
    Saves model into the models folder
    To reload the ocsvm, use:  
    from joblib import load
    ocsvm = load(modelFileName) 
    '''
    # Create a models folder
    print('Saving OCSVM...')
    modelsFolderName = f'{ROOT_DIR}/Models/'
    if not os.path.exists(modelsFolderName):
        os.mkdir(modelsFolderName)
    kmethodFolderName = modelsFolderName + f'{kmethod}/'
    if not os.path.exists(kmethodFolderName):
        os.mkdir(kmethodFolderName)
    modelFileName = kmethodFolderName + f'dsize_{size}_seed_{seed}_n_pc_{n_pc}'
    if kmethod == 'qIT':
        modelFileName += f'_n_shots_{qIT_shots}'
    elif kmethod == 'qRM':
        modelFileName += f'_n_shots_{qRM_shots}_n_settings_{qRM_settings}'
    elif kmethod == 'qVS':
        modelFileName += f'_n_subsamples_{qVS_subsamples}'
    else:
        #TODO: add handling for hyperparameters of qDISC and qBBF
        pass
    modelFileName += f'.joblib'
    dump(ocsvm, modelFileName)