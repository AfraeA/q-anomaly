import time
import os
import re
import numpy as np
from sklearn.svm import OneClassSVM
from Kernel_calculation import get_kernel_matrix_qIT, \
    get_kernel_matrix_qRM, retrieve_interim_kernel_calculation_time
from joblib import dump, load
from functools import partial

ROOT_DIR = os.path.dirname(os.path.abspath('./__file__'))

def train_VS_ensemble_model(X_train, y_train, seed, qVS_subsamples, qVS_maxsize):
    # We select a list of size qVS_subsamples of subsample sizes between 50 and qVS_maxsize
    subsample_sizes_list = np.random.choice(range(50, qVS_maxsize), replace=False, size=qVS_subsamples)
    # We uniformly sample the subsamples from the training split
    subsample_indices_list =  [np.random.choice(len(X_train), size=s, replace=False) for s in subsample_sizes_list]
    subsamples = [np.array([X_train[index] for index in subsample_indices_list[sample]]) for sample in range(qVS_subsamples)]
    get_kernel_matrix = partial(get_kernel_matrix_qIT, seed=seed, n_shots=1000, kmethod='qVS', \
                                qVS_subsamples=qVS_subsamples, qVS_maxsize=qVS_maxsize)
    # We train an OCSVM with each subsample and return the ensemble components as a list
    model = [OneClassSVM(kernel=get_kernel_matrix, nu=0.1, cache_size=2000).fit(subsample) for subsample in subsamples]
    return model

def train_model(X_train, y_train, seed, kmethod, qIT_shots, qRM_shots, qRM_settings, qVS_subsamples, qVS_maxsize):
    print('Training model...')
    if kmethod == 'cRBF':
        model = OneClassSVM(gamma='scale', nu=0.1, cache_size=2000)
    elif kmethod == 'qIT':
        get_kernel_matrix = partial(get_kernel_matrix_qIT, seed=seed, n_shots=qIT_shots)
        model = OneClassSVM(kernel=get_kernel_matrix, nu=0.1, cache_size=2000)
    elif kmethod == 'qRM':
        get_kernel_matrix = partial(get_kernel_matrix_qRM, seed=seed, n_shots=qRM_shots, n_settings=qRM_settings)
        model = OneClassSVM(kernel=get_kernel_matrix, nu=0.1, cache_size=2000)
    previous_t = retrieve_interim_kernel_calculation_time(kmethod, len(X_train), 'train', seed, len(X_train[0]), \
                                    qIT_shots=qIT_shots, qRM_shots=qRM_shots, qRM_settings=qRM_settings, \
                                    qVS_subsamples=qVS_subsamples, qVS_maxsize=qVS_maxsize)
    start_train_time = time.time()
    if kmethod == 'qVS':
        model = train_VS_ensemble_model(X_train, y_train, seed, qVS_subsamples, qVS_maxsize)
    else:
        model = model.fit(X_train)
    end_train_time = time.time()
    train_duration = end_train_time - start_train_time + previous_t
    return model, train_duration

def save_model(model, kmethod, seed, size, n_pc, qIT_shots=None, \
                qRM_shots=None, qRM_settings=None, \
                qVS_subsamples=None, qVS_maxsize=None):
    '''
    Saves model into the models folder
    To reload an individual model, use:  
    from joblib import load
    model = load(modelFileName) 
    '''
    # Create a models folder
    print('Saving model...')
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
        ensembleFolderName = modelFileName + f'_n_subsamples_{qVS_subsamples}_maxsize_{qVS_maxsize}/'
        if not os.path.exists(ensembleFolderName):
            os.mkdir(ensembleFolderName)
    if kmethod == 'qVS':
        for ocsvm in model:
            modelFileName = ensembleFolderName + f'_subsample_size_{ocsvm.shape_fit_[0]}.joblib'
            dump(model, modelFileName)
    else:
        modelFileName += f'.joblib'
        dump(model, modelFileName)
    
def retrieve_model(kmethod, seed, size, n_pc, qIT_shots=None, \
                qRM_shots=None, qRM_settings=None, \
                qVS_subsamples=None, qVS_maxsize=None):
    '''
    Retrieves trained model from models directory for further testing
    '''
    print('Retrieving model...')
    modelsFolderName = f'./Models/'
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
        ensembleFolderName = modelFileName + f'_n_subsamples_{qVS_subsamples}_maxsize_{qVS_maxsize}/'
        model = []
        # Find all model names that correspond to the regex
        regex = re.compile(f'^_subsample_size_.*.joblib$')
        if os.path.exists(ensembleFolderName):
            modelFileNamesList = [file for file in tuple(os.walk(ensembleFolderName))[0][2] if regex.match(file)]
            for filename in modelFileNamesList:
                modelFileName = ensembleFolderName + f'/{filename}'
                if os.path.exists(modelFileName):
                    ocsvm = load(modelFileName)
                    model.append(ocsvm)
        return model
    modelFileName += f'.joblib'
    model = load(modelFileName, modelFileName) if os.path.exists(modelFileName) else None
    return model     
        