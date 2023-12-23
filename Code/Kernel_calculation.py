import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from itertools import product, chain

#from functools import partialmethod
#tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import random_unitary

ROOT_DIR = os.path.dirname(os.path.abspath('./__file__'))

def quantum_feature(x, reuploads):
    '''
    Returns a IQP like circuit with the specified number of reuploads
    '''
    n_pc = len(x)
    qr = QuantumRegister(n_pc)
    cr = ClassicalRegister(n_pc)
    qc = QuantumCircuit(qr, cr)
    for r in range(2*reuploads):
        qc.h(range(n_pc))
        qc.barrier()
        for i in range(n_pc):
            qc.rz(x[i], qr[i])
        qc.barrier()
        for i in range(n_pc):
            for j in range(i+1, n_pc):
                qc.rzz(x[i]*x[j], i, j)
        qc.barrier()
    return qc
                
def inversion_test_circuit(x1, x2, reuploads):
    '''
    Returns the circuit for the inversion test of the data x1 and x2
    '''
    n_pc = len(x1)
    U_x1 = quantum_feature(x1, reuploads)
    U_x2 = quantum_feature(x2, reuploads)
    
    kernel_c = U_x1.compose(U_x2.inverse(), range(n_pc))
    kernel_c.measure(range(n_pc), range(n_pc))
    return kernel_c

def get_kernel_element_qIT(x1, x2, n_shots=1000):
    '''
    Returns the kernel element for the datapoints x1 and x2
    '''
    kernel_c = inversion_test_circuit(x1, x2, 3)
    simulator = QasmSimulator()
    t_circuit = transpile(kernel_c, simulator)
    job = simulator.run(t_circuit, shots=n_shots)
    results = job.result()
    counts = results.get_counts()
    prob0 = 0 if ('0'*len(x1)) not in counts.keys() else (counts[('0'*len(x1))]/n_shots)
    return prob0

def get_kernel_matrix_qIT(X1, X2, seed=None, kmethod='qIT', n_shots=1000, qVS_subsamples=None, qVS_maxsize=None):
    '''
    Returns the kernel matrix using inversion tests
    '''
    X1_size = len(X1)
    X2_size = len(X2)
    n_pc = len(X1[0])
    split = 'train' if np.array_equal(X1,X2) else 'test'
    start_t = time.time()
    # Check if retrieve_interim_kernel_copy() returns a matrix or -1.
    gram_matrix = retrieve_interim_kernel_copy(kmethod, (X1_size, X2_size), seed, n_pc, split=split, qIT_shots=n_shots, \
                                            qVS_subsamples=qVS_subsamples, qVS_maxsize=qVS_maxsize)
    if gram_matrix is not None and gram_matrix[-1, -1] > 0:
        return gram_matrix
    elif gram_matrix is None:
        # If it returns None, then initialize with zeros and start calculating from beginning
        gram_matrix = np.zeros((X1_size, X2_size))
        num_eval = X1_size*X2_size
        indices = product(range(X1_size), range(X2_size))
    else:
        # If it returns a matrix copy, then call find the coordinates of the next entry to be calculated
        next_i, next_j = find_kernel_entry_index(gram_matrix)
        assert next_i != -1, f"Expected gram_matrix to be complete, but gram_matrix is not complete."
        num_eval = (X1_size - next_i) * X2_size - next_j
        indices = chain(product([next_i], range(next_j, X2_size)), \
                        product(range(next_i+1, X1_size), range(0, X2_size)))
    end_t = time.time()
    save_interim_kernel_calculation_time(end_t-start_t, False, kmethod, (X1_size, X2_size), split, seed, n_pc, qIT_shots=n_shots, \
                                        qVS_subsamples=qVS_subsamples, qVS_maxsize=qVS_maxsize)
    progression_bar = tqdm(indices, total=num_eval)
    for i, j in progression_bar:
        start_t = time.time()
        progression_bar.set_description("Processing gram_matrix [%d][%d]" %(i, j))
        if (split == 'test') or (split == 'train' and j >= i):
            gram_matrix[i][j] = get_kernel_element_qIT(X1[i], X2[j], n_shots)
            if j == i or n_pc > 8:
                save_interim_kernel_copy(gram_matrix, kmethod, (X1_size, X2_size), seed, n_pc, split=split, qIT_shots=n_shots, \
                                        qVS_subsamples=qVS_subsamples, qVS_maxsize=qVS_maxsize)
                end_t = time.time()
                save_interim_kernel_calculation_time(end_t-start_t, False, kmethod, (X1_size, X2_size), split, seed, n_pc, qIT_shots=n_shots, \
                                                    qVS_subsamples=qVS_subsamples, qVS_maxsize=qVS_maxsize)
        else:
            continue
    if split=='train':
        gram_matrix = gram_matrix + gram_matrix.T - np.diag(np.diag(gram_matrix))
    save_interim_kernel_copy(gram_matrix, kmethod, (X1_size, X2_size), seed, n_pc, split=split, qIT_shots=n_shots, \
                            qVS_subsamples=qVS_subsamples, qVS_maxsize=qVS_maxsize)
    save_interim_kernel_calculation_time(0, True, kmethod, (X1_size, X2_size), split, seed, n_pc, qIT_shots=n_shots, \
                                        qVS_subsamples=qVS_subsamples, qVS_maxsize=qVS_maxsize)
    return gram_matrix

def get_qRM_settings_list(seed, n_pc, n_settings):
    '''
    Returns a list of length 'n_settings' containing different measurement settings
    Each measurement setting is a list of local random Haar unitaries corresponding to each qubit
    '''
    qRM_settings_list = get_saved_qRM_settings(seed, n_pc, n_settings)
    if qRM_settings_list is None:  
        qRM_settings_list = []
        for _ in range(n_settings):
            measurement_setting = []
            for _ in range(n_pc):
                measurement_setting.append(random_unitary(2, seed=seed))
            qRM_settings_list.append(measurement_setting)
        save_qRM_settings(qRM_settings_list, seed, n_pc, n_settings)
    return qRM_settings_list

def single_random_measurement_circuit(x, n_reuploads, measurement_setting):
    '''
    Returns the circuit of the quantum feature followed by the randomized measurement
    '''
    n_pc = len(x)
    U_x = quantum_feature(x, n_reuploads)
    kernel_c = U_x.copy()
    # Rotate each Qubit using a random Haar unitary
    for i in range(n_pc):
        kernel_c.append(measurement_setting[i], [i])
    # Measure in computational basis
    kernel_c.measure(range(n_pc), range(n_pc))
    return kernel_c

def get_single_random_measurement_results(x, measurement_setting, shots):
    '''
    Runs a single randomized measurement circuit for a single point
    Returns a list of (bitstring, probability) tuples
    '''
    # Run the job on the QasmSimulator
    kernel_c = single_random_measurement_circuit(x, 3, measurement_setting)
    simulator = QasmSimulator()
    t_circuit = transpile(kernel_c, simulator)
    job = simulator.run(t_circuit, shots=shots)
    result = job.result()
    # Format and return the result
    formatted_result = []
    for key, value in result.get_counts().items():
        formatted_result.append((list(key), value/shots))
    return formatted_result

def get_random_measurements_results(x, qRM_settings_list, n_shots=8000):
    '''
    Runs randomized measurements for a single point for each measurement_setting and returns a list of qiskit.result objects
    '''
    result_lst = []
    for measurement_setting in qRM_settings_list:
        result = get_single_random_measurement_results(x, measurement_setting, n_shots)
        result_lst.append(result)
    return result_lst

def get_dataset_randomized_measurements(X, dataset_index, seed, qRM_settings, qRM_shots, split):
    # Check whether there are any measurements saved
    X_measurements = retrieve_interim_qRM_measurements(dataset_index, len(X), seed, len(X[0]), qRM_shots, len(qRM_settings), split)
    X_measurements = list(X_measurements) if X_measurements is not None else None
    # If no measurements are saved create a new measurement list for X
    if X_measurements is None:
        X_measurements = []
        rml_progress_bar = trange(0, len(X), position=0, leave=True)
    elif len(X_measurements) == len(X): # If an incomplete measurement list exists, complete the measurements list
        return X_measurements
    elif len(X_measurements) != len(X):
        rml_progress_bar = trange(len(X_measurements), len(X), position=0, leave=True)
    for i in rml_progress_bar:
        start_t = time.time()
        rml_progress_bar.set_description("Retrieving measurements of %d th datapoint" % i)
        x_results = get_random_measurements_results(X[i], qRM_settings, qRM_shots)
        X_measurements.append(x_results)
        save_interim_qRM_measurements(dataset_index, X_measurements, len(X), seed, len(X[0]), qRM_shots, len(qRM_settings), split)
        end_t = time.time()
        save_interim_kernel_calculation_time(end_t-start_t, False, 'qRM', len(X), split, seed, len(X[0]), qRM_shots=qRM_shots, qRM_settings=len(qRM_settings))
    return X_measurements

def get_kernel_matrix_qRM(X1, X2, seed=None, n_settings=8, n_shots=8000):
    '''
    Takes as input two datasets X1 and X2 of type np.ndarray
    Calculates the similarity between each point pair using randomized measurements
    Returns the kernel matrix
    '''
    X1_size, X2_size, n_pc = len(X1), len(X2), len(X1[0])
    split = 'train' if np.array_equal(X1,X2) else 'test'
    # Retrieve old kernel copy and return it if it is complete
    qRM_settings_list = get_qRM_settings_list(seed, n_pc, n_settings)
    gram_matrix = retrieve_interim_kernel_copy('qRM', (X1_size, X2_size), seed, n_pc, split=split, qRM_shots=n_shots, qRM_settings=n_settings)
    if gram_matrix is not None and is_kernel_complete('qRM', (X1_size, X2_size), split, seed, n_pc, n_shots, n_settings):
        return gram_matrix
    elif gram_matrix is not None and (gram_matrix[-1, -1] != 0 and (gram_matrix[-1, 1] != 1)): # if gram_matrix exists and it is complete without mitigation
        start_t = time.time()
        X1_measurements = get_dataset_randomized_measurements(X1, 1, seed, qRM_settings_list, n_shots, split)
        X2_measurements = get_dataset_randomized_measurements(X2, 2, seed, qRM_settings_list, n_shots, split) if split == 'test' else X1_measurements
        X1_purities = [combine_randomized_measurements(x,x) for x in X1_measurements] if split=='test' else None
        X2_purities = [combine_randomized_measurements(x,x) for x in X2_measurements] if split=='test' else None
        gram_matrix = apply_mitigation(gram_matrix, split, X1_purities=X1_purities, X2_purities=X2_purities)
        if split=='train':
            gram_matrix = gram_matrix + gram_matrix.T - np.diag(np.diag(gram_matrix))
        save_interim_kernel_copy(gram_matrix, 'qRM', (X1_size, X2_size), seed, n_pc, split=split, qRM_shots=n_shots, qRM_settings=n_settings)
        end_t = time.time()
        save_interim_kernel_calculation_time(end_t-start_t, True, 'qRM', (X1_size, X2_size), split, seed, n_pc, qRM_shots=n_shots, qRM_settings=n_settings)
        return gram_matrix
    start_t = time.time()
    qRM_settings_list = get_qRM_settings_list(seed, n_pc, n_settings)
    end_t = time.time()
    save_interim_kernel_calculation_time(end_t-start_t, False, 'qRM', (X1_size, X2_size), split, seed, n_pc, qRM_shots=n_shots, qRM_settings=n_settings)
    # Measure and save the datasets' measurements using randomized measurements, the time for the calculation is saved from within the function
    start_t = time.time()
    X1_measurements = get_dataset_randomized_measurements(X1, 1, seed, qRM_settings_list, n_shots, split)
    X2_measurements = get_dataset_randomized_measurements(X2, 2, seed, qRM_settings_list, n_shots, split) if split == 'test' else X1_measurements
    if gram_matrix is None:
        gram_matrix = np.zeros((X1_size,X2_size))
        num_eval = X1_size*X2_size
        indices = product(range(X1_size), range(X2_size))
    else:
        next_i, next_j = find_kernel_entry_index(gram_matrix)
        num_eval = (X1_size - next_i) * X2_size - next_j
        indices = chain(product([next_i], range(next_j, X2_size)), \
                        product(range(next_i+1, X1_size), range(0, X2_size)))
    end_t = time.time()
    save_interim_kernel_calculation_time(end_t - start_t, False, 'qRM', (X1_size, X2_size), split, seed, n_pc, qRM_shots=n_shots, qRM_settings=n_settings)
    km_progress_bar = tqdm(indices, total=num_eval, position=0, leave=True)
    # Calculate the missing kernel elements and save each row and the time it took to calculate it
    for i, j in km_progress_bar:
        km_progress_bar.set_description("Processing gram_matrix RM [%d][%d]" %(i, j))
        if (split == 'test') or (split == 'train' and j >= i):
            start_t = time.time()
            gram_matrix[i][j] = combine_randomized_measurements(X1_measurements[i], X2_measurements[j])
            if j == i or n_pc > 8:
                save_interim_kernel_copy(gram_matrix, 'qRM', (X1_size, X2_size), seed, n_pc, split=split, qRM_shots=n_shots, qRM_settings=n_settings)
                end_t = time.time()
                save_interim_kernel_calculation_time(end_t - start_t, False, 'qRM', (X1_size, X2_size), split, seed, n_pc, qRM_shots=n_shots, qRM_settings=n_settings)
        else:
            continue
    # Apply mitigation and save the final kernel copy as well as its calculation time
    start_t = time.time()
    X1_purities = [combine_randomized_measurements(x,x) for x in X1_measurements] if split=='test' else None
    X2_purities = [combine_randomized_measurements(x,x) for x in X2_measurements] if split=='test' else None
    gram_matrix = apply_mitigation(gram_matrix, split, X1_purities=X1_purities, X2_purities=X2_purities)
    if split=='train':
        gram_matrix = gram_matrix + gram_matrix.T - np.diag(np.diag(gram_matrix))
    save_interim_kernel_copy(gram_matrix, 'qRM', (X1_size, X2_size), seed, n_pc, split=split, qRM_shots=n_shots, qRM_settings=n_settings)
    end_t = time.time()
    save_interim_kernel_calculation_time(end_t - start_t, True, 'qRM', (X1_size, X2_size), split, seed, n_pc, qRM_shots=n_shots, qRM_settings=n_settings)
    return gram_matrix


def get_exponential_hamming_matrix(A,B):
    '''
    Returns the matrix containing the exponential hamming factors
    '''
    base = np.array(-2)
    powers = -np.count_nonzero(A[:, None, :] != B[None, :, :], axis=-1)
    return np.float_power(base, powers)

def get_single_unitary_trace(probA, probB, strA, strB):
    '''
    Retrace the trace of two quantum states (two datapoints) according to a single measurement setting
    '''
    prob_product = np.outer(probA, probB)
    exponential_hamming_factors = get_exponential_hamming_matrix(strA, strB)
    single_unitary_trace = np.sum(np.multiply(exponential_hamming_factors, prob_product))
    return single_unitary_trace

def combine_randomized_measurements(x1_measurements, x2_measurements):
    '''
    Combine the measurement results of x1 and x2 across different measurement settings 
    following Eq. 7, arXiv:2108.01039
    Returns the kernel element for two data points x1 and x2
    '''
    # Calculate the inner loop
    n_pc = len(x1_measurements[0][0][0])
    # We combine the measurements acc.t. Eq (7)
    traces_by_setting = np.array([])
    # X1_measurements is a list of settings results, which are lists containing lists of shot results
    for x1_result, x2_result in zip(x1_measurements, x2_measurements):
        strA, probA = np.array(np.vstack(np.array(x1_result, dtype=object)[:, 0])), np.array(x1_result, dtype=object)[:, 1]
        strB, probB =  np.array(np.vstack(np.array(x2_result, dtype=object)[:, 0])), np.array(x2_result, dtype=object)[:, 1]
        new_trace = get_single_unitary_trace(probA, probB, strA, strB)
        traces_by_setting = np.append(traces_by_setting, new_trace)
    traces_by_setting = (2**n_pc) * traces_by_setting.mean()
    return traces_by_setting

def apply_mitigation(gram_matrix, split, X1_purities=None, X2_purities=None):
    num_eval = gram_matrix.shape[0]*gram_matrix.shape[1]
    indices = product(range(gram_matrix.shape[0]), range(gram_matrix.shape[1]))
    em_progress_bar = tqdm(indices, total=num_eval, position=0, leave=True)
    if split == 'train':
        for i, j in em_progress_bar:
            em_progress_bar.set_description("Applying error mitigation [%d][%d]" %(i, j))
            if i != j:
                gram_matrix[i][j] = gram_matrix[i][j] / np.sqrt(gram_matrix[i][i] * gram_matrix[j][j])
            else:
                continue
        for i in range(gram_matrix.shape[0]):
            gram_matrix[i][i] = 1
    else:
        for i, j in em_progress_bar:
            em_progress_bar.set_description("Applying error mitigation [%d][%d]" %(i, j))
            gram_matrix[i][j] = gram_matrix[i][j] / np.sqrt(X1_purities[i] * X2_purities[j])
    return gram_matrix

def save_interim_kernel_copy(interimKernelCopy, kmethod, size, seed, n_pc, split=None, qIT_shots=None,\
                            qRM_shots=None, qRM_settings=None, qVS_subsamples=None, qVS_maxsize=None):
    '''
    Saves the interim copy of the quantum kernel in a file in the InterimResults folder
    Can be called from all quantum kernel
    The kernel copy can then be loaded using numpy.load()
    '''
    interimResultsFolder = f'{ROOT_DIR}/InterimResults/{kmethod}/{split}/'
    if not os.path.exists(interimResultsFolder):
        os.makedirs(interimResultsFolder)
    interimKernelCopyPath = interimResultsFolder + f'dsize_{size}_seed_{seed}_n_pc_{n_pc}'
    if kmethod=='qIT':
        interimKernelCopyPath += f'_n_shots_{qIT_shots}'
    elif kmethod=='qRM':
        interimKernelCopyPath += f'_n_shots_{qRM_shots}_n_settings_{qRM_settings}'
    elif kmethod=='qVS':
        interimKernelCopyPath += f'_n_subsamples_{qVS_subsamples}_n_maxsize_{qVS_maxsize}'
    interimKernelCopyPath += '.npy'
    # TODO: Add handling for qVS and qBBF
    np.save(interimKernelCopyPath, interimKernelCopy)

def find_kernel_entry_index(interimKernelCopy):
    '''
    Returns the index of the kernel element to be calculated next
    '''
    shape = interimKernelCopy.shape
    nonzero_entries = np.argwhere(interimKernelCopy != 0)
    if nonzero_entries.size == 0:
        return 0, 0
    # Get the index of the last calculated element
    last_index = nonzero_entries[-1]
    # Check if the last column was fully calculated
    if last_index[1] == (shape[1]-1):
        # Check if the last row was fully calculated
        if last_index[0] == (shape[0]-1):
            return -1, -1
        else:
            new_index = (last_index[0]+1, 0)
    else:
        new_index = (last_index[0], last_index[1]+1)
    return new_index

def retrieve_interim_kernel_copy(kmethod, size, seed, n_pc, split=None, qIT_shots=None, \
                                qRM_shots=None, qRM_settings=None, qVS_subsamples=None, qVS_maxsize=None):
    '''
    Returns the last saved kernel copy and the last_coordinates computed
    The argument last_coordinates should help restarting kernel calculation
    '''
    interimResultsFolder = f'{ROOT_DIR}/InterimResults/{kmethod}/{split}/'
    interimKernelCopyPath = interimResultsFolder + f'dsize_{size}_seed_{seed}_n_pc_{n_pc}'
    if kmethod=='qIT':
        interimKernelCopyPath += f'_n_shots_{qIT_shots}'
    elif kmethod=='qRM':
        interimKernelCopyPath += f'_n_shots_{qRM_shots}_n_settings_{qRM_settings}'
    elif kmethod=='qVS':
        interimKernelCopyPath += f'_n_subsamples_{qVS_subsamples}_n_maxsize_{qVS_maxsize}'
    interimKernelCopyPath += '.npy'
    # First we check if the path exists. If the path exists, then we load and return interim KernelCopy
    if os.path.exists(interimKernelCopyPath):
        interimKernelCopy = np.load(interimKernelCopyPath)
        return interimKernelCopy
    # If the path does not exist, we return None
    return None
    
def save_interim_qRM_measurements(dataset_index, qRMmeasurements, size, seed, n_pc, qRM_shots, qRM_settings, split):
    '''
    Saves the quantum measurements
    '''
    qRMMeasurementsFolderName = f'{ROOT_DIR}/InterimResults/qRM/Measurements/'
    if not os.path.exists(qRMMeasurementsFolderName):
        os.mkdir(qRMMeasurementsFolderName)
    qRMmeasurementsFileName = qRMMeasurementsFolderName + f'{split}_X{dataset_index}'
    qRMmeasurementsFileName += f'_dsize_{size}_seed_{seed}_n_pc_{n_pc}'
    qRMmeasurementsFileName += f'_n_shots_{qRM_shots}_n_settings_{qRM_settings}'
    qRMmeasurementsFileName += '.npy'
    np.save(qRMmeasurementsFileName, np.array(qRMmeasurements, dtype=object), allow_pickle=True)

def retrieve_interim_qRM_measurements(dataset_index, size, seed, n_pc, qRM_shots, qRM_settings, split):
    '''
    Returns formatted list of all measurements needed for kernel matrix calculation
    '''
    qRMMeasurementsFolderName = f'{ROOT_DIR}/InterimResults/qRM/Measurements/'
    qRMmeasurementsFileName = qRMMeasurementsFolderName + f'{split}_X{dataset_index}'
    qRMmeasurementsFileName += f'_dsize_{size}_seed_{seed}_n_pc_{n_pc}'
    qRMmeasurementsFileName += f'_n_shots_{qRM_shots}_n_settings_{qRM_settings}'
    qRMmeasurementsFileName += '.npy'
    # First we check if the path exists. If the path exists, then we load and return measurement
    if os.path.exists(qRMmeasurementsFileName):
        interimMeasurementsResults = np.load(qRMmeasurementsFileName, allow_pickle=True)
        return interimMeasurementsResults
    # If the path does not exist, we return None
    return None

def save_qRM_settings(qRM_settings_list, seed, n_pc, qRM_settings):
    qRMMeasurementsFolderName = f'{ROOT_DIR}/InterimResults/qRM/Measurements/'
    if not os.path.exists(qRMMeasurementsFolderName):
        os.makedirs(qRMMeasurementsFolderName)
    qRMsettingsFileName = qRMMeasurementsFolderName + f'settings'
    qRMsettingsFileName += f'_seed_{seed}_n_pc_{n_pc}'
    qRMsettingsFileName += f'_n_settings_{qRM_settings}'
    qRMsettingsFileName += '.npy'
    np.save(qRMsettingsFileName, np.array(qRM_settings_list, dtype=object), allow_pickle=True)

def get_saved_qRM_settings(seed, n_pc, qRM_settings):
    '''
    Gets the measurements settings from the last run for 
    the specified seed, n_pc, n_shots and n_settings if they exist
    '''
    # First we check if the path exists. If the path exists, then we load and return measurement
    qRMMeasurementsFolderName = f'{ROOT_DIR}/InterimResults/qRM/Measurements/'
    qRMsettingsFileName = qRMMeasurementsFolderName + f'settings'
    qRMsettingsFileName += f'_seed_{seed}_n_pc_{n_pc}'
    qRMsettingsFileName += f'_n_settings_{qRM_settings}'
    qRMsettingsFileName += '.npy'
    if os.path.exists(qRMsettingsFileName):
        qRM_settings_list = np.load(qRMsettingsFileName, allow_pickle=True)
        return qRM_settings_list
    # If the path does not exist, we return None
    return None

def save_interim_kernel_calculation_time(calc_time, complete, kmethod, size, split, seed,\
                n_pc, qIT_shots=None, qRM_shots=None, qRM_settings=None, qVS_subsamples=None, qVS_maxsize=None):
    '''
    Saves the time it took to calculate the interim kernel copy
    '''
    interimResultsFolder = f'./InterimResults/{kmethod}/{split}/'
    if not os.path.exists(interimResultsFolder):
        os.makedirs(interimResultsFolder)
    KernelCalcTimeFile = interimResultsFolder
    KernelCalcTimeFile += f'kernel_calculation_times_dsize_{size}'
    KernelCalcTimeFile += f'_seed_{seed}_n_pc_{n_pc}'
    if kmethod=='qIT':
        KernelCalcTimeFile += f'_n_shots_{qIT_shots}'
    elif kmethod=='qRM':
        KernelCalcTimeFile += f'_n_shots_{qRM_shots}_n_settings_{qRM_settings}'
    elif kmethod=='qVS':
        KernelCalcTimeFile += f'_n_subsamples_{qVS_subsamples}_maxsize_{qVS_maxsize}'
    KernelCalcTimeFile += '.npy'
    if not os.path.exists(KernelCalcTimeFile):
        time_array = np.array([calc_time, complete])
        np.save(KernelCalcTimeFile, time_array)
    else:
        time_array = np.load(KernelCalcTimeFile)
        if time_array.size == 0:
            time_array = np.array([calc_time, complete])
        elif time_array[1] == False:
            time_array[0] += calc_time
            time_array[1] = complete
        np.save(KernelCalcTimeFile, time_array)

def retrieve_interim_kernel_calculation_time(kmethod, size, split, seed, n_pc, qIT_shots=None, \
                                        qRM_shots=None, qRM_settings=None, qVS_subsamples=None, qVS_maxsize=None):
    '''
    Returns the time it took previously to calculate the interim kernel copy if there was a previous calculation, otherwise it returns 0
    '''
    interimResultsFolder = f'./InterimResults/{kmethod}/{split}/'
    if not os.path.exists(interimResultsFolder):
        return 0
    KernelCalcTimeFile = interimResultsFolder
    KernelCalcTimeFile += f'kernel_calculation_times_dsize_{size}'
    KernelCalcTimeFile += f'_seed_{seed}_n_pc_{n_pc}'
    if kmethod=='qIT':
        KernelCalcTimeFile += f'_n_shots_{qIT_shots}'
    elif kmethod=='qRM':
        KernelCalcTimeFile += f'_n_shots_{qRM_shots}_n_settings_{qRM_settings}'
    elif kmethod=='qVS':
        # Find all kernel calculation time files that correspond to the regex
        regex = re.compile(f'^kernel_calculation_times_dsize_.*_seed_{seed}_n_pc_\
                        {n_pc}_n_subsamples_{qVS_subsamples}_maxsize_{qVS_maxsize}.npy$')
        calculation_time_files_list = [file for file in tuple(os.walk(interimResultsFolder))[0][2] if regex.match(file)]
        # Gather the time from the different files
        previous_time = 0
        for filename in calculation_time_files_list:
            KernelCalcTimeFile = interimResultsFolder + f'{filename}'
            if os.path.exists(KernelCalcTimeFile):
                time_array = np.load(KernelCalcTimeFile)
                if time_array.size != 0:
                    previous_time += time_array[0]
        return previous_time
    KernelCalcTimeFile += '.npy'
    if not os.path.exists(KernelCalcTimeFile):
        return 0
    else:
        time_array = np.load(KernelCalcTimeFile)
        if time_array.size == 0:
            return 0
        return time_array[0]

def is_kernel_complete(kmethod, size, split, seed, n_pc, qIT_shots=None, qRM_shots=None, qRM_settings=None, qVS_subsamples=None, qVS_maxsize=None):
    '''
    Checks whether the kernel was completed (including mitigation for qRM)
    '''
    interimResultsFolder = f'./InterimResults/{kmethod}/{split}/'
    if not os.path.exists(interimResultsFolder):
        return 0
    KernelCalcTimeFile = interimResultsFolder
    KernelCalcTimeFile += f'kernel_calculation_times_dsize_{size}'
    KernelCalcTimeFile += f'_seed_{seed}_n_pc_{n_pc}'
    if kmethod=='qIT':
        KernelCalcTimeFile += f'_n_shots_{qIT_shots}'
    elif kmethod=='qRM':
        KernelCalcTimeFile += f'_n_shots_{qRM_shots}_n_settings_{qRM_settings}'
    KernelCalcTimeFile += '.npy'
    if not os.path.exists(KernelCalcTimeFile):
        return False
    else:
        time_array = np.load(KernelCalcTimeFile)
        if time_array.size == 0:
            return False
        return time_array[1]
    # TODO: refactor all checks here