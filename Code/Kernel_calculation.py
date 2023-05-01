import numpy as np
from tqdm import tqdm, trange
from itertools import product
from scipy.spatial.distance import hamming

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import random_unitary

def quantum_feature(x, reuploads):
    '''
    Returns a IQP like circuit with the specified number of reuploads
    '''
    n_pc = len(x)
    for r in range(2*reuploads):
        qr = QuantumRegister(n_pc)
        cr = ClassicalRegister(n_pc)
        qc = QuantumCircuit(qr, cr)
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


def get_kernel_element(x1, x2, n_shots=1000):
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

def get_kernel_matrix_qIT(X1, X2, n_shots=1000):
    '''
    Returns the kernel matrix using inversion tests
    '''
    print('n_shots', n_shots)
    gram_matrix = np.zeros((len(X1),len(X2)))
    num_eval = len(X1)*len(X2)
    indices = product(range(len(X1)), range(len(X2)))
    progression_bar = tqdm(indices, total=num_eval)
    for i, j in progression_bar:
        progression_bar.set_description("Processing gram_matrix [%d][%d]" %(i, j))
        if j > i:
            gram_matrix[i][j] = get_kernel_element(X1[i], X2[j], n_shots)
        elif i==j:
            gram_matrix[i][i] = get_kernel_element(X1[i], X2[i], n_shots)
        else:
            continue
    return gram_matrix

def get_measurement_settings_list(n_pc, n_settings):
    '''
    Returns a list of length 'n_settings' containing different measurement settings
    Each measurement setting is a list of local random Haar unitaries corresponding to each qubit
    '''
    measurement_settings_list = []
    for _ in range(n_settings):
        measurement_setting = []
        for _ in range(n_pc):
            measurement_setting.append(random_unitary(2))
        measurement_settings_list.append(measurement_setting)
    return measurement_settings_list

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
    kernel_c = single_random_measurement_circuit(x, 3, measurement_setting)
    simulator = QasmSimulator()
    t_circuit = transpile(kernel_c, simulator)
    job = simulator.run(t_circuit, shots=shots)
    result = job.result()
    formatted_result = []
    for key, value in result.get_counts().items():
        formatted_result.append((list(key), value/shots))
    return formatted_result

def get_random_measurements_results(x, measurement_settings_list, n_shots=8000):
    '''
    Runs n_repeats randomized measurements for a single point and returns a list of qiskit.result objects
    '''
    result_lst = []
    for measurement_setting in measurement_settings_list:
        result = get_single_random_measurement_results(x, measurement_setting, n_shots)
        result_lst.append(result)
    return result_lst

def retrieve_measurement_lists(X1, X2, measurement_settings_list, n_shots):
    X1_measurements = []
    print('Retrieving Measurements...')
    rml_progress_bar = trange(len(X1), position=0, leave=True)
    for i in rml_progress_bar:
        rml_progress_bar.set_description("Retrieving measurements of %d th datapoint" % i)
        x_results = get_random_measurements_results(X1[i], measurement_settings_list, n_shots)
        X1_measurements.append(x_results) 
    if not np.array_equal(X1, X2):
        X2_measurements = []
        rml2_progress_bar = trange(len(X2), position=0, leave=True)
        for i in rml2_progress_bar:
            rml2_progress_bar.set_description("Retrieving measurements of %d th datapoint in 2nd dataset " % i)
            x_results = get_random_measurements_results(X2[i], measurement_settings_list, n_shots)
            X2_measurements.append(x_results)
    else:
        X2_measurements = X1_measurements
    return X1_measurements, X2_measurements
    
def get_kernel_matrix_qRM(X1, X2, n_settings=8, n_shots=8000):
    '''
    Takes as input two datasets X1 and X2 of type np.ndarray
    Calculates the similarity between each point pair from randomized measurements
    Returns the kernel matrix
    '''
    print(n_settings, n_shots)
    n_pc = len(X1[0])
    measurement_settings_list = get_measurement_settings_list(n_pc, n_settings)
    X1_measurements, X2_measurements = retrieve_measurement_lists(X1, X2, measurement_settings_list, n_shots)
    gram_matrix = np.zeros((len(X1),len(X2)))
    num_eval = len(X1)*len(X2)
    indices = product(range(len(X1)), range(len(X2)))
    km_progress_bar = tqdm(indices, total=num_eval, position=0, leave=True)
    # Calculating the diagonals, which are needed for the noise mitigation
    for i, j in km_progress_bar:
        km_progress_bar.set_description("Processing gram_matrix RM [%d][%d]" %(i, j))
        if j >= i:
            gram_matrix[i][j] = combine_randomized_measurements(X1_measurements[i], X2_measurements[j])
        else:
            continue
    print(gram_matrix)
    gram_matrix = apply_mitigation(gram_matrix)
    return gram_matrix

def combine_randomized_measurements(x1_measurements, x2_measurements):
    '''
    Combine the measurement results of x1 and x2 across different measurement settings 
    following Eq. 7, arXiv:2108.01039
    Returns the kernel element for two data points x1 and x2
    '''
    # Calculate the inner loop
    n_pc = len(x1_measurements[0][0][0])
    n_settings = len(x1_measurements)
    # We combine the measurements acc.t. Eq (7)
    kernel_element_per_setting_list = np.array([])
    for x1_result, x2_result in zip(x1_measurements, x2_measurements):
        kernel_element_per_setting = 0
        for x1_key, x1_probability in x1_result:
            for x2_key, x2_probability in x2_result:
                hamming_distance = int(hamming(x1_key, x2_key)*len(x1_key))
                kernel_element_per_setting += (-2)**(-hamming_distance) * x1_probability * x2_probability
        kernel_element_per_setting_list = np.append(kernel_element_per_setting_list, kernel_element_per_setting)
        kernel_element = 2**n_pc * kernel_element_per_setting_list.mean()
    return kernel_element

def apply_mitigation(gram_matrix):
    num_eval = len(gram_matrix)**2
    indices = product(range(len(gram_matrix)), range(len(gram_matrix)))
    em_progress_bar = tqdm(indices, total=num_eval, position=0, leave=True)
    for i, j in em_progress_bar:
        em_progress_bar.set_description("Applying error mitigation [%d][%d]" %(i, j))
        if j >= i:
            gram_matrix[i][j] = gram_matrix[i][j] / np.sqrt(gram_matrix[i][i] * gram_matrix[j][j])
        else:
            continue
    return gram_matrix