import os
from urllib import request
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT_DIR = os.path.dirname(os.path.abspath('./__file__'))

def import_data(data_url):
    '''
    Downloads data and returns a pandas dataframe
    '''
    if not os.path.exists(f'{ROOT_DIR}/Data/'):
        os.mkdir(f'{ROOT_DIR}/Data/')
    data_file = f'{ROOT_DIR}/Data/dataset.csv' 
    if not os.path.exists(data_file):
        request.urlretrieve(data_url, data_file)
        data, meta = arff.loadarff(data_file)
        data = pd.DataFrame(data)
        data.to_csv(data_file, index=False)
    data = pd.read_csv(data_file)
    data.drop(columns=["Time", "Amount"], inplace=True)
    return data

def get_split_indices(data, size, train_split=False, anomaly_ratio=0.05):
    
    anomalous_indices = data[data['Class'] == 1].index
    nominal_indices = data[data['Class'] == 0].index
    
    if train_split:
        sample_indices = np.random.choice(nominal_indices, replace=False, size=size)
    else:
        anomalous_size = int(anomaly_ratio*size)
        nominal_size = size - anomalous_size
        
        sample_nominal_indices = np.random.choice(nominal_indices, replace=False, size=nominal_size)
        sample_anomalous_indices = np.random.choice(anomalous_indices, replace=False, size=anomalous_size)

        sample_indices = np.append(sample_nominal_indices, sample_anomalous_indices)
    return sample_indices

def split_dataset(data, train_size, test_size, anomaly_ratio=0.05):

    train_indices = get_split_indices(data, train_size, train_split=True)
    test_indices = get_split_indices(data.drop(train_indices), test_size, False, anomaly_ratio)
    
    assert np.intersect1d(train_indices, test_indices).size == 0, \
                "The train and test set contains common elements"
    train_data = np.take(data, train_indices, axis=0)
    test_data = np.take(data, test_indices, axis=0)
    return train_data, test_data

def preprocess_data(data, kmethod, seed, n_pc, train_split=False, pca_sc=None, pca=None, sc=None):
    
    features = data.drop(columns=["Class"])
    labels = data["Class"]
    
    if train_split:
        # Standard scaling
        pca_sc = StandardScaler()
        features = pca_sc.fit_transform(features)

        # PCA
        pca = PCA(n_components=n_pc, random_state=seed)
        features = pca.fit_transform(features)
        
        if kmethod.startswith('q'):
            features = features * 0.1
        elif kmethod == "qRM":
            sc = StandardScaler()
            features = sc.fit_transform(features) / np.sqrt(n_pc)
            
    else:
        try:
            features = pca_sc.transform(features)
            features = pca.transform(features)
            
            if kmethod.startswith('q'):
                features = features * 0.1
            elif kmethod == "qRM":
                features = sc.fit_transform(features) / np.sqrt(n_pc)
        except:
            print('PCA or Scalers not available')
    return features, labels, pca_sc, pca, sc