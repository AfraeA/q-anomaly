from os import path
from urllib import request
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def import_data(data_url, data_file):
    '''
    Downloads data and returns a pandas dataframe
    '''
    if not path.exists(data_file):
        request.urlretrieve(data_url, data_file)
        data, meta = arff.loadarff(data_file)
        data = pd.DataFrame(data)
        data.to_csv(data_file, index=False)
    data = pd.read_csv(data_file)
    data.drop(columns=["Time", "Amount"], inplace=True)
    return data

def get_split_indices(data, size, train_split=False):
    
    anomalous_indices = data[data['Class'] == 1].index
    nominal_indices = data[data['Class'] == 0].index
    
    if train_split:
        sample_indices = np.random.choice(nominal_indices, replace=False, size=size)
    else:
        anomalous_size = int(0.05*size)
        nominal_size = size - anomalous_size
        
        sample_nominal_indices = np.random.choice(nominal_indices, replace=False, size=nominal_size)
        sample_anomalous_indices = np.random.choice(anomalous_indices, replace=False, size=anomalous_size)

        sample_indices = np.sort(np.append(sample_nominal_indices, sample_anomalous_indices))
    return sample_indices

def split_dataset(data, size):
    
    train_indices = get_split_indices(data, size, train_split=True)
    test_indices = get_split_indices(data.drop(train_indices), size)
    
    assert np.intersect1d(train_indices, test_indices).size == 0, \
                "The train and test set contains common elements"
    train_data = np.take(data, train_indices, axis=0)
    test_data = np.take(data, test_indices, axis=0)
    return train_data, test_data

def preprocess_data(data, seed, n_pc, train_split=False, pca_sc=None, pca=None):#, sc=None):
    
    features = data.drop(columns=["Class"])
    labels = data["Class"]
    
    if train_split:
        # Standard scaling
        pca_sc = StandardScaler()
        features = pca_sc.fit_transform(features)

        # PCA
        pca = PCA(n_components=n_pc, random_state=seed)
        features = pca.fit_transform(features)
        
        # Rescaling to zero mean and 1/sqrt(M) variance 
        #sc = StandardScaler()
        #features = sc.fit_transform(features) / np.sqrt(n_pc)
    else:
        try:
            features = pca_sc.transform(features)
            features = pca.transform(features)
            #features = sc.transform(features) / np.sqrt(n_pc)
        except:
            print('PCA or Scalers not available')

    return features, labels, pca_sc, pca#, sc