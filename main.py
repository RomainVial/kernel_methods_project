import numpy as np
from sklearn.svm import SVC
from dataset import Dataset
from kernels import compute_kernel_matrix

dataset_id = 1
dataset = Dataset(['data/Xtr{}.csv'.format(dataset_id)],
                  ['data/Xtr{}_mat50.csv'.format(dataset_id)],
                  ['data/Ytr{}.csv'.format(dataset_id)],
                  ['data/Xte{}.csv'.format(dataset_id)],
                  ['data/Xte{}_mat50.csv'.format(dataset_id)])

print 'train: {}, val: {}'.format(len(dataset.dataset['train']['labels']), len(dataset.dataset['val']['labels']))
print 'train 0/1 proportion:', (np.bincount(dataset.dataset['train']['labels']).astype(float)
                                / len(dataset.dataset['train']['labels']))
print 'val 0/1 proportion:', (np.bincount(dataset.dataset['val']['labels']).astype(float)
                              / len(dataset.dataset['val']['labels']))

spectrum_size = 4
feature_extractor = 'mismatch'
normalization = True
K_train = compute_kernel_matrix(dataset.dataset['train']['sequences'],
                                spectrum_size=spectrum_size,
                                feature_extractor=feature_extractor, normalization=normalization)
K_val = compute_kernel_matrix(dataset.dataset['val']['sequences'], dataset.dataset['train']['sequences'],
                              spectrum_size=spectrum_size,
                              feature_extractor=feature_extractor, normalization=normalization)

svc = SVC(kernel='precomputed', C=10.)
svc.fit(K_train, dataset.dataset['train']['labels'])
print svc.score(K_train, dataset.dataset['train']['labels'])
print svc.score(K_val, dataset.dataset['val']['labels'])
