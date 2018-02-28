import numpy as np
import time
from sklearn.svm import SVC
from dataset import Dataset
from kernels import compute_kernel_matrix
from svm import MySVM
import os

# dataset 0, spectrum_size = 5
# dataset 1, spectrum_size = 4
# dataset 2, spectrum_size = 3
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

print "Computing Kernels..."
K_train_path = './K_train_dataset_%s.npy' % dataset_id
K_val_path = './K_val_dataset_%s.npy' % dataset_id

if os.path.exists(K_val_path) and os.path.exists(K_train_path):
    K_val = np.load(K_val_path)
    K_train = np.load(K_train_path)
else:
    K_train = compute_kernel_matrix(dataset.dataset['train']['sequences'],
                                    spectrum_size=spectrum_size,
                                    feature_extractor=feature_extractor, normalization=normalization)
    K_val = compute_kernel_matrix(dataset.dataset['val']['sequences'], dataset.dataset['train']['sequences'],
                                  spectrum_size=spectrum_size,
                                  feature_extractor=feature_extractor, normalization=normalization)
    np.save(K_train_path, K_train)
    np.save(K_val_path, K_val)

svc = SVC(kernel='precomputed', C=10.)
my_svm = MySVM(C=10., dual=False, verbose=True)

print "Fitting sklearn model..."
start = time.time()
svc.fit(K_train, dataset.dataset['train']['labels'])
print "Fitting took %.3f seconds" % (time.time() - start)

print "Fitting our model..."
start = time.time()
my_svm.fit(K_train, dataset.dataset['train']['labels'])
print "Fitting took %.3f seconds" % (time.time() - start)


print "Scoring with sklearn model:"
print svc.score(K_train, dataset.dataset['train']['labels'])
print svc.score(K_val, dataset.dataset['val']['labels'])

print "Scoring with our model:"
print my_svm.score(K_train, dataset.dataset['train']['labels'])
print my_svm.score(K_val, dataset.dataset['val']['labels'])