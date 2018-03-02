import os
import time
import utils
import numpy as np
from svm import MySVM
from dataset import Dataset
from kernels import compute_kernel_matrix


params = {0: {'spectrum_size': 5, 'feature_extractor': 'spectrum', 'kernel': 'rbf', 'C': 1.},
          1: {'spectrum_size': 6, 'feature_extractor': 'spectrum', 'kernel': 'rbf', 'C': 10.},
          2: {'spectrum_size': 5, 'feature_extractor': 'spectrum', 'kernel': 'rbf', 'C': 10.}}
preds = {0: [], 1: [], 2: []}

for dataset_id in [0, 1, 2]:
    print 'Dataset {}'.format(dataset_id)
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

    spectrum_size = params[dataset_id]['spectrum_size']
    feature_extractor = params[dataset_id]['feature_extractor']
    kernel = params[dataset_id]['kernel']
    normalization = True

    print "Computing Kernels..."
    if not os.path.exists('./cache/'):
        os.makedirs('./cache/')
    K_train_path = './cache/K_train_dataset_%s.npy' % dataset_id
    K_val_path = './cache/K_val_dataset_%s.npy' % dataset_id
    K_test_path = './cache/K_test_dataset_%s.npy' % dataset_id

    if os.path.exists(K_val_path) and os.path.exists(K_train_path) and os.path.exists(K_test_path):
        K_val = np.load(K_val_path)
        K_train = np.load(K_train_path)
        K_test = np.load(K_test_path)
    else:
        K_train = compute_kernel_matrix(dataset.dataset['train']['sequences'],
                                        spectrum_size=spectrum_size,
                                        feature_extractor=feature_extractor, kernel=kernel,
                                        normalization=normalization)
        K_val = compute_kernel_matrix(dataset.dataset['val']['sequences'], dataset.dataset['train']['sequences'],
                                      spectrum_size=spectrum_size,
                                      feature_extractor=feature_extractor, kernel=kernel,
                                      normalization=normalization)
        K_test = compute_kernel_matrix(dataset.dataset['test']['sequences'], dataset.dataset['train']['sequences'],
                                       spectrum_size=spectrum_size,
                                       feature_extractor=feature_extractor, kernel=kernel,
                                       normalization=normalization)
        np.save(K_train_path, K_train)
        np.save(K_val_path, K_val)
        np.save(K_test_path, K_test)

    C = params[dataset_id]['C']
    my_svm = MySVM(C=C, dual=True, verbose=True)

    print "Fitting svm model..."
    start = time.time()
    my_svm.fit(K_train, dataset.dataset['train']['labels'])
    print "Fitting took %.3f seconds" % (time.time() - start)

    print "Scoring:"
    print my_svm.score(K_train, dataset.dataset['train']['labels'])
    print my_svm.score(K_val, dataset.dataset['val']['labels'])

    preds[dataset_id] = my_svm.predict(K_test)

utils.to_csv(preds, 'cache/predictions.csv')
