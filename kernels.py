import itertools
import numpy as np

alphabet = ['A', 'T', 'C', 'G']


def compute_spectrum(spectrum_size):
    spectrum = []
    for element in itertools.product(alphabet, repeat=spectrum_size):
        spectrum.append(''.join(element))
    return spectrum


def spectrum_feature(X, spectrum):
    spectrum_size = len(spectrum[0])
    nb_samples = len(X)
    features = np.zeros((nb_samples, len(spectrum)))
    for i in range(nb_samples):
        for j in range(len(X[i]) - spectrum_size + 1):
            features[i, spectrum.index(X[i][j:j + spectrum_size])] += 1
    return features


def mismatch_feature(X, spectrum):
    spectrum_size = len(spectrum[0])
    nb_samples = len(X)
    features = np.zeros((nb_samples, len(spectrum)))
    for i in range(nb_samples):
        for j in range(len(X[i]) - spectrum_size + 1):
            substring = X[i][j:j+spectrum_size]

            mismatches = []
            for k in range(spectrum_size):
                mismatches += [substring[:k] + l + substring[k+1:] for l in alphabet]
            mismatches = set(mismatches)

            for mismatch in mismatches:
                features[i, spectrum.index(mismatch)] += 1
    return features


def normalize(K, features1, features2=None):
    if features2 is not None:
        k1 = np.expand_dims(1. / np.sqrt(np.sum(features1 ** 2, axis=1)), axis=1)
        k2 = np.expand_dims(1. / np.sqrt(np.sum(features2 ** 2, axis=1)), axis=1)
        return K * k1.dot(k2.T) + 1.
    else:
        k = np.expand_dims(1. / np.sqrt(np.diagonal(K)), axis=1)
        return K * k.dot(k.T) + 1.


def linear_kernel(features1, features2=None, **kwargs):
    if features2 is not None:
        K = features1.dot(features2.T)
    else:
        K = features1.dot(features1.T)

    if kwargs['normalization']:
        K = normalize(K, features1, features2)

    return K


def rbf_kernel(features1, features2=None, **kwargs):
    if 'gamma' in kwargs:
        if kwargs['gamma'] is None:
            gamma = 1. / features1.shape[1]
        else:
            gamma = kwargs['gamma']
    else:
        gamma = 1. / features1.shape[1]

    XX = np.linalg.norm(features1, axis=1)[:, np.newaxis] ** 2

    if features2 is not None:
        YY = np.linalg.norm(features2, axis=1)[:, np.newaxis] ** 2
        distances = XX - 2 * features1.dot(features2.T) + YY.T
    else:
        distances = XX - 2 * features1.dot(features1.T) + XX.T

    K = np.exp(-gamma * distances)
    return K


def compute_kernel_matrix(X1, X2=None, spectrum_size=3, feature_extractor='spectrum', kernel='linear', **kwargs):
    spectrum = compute_spectrum(spectrum_size)
    feature_fun = get_feature_extractor(feature_extractor)
    kernel_fun = get_kernel(kernel)
    features1 = feature_fun(X1, spectrum)
    features2 = None

    if X2 is not None:
        features2 = feature_fun(X2, spectrum)

    K = kernel_fun(features1, features2, **kwargs)

    return K


def get_feature_extractor(name):
    if name == 'spectrum':
        return spectrum_feature
    elif name == 'mismatch':
        return mismatch_feature


def get_kernel(name):
    if name == 'linear':
        return linear_kernel
    elif name == 'rbf':
        return rbf_kernel


if __name__ == "__main__":
    seq1 = 'CGGGCCTCCTCCAGGCTCAGAATCGACCCCCCCCCATCCTGATAGACCACAACGAAAGCCGTAGCGACGGCCGCAGGAGCTGGCGCGACAGCCCAGAGCTC'
    seq2 = 'GGTGTCTAGTCACATTACCTGAGATCTGACATTTTTATCCGTCTGAGCACTTGATGGTGATTTCTTTCACTGTAGTCAAGTTTCTGCCTGTATAGCAGTGG'
    spec = compute_spectrum(3)
    print mismatch_feature([seq1], spec)
