import numpy as np

alphabet = ['A', 'T', 'C', 'G']
spectrum = [l1 + l2 + l3 for l1 in alphabet for l2 in alphabet for l3 in alphabet]


def spectrum_feature(X):
    nb_samples = len(X)
    features = np.zeros((nb_samples, len(spectrum)))
    for i in range(nb_samples):
        for j in range(len(X[i])-2):
            features[i, spectrum.index(X[i][j:j + 3])] += 1
    return features


def mismatch_feature(X):
    nb_samples = len(X)
    features = np.zeros((nb_samples, len(spectrum)))
    for i in range(nb_samples):
        for j in range(len(X[i]) - 2):
            mismatches = set([l + X[i][j+1:j+3] for l in alphabet] +
                             [X[i][j] + l + X[i][j+2] for l in alphabet] +
                             [X[i][j:j+2] + l for l in alphabet])
            for mismatch in mismatches:
                features[i, spectrum.index(mismatch)] += 1
    return features


def compute_kernel_matrix(X1, X2=None, feature_extractor='spectrum', normalize=True):
    feature_fun = get_feature_extractor(feature_extractor)
    features1 = feature_fun(X1)

    if X2 is not None:
        features2 = feature_fun(X2)
        K = features1.dot(features2.T)

        if normalize:
            k1 = np.expand_dims(1. / np.sqrt(np.sum(features1 ** 2, axis=1)), axis=1)
            k2 = np.expand_dims(1. / np.sqrt(np.sum(features2 ** 2, axis=1)), axis=1)
            K = K * k1.dot(k2.T) + 1.
    else:
        K = features1.dot(features1.T)

        if normalize:
            k = np.expand_dims(1. / np.sqrt(np.diagonal(K)), axis=1)
            K = K * k.dot(k.T) + 1.

    return K


def get_feature_extractor(name):
    if name == 'spectrum':
        return spectrum_feature
    elif name == 'mismatch':
        return mismatch_feature
