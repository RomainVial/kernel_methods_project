import csv
import numpy as np


def to_csv(preds, csv_path):
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Id', 'Bound'])
        count = 0
        for dataset_id in [0, 1, 2]:
            for y in preds[dataset_id]:
                writer.writerow([count, int(y)])
                count += 1


def cross_val(clf, K, Y, k=5):
    nb_samples = len(Y)
    fold_size = int(float(len(Y)) / k)
    nb_train = nb_samples - fold_size
    nb_val = fold_size
    scores = np.zeros((k,))
    for i in range(k):
        idx_val = [j for j in range(i * fold_size, (i+1) * fold_size)]
        idx_train = [j for j in (range(i * fold_size) + range((i+1) * fold_size, nb_samples))]

        Y_train = [Y[j] for j in idx_train]
        Y_val = [Y[j] for j in idx_val]

        K_train = np.reshape([K[j, l] for j in idx_train for l in idx_train], (nb_train, nb_train))
        K_val = np.reshape([K[j, l] for j in idx_val for l in idx_train], (nb_val, nb_train))

        clf.fit(K_train, Y_train)
        scores[i] = clf.score(K_val, Y_val)
        print 'Fold {}: {}'.format(i, scores[i])
    return scores
