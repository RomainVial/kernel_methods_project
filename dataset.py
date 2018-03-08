import csv
import numpy as np


class Dataset:
    def __init__(self, train_sequence_files, train_feature_files, train_label_files,
                 test_sequence_files, test_feature_file):
        self.dataset = {
            'train': {'sequences': [], 'labels': [], 'basic_features': []},
            'val': {'sequences': [], 'labels': [], 'basic_features': []},
            'test': {'sequences': [], 'labels': [], 'basic_features': []}
        }

        for file_path in train_sequence_files:
            self._read_X_csv(file_path, 'train')

        for file_path in train_feature_files:
            self._read_X_mat_csv(file_path, 'train')

        for file_path in train_label_files:
            self._read_Y_csv(file_path, 'train')

        for file_path in test_sequence_files:
            self._read_X_csv(file_path, 'test')

        for file_path in test_feature_file:
            self._read_X_mat_csv(file_path, 'test')

    def _read_X_csv(self, csv_path, split):
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                self.dataset[split]['sequences'].append(row[0])

    def _read_X_mat_csv(self, csv_path, split):
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                feature = np.asarray(map(lambda s: float(s), row[0].split(' ')))
                self.dataset[split]['basic_features'].append(feature)

    def _read_Y_csv(self, csv_path, split):
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            reader.next()
            for row in reader:
                self.dataset[split]['labels'].append(int(row[1]))


if __name__ == "__main__":
    dataset = Dataset(['data/Xtr0.csv'],
                      ['data/Xtr0_mat50.csv'],
                      ['data/Ytr0.csv'],
                      ['data/Xte0.csv'],
                      ['data/Xte0_mat50.csv'])

    print np.bincount(dataset.dataset['train']['labels']).astype(float) / len(dataset.dataset['train']['labels'])
