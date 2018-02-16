import csv
import copy
import numpy as np


class Dataset:
    def __init__(self, train_sequence_files, train_label_files, test_sequence_files, val_ratio=0.2):
        self.dataset = {
            'train': {'sequences': [], 'labels': [], 'features': []},
            'val': {'sequences': [], 'labels': [], 'features': []},
            'test': {'sequences': [], 'labels': [], 'features': []}
        }

        for file_path in train_sequence_files:
            self.read_X_csv(file_path, 'train')

        for file_path in train_label_files:
            self.read_Y_csv(file_path, 'train')

        for file_path in test_sequence_files:
            self.read_X_csv(file_path, 'test')

        self.compute_val_split(val_ratio)

    def read_X_csv(self, csv_path, split):
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                self.dataset[split]['sequences'].append(row[0])

    def read_Y_csv(self, csv_path, split):
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            reader.next()
            for row in reader:
                self.dataset[split]['labels'].append(int(row[1]))

    def compute_val_split(self, val_ratio):
        np.random.seed(777)
        split_copy = copy.copy(self.dataset['train'])
        nb_sequences = len(split_copy['sequences'])

        val_indexes = np.random.choice(nb_sequences, int(val_ratio * nb_sequences), replace=False)
        self.dataset['val']['sequences'] = [split_copy['sequences'][idx] for idx in val_indexes]
        self.dataset['val']['labels'] = [split_copy['labels'][idx] for idx in val_indexes]

        train_indexes = np.delete(np.arange(nb_sequences), val_indexes)
        self.dataset['train']['sequences'] = [split_copy['sequences'][idx] for idx in train_indexes]
        self.dataset['train']['labels'] = [split_copy['labels'][idx] for idx in train_indexes]

    def compute_features(self, split):
        # The critical part
        features = []
        char_dict = {'A':[1,0,0,0], 'T':[0,1,0,0], 'C':[0,0,1,0], 'G':[0,0,0,1]}
        for sequence in self.dataset[split]['sequences']:
            sequence_feature = []
            for char in sequence:
                sequence_feature += char_dict[char]
            features.append(sequence_feature)

        self.dataset[split]['features'] = features


if __name__ == "__main__":
    dataset = Dataset(['data/Xtr0.csv', 'data/Xtr1.csv', 'data/Xtr2.csv'],
                      ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv'],
                      ['data/Xte0.csv', 'data/Xte1.csv', 'data/Xte2.csv'])

    print np.bincount(dataset.dataset['train']['labels']).astype(float) / len(dataset.dataset['train']['labels'])
    print np.bincount(dataset.dataset['val']['labels']).astype(float) / len(dataset.dataset['val']['labels'])

    dataset.compute_features('train')
    dataset.compute_features('val')