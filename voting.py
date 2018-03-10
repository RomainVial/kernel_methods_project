import numpy as np

import utils
from svm import MySVM


class Voting:
    def __init__(self, models):
        """

        :param models: List of MySVM models
        """
        self.models = models  # type: list[MySVM]
        self.n = len(models)

    def predict(self, K, threshold):
        """

        :param K: Kernel Matrix between training points and validation points
        :param threshold: the portion of positive votes needed to decide that the answer is positive
        :return:
        """
        preds = np.zeros((K.shape[0], self.n))
        for i, model in enumerate(self.models):
            pred = model.predict(K)
            preds[:, i] = pred

        sum_preds = np.sum(preds, axis=1)
        voted_pred = (sum_preds > threshold * self.n).astype(int)
        return voted_pred

    def score(self, K_val, y_true, threshold):
        """

        :param K_val: This is the kernel matrix between Validation features and Training features
        :param y_true: true labels
        :param threshold: the portion of positive votes needed to decide that the answer is positive
        :return: Accuracy score
        """
        y_pred = self.predict(K_val, threshold)
        return utils.accuracy(y_pred, y_true)
