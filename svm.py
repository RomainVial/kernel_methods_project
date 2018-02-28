import cvxopt
from cvxopt.base import matrix
import numpy as np


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    """
    Wrapper for CVXOPT solver:
        min [ 1/2 (x^T P x) + q^T x ]
            s.t G x <= h and A x = b
    :return: ndarray solution
    """
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    try:
        sol = cvxopt.solvers.qp(*args)
    except ValueError as e:
        print "Warning: Some constraints are redundant. %s" % e.message
        sol = cvxopt.solvers.qp(*args, kktsolver='ldl', options={'kktreg': 1e-9})
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


class MySVM:
    def __init__(self, C, dual=True, verbose=False):
        self.C = C
        self.dual = dual
        self.alpha = None
        self.verbose = verbose

    def solve_dual_svm(self, K, y):
        n = K.shape[0]
        P = 2. * K
        q = - 2. * y.reshape((n, 1))

        # computing G
        G = np.vstack([-np.diag(y), np.diag(y)]).astype(float)

        h = np.vstack([np.zeros((n, 1)), self.C * np.ones((n, 1))])
        return cvxopt_solve_qp(P=P, q=q, G=G, h=h)

    def solve_primal_svm(self, K, y):
        """
        The primal problem is:
            min_{alpha, xi} [ 1 / n sum_i {xi_i} + lambda * alpha.T K alpha ]
                s.t y_i [K alpha]_i + xi_i - 1 >= 0 for i = 1..n
                    and x_i >= 0 for i = 1..n
        :param K: np array of shape (n, n) - Kernel matrix
        :param y: 1-d array labels
        :return: 1-d array solution [alpha, xi]
        """
        n = K.shape[0]
        lambda_ = 1. / (2. * n * self.C)

        # computing P
        temp = np.hstack([K, np.zeros((n, n))])
        P = 2. * lambda_ * np.vstack([temp, np.zeros((n, 2 * n))])

        q = 1. / n * np.vstack([np.zeros((n, 1)), np.ones((n, 1))])

        # computing G
        temp1 = np.hstack([- y.reshape((n, 1)) * K, - np.identity(n)])
        temp2 = np.hstack([np.zeros((n, n)), - np.identity(n)])
        G = np.vstack([temp1, temp2])
        h = -1. * np.vstack([np.ones((n, 1)), np.zeros((n, 1))])
        return cvxopt_solve_qp(P=P, q=q, G=G, h=h)

    def fit(self, K, y):
        assert len(set(y)) == 2, "This class only supports binary classification."
        assert set(y) == {0, 1}, "Please transform labels into (0, 1) values."

        y_copy = np.array(y)
        for i, class_ in enumerate(y):
            if class_ == 0:
                y_copy[i] = -1
        if not self.dual:
            self.alpha, _ = np.split(self.solve_primal_svm(K, y_copy), 2)
        else:
            self.alpha = self.solve_dual_svm(K, y_copy)

    def predict(self, K_val):
        y_pred = np.zeros((K_val.shape[0],))
        f = K_val.dot(self.alpha)
        for i, value in enumerate(f):
            if value > 0.:
                y_pred[i] = 1
        return y_pred

    def score(self, K_val, y_true):
        """

        :param K_val: This is the kernel matrix between Validation features and Training features
        :param y_true: true labels
        :return: Accuracy score
        """
        score = 0
        y_pred = self.predict(K_val)
        n = len(y_true)
        assert n == y_pred.shape[0], "Something is wrong, y_true and y_pred have different lengths"
        for i in xrange(n):
            if y_pred[i] == y_true[i]:
                score += 1

        return score / float(n)
