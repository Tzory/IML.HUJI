from typing import NoReturn
from ...base import BaseEstimator
from ..gaussian_estimators import *
import numpy as np
from numpy.linalg import det, inv
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
from IMLearn.metrics import *



class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape[0], X.shape[1]
        self.classes_ = np.unique(y)
        clas_num = self.classes_.shape[0]
        self.pi_ = np.zeros(clas_num)
        self.mu_ = np.zeros((clas_num, n_features))
        self.cov_ = np.zeros((n_features, n_features))
        for i, label in enumerate(self.classes_):
            self.pi_[i] = np.mean(y == label)
            self.mu_[i] = np.mean(X[y == label], axis=0)
            self.cov_ += (X[y == label] - self.mu_[i]).T @ (X[y == label] - self.mu_[i])

        # making the cov unbiased
        self.cov_ /= (n_samples - clas_num)
        self._cov_inv = np.linalg.inv(self.cov_)
        self.fitted_ = True


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        matrix = self.likelihood(X)
        matrix = matrix * self.pi_
        location = np.argmax(matrix, axis=1)
        ret = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            ret[i] = self.classes_[location[i]]
        return ret


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        multiGaus = MultivariateGaussian()

        samp_num = X.shape[0]
        clas_num = self.classes_.shape[0]
        ret = np.zeros((samp_num, clas_num))
        multiGaus.fitted_ = True

        for i, num in enumerate(self.classes_):
            multiGaus.mu_ = self.mu_[i]
            multiGaus.cov_ = self.cov_

            ret[:, i] = multiGaus.pdf(X).T

        return ret * self.pi_



    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(X, y)
