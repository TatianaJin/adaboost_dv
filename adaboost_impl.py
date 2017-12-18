# coding: utf-8

# In[1]:

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.core.umath_tests import inner1d

from six import with_metaclass
from six.moves import xrange

from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble.forest import BaseForest
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.tree import BaseDecisionTree
from sklearn.tree._tree import DTYPE
from sklearn.utils.validation import check_array, check_X_y, check_random_state, has_fit_parameter, check_is_fitted

from time import time

# In[2]:


class BaseWeightBoosting(with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for Adaboost classifier

    class BaseWeightBoosting inherits BaseEnsemble

    Attributes
    ----------
    learning_rate, random_state, estimator_weights_, estimator_errors_
    """

    # the same as in sklearn.ensemble.BaseWeightBoosting
    @abstractmethod
    def __init__(
        self,
        base_estimator=None,
        n_estimators=50,
        estimator_params=tuple(),
        learning_rate=1.,
        random_state=None
    ):
        super(BaseWeightBoosting, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params
        )
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _clear(self):
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        pass

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.

        y: array-like of shape = [n_samples]
            The target values (class labels).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Returns self.
        """
        ##### CHECK PARAMETERS (Support classification only) #####
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(
            X, y, accept_sparse=accept_sparse, dtype=dtype, y_numeric=False
        )

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight.fill(1. / X.shape[0])
        else:
            sample_weight = check_array(
                sample_weight, accept_sparse=False, ensure_2d=False
            )
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples."
                )

        self._validate_estimator()
        random_state = check_random_state(self.random_state)
        ########## END OF CHECK PARAMETERS ##########

        # Clear previous states
        self._clear()

        ###### BOOSTING #####
        for iboost in xrange(self.n_estimators):
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, y, sample_weight, random_state
            )

            # Early termination
            if sample_weight is None:  # None when the base classifier is worse than random
                break

            # estimator_errors seem not be used but only recorded
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Early termination
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)
            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        ###### END OF BOOSTING #####

        return self

    def staged_score(self, X, y, sample_weight=None):
        for y_pred in self.staged_predict(X):
            yield accuracy_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError(
                "Estimator not fitted, "
                "call `fit` before `feature_importances_`."
            )

        try:
            norm = self.estimator_weights_.sum()
            return (
                sum(
                    weight * clf.feature_importances_
                    for weight, clf in
                    zip(self.estimator_weights_, self.estimators_)
                ) / norm
            )

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute"
            )

    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format"""
        if (self.base_estimator is None or
                isinstance(self.base_estimator,
                           (BaseDecisionTree, BaseForest))):
            X = check_array(X, accept_sparse='csr', dtype=DTYPE)

        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        return X


# In[3]:


def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].
    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
    """
    proba = estimator.predict_proba(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
    log_proba = np.log(proba)

    return (n_classes - 1) * (
        log_proba - (1. / n_classes) * log_proba.sum(axis=1)[:, np.newaxis]
    )


# In[4]:


class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):
    """
    Attributes (self)
    ----------
    algorithm:   the adaboost algorithm to use
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=50,
        learning_rate=1,
        algorithm='BINARY',  # default original adaboost
        random_state=None,
        dv_interval=5,
        dv_solver=None,
    ):
        """dv_solver should set warm_start=True, fit_intercept=False"""

        # If algorithm is DYNAMIC_VOTES,
        # the default is to adjust votes only at the end of training
        super(AdaBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
        self.algorithm = algorithm
        self.dv_interval = dv_interval
        if dv_solver == None:
            # default solver
            self.dv_solver = LogisticRegression(fit_intercept=False, max_iter=10, penalty='l2', C=1, warm_start=True)
        else:
            self.dv_solver=dv_solver

    def fit_test(self, X, y, X_test, y_test):
        """ test after each iteration for monitoring """

        start = time()
        testing_time = 0

        # Initialize weights to 1 / n_samples
        sample_weight = np.empty(X.shape[0], dtype=np.float64)
        sample_weight.fill(1. / X.shape[0])

        ##### CHECK PARAMETERS (Support classification only) #####
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(
            X, y, accept_sparse=accept_sparse, dtype=dtype, y_numeric=False
        )
        X_test, y_test = check_X_y(
            X_test,
            y_test,
            accept_sparse=accept_sparse,
            dtype=dtype,
            y_numeric=False
        )

        self._validate_estimator()
        random_state = check_random_state(self.random_state)
        ########## END OF CHECK PARAMETERS ##########

        # Clear previous states
        self._clear()

        ###### BOOSTING #####
        for iboost in xrange(self.n_estimators):
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, y, sample_weight, random_state
            )

            # Early termination
            if sample_weight is None:  # None when the base classifier is worse than random
                break

            # estimator_errors seem not be used but only recorded
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            start_testing = time()
            # test on testing data
            error = (self.predict(X_test) != y_test).sum() / len(y_test)
            # try training data error = (self.predict(X) != y).sum() / len(y)
            print("[Iter %s] error=%s" % (iboost, error))
            testing_time += time() - start_testing

            # Early termination
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)
            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        ###### END OF BOOSTING #####

        training_time = time() - start - testing_time
        print("fitting time: %ss" % training_time)
        return self

    def fit(self, X, y, sample_weight=None):
        # Check that the algorithm is supported
        if self.algorithm not in (
            'SAMME.R', 'SAMME', 'DYNAMIC_VOTES', 'BINARY'
        ):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
        return super(AdaBoostClassifier, self).fit(X, y, sample_weight)

    def _validate_estimator(self):
        super(
            AdaBoostClassifier, self
        )._validate_estimator(default=DecisionTreeClassifier(max_depth=1))

        # SAMME-R requires predict_proba-enabled base estimators
        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator_, 'predict_proba'):
                raise TypeError(
                    "AdaBoostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead."
                )
        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError(
                "%s doesn't support sample_weight." %
                self.base_estimator_.__class__.__name__
            )

        if self.algorithm == "DYNAMIC_VOTES":
            # Validate dv_interval
            if self.dv_interval <= 0:
                raise ValueError("dv_interval must be postive")

    def _boost(self, iboost, X, y, sample_weight, random_state):
        if self.algorithm == 'SAMME.R':
            return self._boost_real(iboost, X, y, sample_weight, random_state)

        elif self.algorithm == 'BINARY':
            return self._boost_binary(iboost, X, y, sample_weight, random_state)

        elif self.algorithm == 'DYNAMIC_VOTES':
            return self._boost_dv(iboost, X, y, sample_weight, random_state)

        else:  # elif self.algorithm == "SAMME":
            return self._boost_discrete(
                iboost, X, y, sample_weight, random_state
            )

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)

        # Predict class probabilities
        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Get the class with largest probability
        y_predict = self.classes_.take(
            np.argmax(y_predict_proba, axis=1), axis=0
        )

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0)
        )

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (
            -1. * self.learning_rate *
            (((n_classes - 1.) / n_classes
              ) * inner1d(y_coding, np.log(y_predict_proba)))
        )

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(
                estimator_weight *
                ((sample_weight > 0) | (estimator_weight < 0))
            )

        return sample_weight, 1., estimator_error

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0)
        )

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    'BaseClassifier in AdaBoostClassifier '
                    'ensemble is worse than random, ensemble '
                    'can not be fit.'
                )
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.)
        )

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(
                estimator_weight * incorrect *
                ((sample_weight > 0) | (estimator_weight < 0))
            )

        return sample_weight, estimator_weight, estimator_error

    def _boost_binary(
        self, iboost, X, y, sample_weight, random_state, keep_predict=False
    ):
        """Implement a single boost for binary classification."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)
            if self.n_classes_ is not 2:
                raise ValueError(
                    "The binary classification algorithm"
                    "only supports two classes"
                )

        # Record the past predictions for DV
        classes = self.classes_[0] # class 0 is -1
        if keep_predict:
            if iboost == 0:
                self.predicts_ = np.array([(y_predict == classes)*-2+1])
            else:
                self.predicts_ = np.append(
                    self.predicts_, np.array([(y_predict == classes)*-2+1]), axis=0
                )

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.average(incorrect, weights=sample_weight, axis=0)

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Convert to 1 / -1 from 1 / 0
        incorrect = incorrect * 2 - 1

        # Stop if the error is as bad as random guessing
        if estimator_error == 0.5:
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    'BaseClassifier in AdaBoostClassifier '
                    'ensemble is as random, ensemble '
                    'can not be fit.'
                )
            return None, None, None

        estimator_weight = 0.5 * (
            np.log((1. - estimator_error) / estimator_error)
        )

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(
                estimator_weight * incorrect *
                ((sample_weight > 0) | (estimator_weight < 0))
            )

        return sample_weight, estimator_weight, estimator_error

    def _boost_dv(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost adjusting all votes every
        dv_interval iterations"""
        sample_weight, estimator_weight, estimator_error = self._boost_binary(
            iboost, X, y, sample_weight, random_state, keep_predict=True
        )

        if (iboost + 1) % self.dv_interval == 0:  # adjust the votes
            print("[Iter %s] Dynamically adjusting votes" % iboost)

            X_map = self.predicts_.T

            # 1. Train the votes
            vote_model = self.dv_solver

            coef_init = np.array([
                np.append(self.estimator_weights_[:iboost], estimator_weight)
            ])

            # Better that warm_start is True
            vote_model.coef_ = coef_init

            # Assume equal initial sample weights for simplicity
            vote_model.fit(X_map, y)

            # Scale the votes to keep constant sum
            self.estimator_weights_[:iboost + 1] = vote_model.coef_[
                0
            ] / vote_model.coef_[0].sum() * coef_init.sum()
            estimator_weight = self.estimator_weights_[iboost]

            # Calculate sample weights (assume equal initial sample weights)
            sample_weight = np.exp(
                X_map.dot(self.estimator_weights_[:iboost + 1]) * ((y == self.classes_[0]) * -2 + 1) * -1
            )

        # NOTICE: the estimator error reports error irresponsive to dynamic votes
        return sample_weight, estimator_weight, estimator_error

    def decision_function(self, X):
        """Compute the decision function of ``X``.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(
                _samme_proba(estimator, n_classes, X)
                for estimator in self.estimators_
            )
        else:  # self.algorithm in ("SAMME", "BINARY", "DYNAMIC_VOTES")
            pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in
                       zip(self.estimators_, self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.
        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_, self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_pred = estimator.predict(X)
                current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm

    def predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        check_is_fitted(self, "n_classes_")

        n_classes = self.n_classes_
        X = self._validate_X_predict(X)

        if n_classes == 1:
            return np.ones((X.shape[0], 1))

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(
                _samme_proba(estimator, n_classes, X)
                for estimator in self.estimators_
            )
        else:  # self.algorithm == "SAMME"
            proba = sum(
                estimator.predict_proba(X) * w
                for estimator, w in
                zip(self.estimators_, self.estimator_weights_)
            )

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba

    def staged_predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.
        This generator method yields the ensemble predicted class probabilities
        after each iteration of boosting and therefore allows monitoring, such
        as to determine the predicted class probabilities on a test set after
        each boost.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        p : generator of array, shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        proba = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_, self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_proba = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_proba = estimator.predict_proba(X) * weight

            if proba is None:
                proba = current_proba
            else:
                proba += current_proba

            real_proba = np.exp((1. / (n_classes - 1)) * (proba / norm))
            normalizer = real_proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            real_proba /= normalizer

            yield real_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.
        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        return np.log(self.predict_proba(X))

    def predict(self, X):
        """Predict classes for X.
        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def staged_predict(self, X):
        """Return staged predictions for X.
        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.
        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : generator of array, shape = [n_samples]
            The predicted classes.
        """
        n_classes = self.n_classes_
        classes = self.classes_

        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))

        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(np.argmax(pred, axis=1), axis=0))
