from typing import Tuple
import joblib
import numpy as np

from msepm.helpers import get_fold_step_size


def lstsq(X: np.ndarray, y: np.ndarray, rid: int = 0,
          fit_intercept: bool = True, weights: np.ndarray = None):
    """Solve $$Ax=B$$, by computing $$x$$, minimize $$||B-Ax||$$ using np.linalg.lsqrt,
    intercept and weight calculation implemented using a scaled back implementation of scikit-learn LinearRegression

    Params:
        * *X: (np.ndarray)*: array of m samples and n features
        * *y: (np.ndarray)*: vector of m target values
        * *rid: (int)*: regression ID to ensure results matrix constructed correctly if multi-threaded
        * *weights: (np.ndarray)*: weights for m samples, not currently implemented
    Returns:
        * *rid (int)*: regression ID
        * *_coefs (np.ndarray)*: least squares solution
        * *_res (np.ndarray)*: sums of squared residuals
        * *_rank (int)*: rank of matrix A
        *_sin_vals (np.ndarray)*: singular values of matrix A
        * *_intercept (float)*: regression intercept
    """
    #ToDo: support weighted sample fitting
    X_fit, y_fit = np.copy(X, order='k'), np.copy(y, order='k')
    X_offset, y_offset = 0.0, 0.0
    if fit_intercept or weights is not None:
        X_offset, y_offset = np.average(X_fit, axis=0, weights=weights), np.average(y_fit, axis=0, weights=weights)
        X_fit -= X_offset
        y_fit -= y_offset
    _coefs, _res, _rank, _sin_vals = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    _intercept = 0.0 if not fit_intercept else y_offset - np.dot(X_offset, _coefs.T)
    return rid, _coefs, _res, _rank, _sin_vals, _intercept


def construct_lstsq_solutions_matrix(system_solutions):
    """Unpack system of regression models
    Params:
        * *systems_solutions (Tuple)*: tuple of lstsq results
    Returns:
        * *coefs (np.ndarray)*: array of regression coefficients
        * *intercepts (np.ndarray)*: vector of regression intercepts
        * *sum_res (float)*: sum of all regression model sum of squared residuals
    """
    # construct least square solution matrix
    coefs = np.zeros((len(system_solutions), len(system_solutions[0][1])))
    # intercepts
    intercepts = np.zeros(len(system_solutions))
    # construct residuals matrix
    res = np.zeros(len(system_solutions))
    # construct using label to ensure matrix order
    for sol in system_solutions:
        coefs[sol[0]] = sol[1].reshape(1, -1)
        intercepts[sol[0]] = sol[5]
        res[sol[0]] = sol[2]
    return coefs, intercepts, res


def solve_regression_system(X: np.ndarray, Y: np.ndarray, n_jobs: int = 1, fit_intercept: bool = True):
    """Fit linear least squares regression
     with every row of Y $$n \\times m$$ matrix

    """
    batch_size = get_fold_step_size(Y.shape[0], n_jobs)
    solutions = joblib.Parallel(n_jobs=n_jobs, batch_size=batch_size)(joblib.delayed(lstsq)(*[X, Y[row], rid,
                                                                                              fit_intercept, None]) for
                                               rid, row in enumerate(range(Y.shape[0])))
    return construct_lstsq_solutions_matrix(solutions)


def predict_epm_states(epm_coefs, epm_intercepts, Y) -> Tuple[np.ndarray, np.ndarray]:
    # predict epm results with trained system
    rid, _coefs, _res, _rank, _sin_vals, _intercept = lstsq(epm_coefs, Y - epm_intercepts.reshape(-1, 1),
                                                            fit_intercept=False)
    return _coefs.T


def get_site_values(epm_coefs, epm_intercepts, X):
    return epm_intercepts.reshape(-1, 1) + np.dot(epm_coefs, X.T)


def get_state_gradient(coefs, intercepts, X, Y):
    neg_residuals = Y - get_site_values(coefs, intercepts, X)
    grad = np.ones(X.shape)
    for p in range(X.shape[1]):
        grad[:, p] = -2 * np.sum(coefs[:, p].reshape(-1, 1) * neg_residuals, axis=0) / neg_residuals.shape[0]
    return grad
