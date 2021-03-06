import numpy as np
from msepm.base import EPMBase


class MultistateEpigeneticPacemaker(EPMBase):

    def __init__(self, iter_limit=100, n_jobs=1,
                 error_tolerance=0.001, learning_rate=0.01,
                 scale_X=False, verbose=False):
            EPMBase.__init__(self)
            self.iter_limit = iter_limit
            self.n_jobs = n_jobs
            self.error_tolerance = error_tolerance
            self.learning_rate = learning_rate
            self.scale_X = scale_X
            self.verbose = verbose

    def fit(self, X, Y, sample_weights=None):
        return self.fit_epm(X, Y, sample_weights, verbose=self.verbose)
