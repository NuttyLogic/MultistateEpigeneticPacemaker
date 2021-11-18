import os
import unittest
import numpy as np
from tqdm import tqdm
from msepm.msepm import MultistateEpigeneticPacemaker
from msepm.msepm_cv import MultistateEpigeneticPacemakerCV


data_dir = os.path.dirname(os.path.realpath(__file__))

# import simulated control data
phenos = np.genfromtxt(f'{data_dir}/test_data/val_phenos.tsv', delimiter='\t')
meth = np.genfromtxt(f'{data_dir}/test_data/val_meth.tsv', delimiter='\t')
site_info = np.genfromtxt(f'{data_dir}/test_data/val_site_info.tsv', delimiter='\t')

# unpack data
ages, expected_states = phenos[:, 0], phenos[:, 1]
m_nots, rates = site_info[:, 0], site_info[:, 1]

# fit epm


epm_cv = MultistateEpigeneticPacemakerCV(cv_folds=4, learning_rate=0.1,
                                         scale_X=True, verbose=True)
epm_cv_predictions = epm_cv.fit(ages, meth, return_out_of_fold_predictions=True)

epm_cv_ran = MultistateEpigeneticPacemakerCV(cv_folds=4, learning_rate=0.1,
                                             scale_X=True, verbose=True,
                                             randomize_sample_order=True)
epm_cv_ran_predictions = epm_cv_ran.fit(ages, meth, return_out_of_fold_predictions=True)

fold_predictions = []
for step in tqdm(range(4), desc='EPM Validation Folds'):

    # set fold test / train samples
    test_indices = [count + 250 * step for count in range(250)]
    train_indices = [count for count in range(1000) if count not in test_indices]

    # fit epm
    epm = MultistateEpigeneticPacemaker(learning_rate=0.1, scale_X=True, verbose=False)
    epm.fit(ages[train_indices], meth[:, train_indices])
    fold_predictions.append(epm.predict(meth[:, test_indices]))

fold_predictions = np.array(fold_predictions).flatten()


class TestMSEPMCV(unittest.TestCase):

    def setUp(self):
        pass

    def test_fold_prediction(self):
        """Check that cv gives same age prediction"""
        for cv_age, epm_age in zip(epm_cv_predictions[:, 0], fold_predictions):
            self.assertAlmostEqual(cv_age, epm_age)

    def test_ran_fold_prediction(self):
        """Check that cv gives same age prediction"""
        for cv_age, ran_cv_age in zip(epm_cv_predictions[:, 0],
                                      epm_cv_ran_predictions[:, 0]):
            self.assertAlmostEqual(cv_age, ran_cv_age, 2)

    def test_site_rates(self):
        for epm_rate, sim_rate in zip(epm_cv._coefs[:, 0], rates):
            self.assertAlmostEqual(epm_rate, sim_rate, 1)

    def test_site_intercepts(self):
        for epm_intercept, m_not in zip(epm_cv._intercepts, m_nots):
            self.assertAlmostEqual(epm_intercept, m_not, 1)


if __name__ == '__main__':
    unittest.main()
