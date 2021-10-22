import os
import unittest
import numpy as np
from msepm.helpers import pearson_correlation

data_dir = os.path.dirname(os.path.realpath(__file__))

# import simulated control data
phenos = np.genfromtxt(f'{data_dir}/test_data/val_phenos.tsv', delimiter='\t')
meth = np.genfromtxt(f'{data_dir}/test_data/val_meth.tsv', delimiter='\t')
site_info = np.genfromtxt(f'{data_dir}/test_data/val_site_info.tsv', delimiter='\t')

# unpack data
ages, expected_states = phenos[:, 0], phenos[:, 1]
m_nots, rates = site_info[:, 0], site_info[:, 1]






class TestHelper(unittest.TestCase):

    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()