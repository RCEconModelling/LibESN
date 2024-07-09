#
#
#

import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from libesn_legacy.base_functions import *

class TestBaseFunctions(unittest.TestCase):

    def test_nb_identity(self):
        self.assertEqual(nb_identity(1), 1, "Should be 1")

    def test_nb_radbas(self):
        self.assertEqual(nb_radbas(2), np.exp(-4), "Should be exp(-4)")

if __name__ == '__main__':
    unittest.main()