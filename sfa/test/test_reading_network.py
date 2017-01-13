# -*- coding: utf-8 -*-

import os
import unittest

import networkx as nx

import sfa
import sfa.data


class TestReadingNetwork(unittest.TestCase):
    
    def test_read_network(self):
        dpath = sfa.data.__path__[0]
        fpath = os.path.join(dpath, "borisov_2009", "network.sif")
        A, n2i, dg = sfa.read_sif(fpath, as_nx=True)

        self.assertTrue( isinstance(dg, nx.DiGraph) )        
        self.assertEqual(A.shape[0], dg.number_of_nodes())
        self.assertEqual(A.nonzero()[0].size, dg.number_of_edges())
    # end of def test_read_network

if __name__ == "__main__":
    unittest.main(verbosity=2)
