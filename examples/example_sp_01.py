# -*- coding: utf-8 -*-
"""
@author: dwlee
"""



import sfa
from sfa.algorithms import sp


if __name__ == "__main__":
    
    
    data = Data("Borisov")
    data.A # Adjacency matrix
    data.df_ba # Basal activity data
    data.df_exp # Experimental result
    
    alg = sp.create_algorithm()
        
    
    