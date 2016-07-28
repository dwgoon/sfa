# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:15:07 2016

@author: dwlee
"""

import os
import sys

pkg_path = os.path.abspath(os.path.join('..'))
sys.path.append(pkg_path)

from sfa import JobManager
from sfa import Algorithm

if __name__ == "__main__":
    alg = Algorithm()    
    mgr = JobManager()
    
    mgr.add_algorithm(alg)