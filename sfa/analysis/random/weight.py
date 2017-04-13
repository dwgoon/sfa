# -*- coding: utf-8 -*-


import time
# from concurrent.futures import ProcessPoolExecutor
# from pathos.pools import ParallelPool as Pool
from multiprocessing import Pool


import numpy as np
import pandas as pd
import networkx as nx

import sfa

class RandomWeightSimulator(object):
    
    def __init__(self, bounds=(-3, 0)):
        self.lb = bounds[0]
        self.ub = bounds[1]
        
        # Consider RuntimeWarning from NumPy as an error
        np.seterr(all='raise')

    def _initialize(self, alg):    
        self.S = np.sign(alg.data.A)  # Sign matrix
        self.ir, self.ic = alg.data.A.nonzero()
        self.num_links = self.ir.size
        self.W_rand = np.zeros_like(alg.data.A, dtype=np.float)
                
    def _randomize_weights(self):
        weights_rand = 10**np.random.uniform(self.lb, self.ub,
                                             size=(self.num_links,))
        for i in range(self.num_links):    
            p, q = self.ir[i], self.ic[i]
            self.W_rand[p, q] = weights_rand[i]*self.S[p, q]
        # end of for
            
    def _apply_norm(self, alg, use_norm):
        """
        Apply normalization
        """
        if use_norm:
            alg.W = sfa.normalize(self.W_rand)
        else:
            alg.W = self.W_rand
        

    # end of def

    def _need_to_print(self, use_print, freq_print, cnt):
        return use_print and (cnt%freq_print == 0)
        

    def _simulate_single(self, args):
        num_samp = args[0]
        alg = args[1]    
        data = args[2]
        use_norm = args[3]
        use_print = args[4]
        freq_print = args[5]
    
        alg.data = data
        alg.initialize(network=False)                         
                         
        results = np.zeros((num_samp,), dtype=np.float)
        cnt = 0
        
        if self._need_to_print(use_print, freq_print, cnt):
            print ("%s simulation for %s starts..."%(alg.abbr, data.abbr))
            
        while cnt<num_samp:
            self._randomize_weights()
            
            # No norm
            self._apply_norm(alg, use_norm)                
            try:
                alg.compute_batch()
                acc = sfa.calc_accuracy(self.alg.result.df_sim,
                                        self.alg.data.df_exp)
            except FloatingPointError as pe:
                # Skip these weights
                if self._need_to_print(use_print, freq_print, cnt):
                    print ("%s: skipped..."%(pe))
                continue        
            except RuntimeWarning as rw:
                # Skip these weights
                if self._need_to_print(use_print, freq_print, cnt):
                    print ("%s: skipped..."%(rw))
                continue        
            
            # Skip these weights assuming acc cannot be exactly 0.
            if acc == 0:
                if self._need_to_print(use_print, freq_print, cnt):
                    print ("Zero accuracy: skipped...")
                continue
            
            results[cnt] = acc            
            cnt += 1
            if self._need_to_print(use_print, freq_print, cnt):
                print ("[Iteration #%d] acc: %f"%(cnt, acc))
        # end of loop
        df = pd.DataFrame(results)
        df.index = range(1, num_samp+1)
        df.columns = [data.abbr]
        return df

    def simulate_single(self, num_samp, alg, data, use_norm=False,
                        use_print=False, freq_print=100):
        self.alg = alg
        self.alg.data = data
        self.alg.initialize()
        self._initialize(alg)

        df = self._simulate_single((num_samp, alg, data, use_norm,
                                   use_print, freq_print))
        return df
    # end of def
        
    def simulate_multiple(self, num_samp, alg, mdata, use_norm=False,
                          use_print=False, freq_print=100,
                          max_workers=1):
        self.alg = alg                

        # Initialize network information only
        self.alg.data = sfa.get_avalue(mdata)
        self.alg.initialize()
        self._initialize(alg)
        
        if isinstance(mdata, list):
            list_data = [(data.abbr, data) for data in mdata]
        elif isinstance(mdata, dict):
            list_data = [(abbr, mdata[abbr]) for abbr in mdata]

        #columns = []        
        dfs = []
        if max_workers == 1:
            for (abbr, data) in list_data:
                df = self._simulate_single(num_samp, alg, data, use_norm,
                                           use_print, freq_print)
                dfs.append(df)
            # end of for
        elif max_workers > 1:
            args = ((num_samp, alg, data, use_norm, use_print, freq_print)
                                            for (abbr, data) in list_data)
            #pool = ProcessPoolExecutor(max_workers=max_workers)
            pool = Pool(processes=max_workers)
            #pool.ncpus = max_workers
            dfs = list(pool.map(self._simulate_single, args))
            pool.close()
            pool.join()
        else:
            raise ValueError("max_workers should be a positive integer.")
        
        df_res = pd.concat(dfs, axis=1)
        return df_res
    # end of def
# end of class

if __name__ == "__main__":
    aset = sfa.AlgorithmSet()
    aset.create("SP")
    alg = aset["SP"]
    alg.params.initialize()
    alg.params.exsol_forbidden = True
    alg.params.use_rel_change = True
    alg.params.alpha = 0.5
    
    ds = sfa.DataSet()
    ds.create("SCHLIEMANN_2011")
    mdata = ds["SCHLIEMANN_2011"]
    #data = mdata["AUC_EGF=1+I=100"]
    
    rws = RandomWeightSimulator(bounds=(-3,3))
    #df_res = rws.simulate_single(100, alg, data, use_norm=False)
    t_beg = time.time()

    df_res = rws.simulate_multiple(10000, alg, mdata, use_norm=False,
                                   use_print=True, freq_print=10,
                                   max_workers=25)
    df_desc = df_res.describe().T
    t_end = time.time()
    
    print(df_desc)
    print("Time elapsed: ", t_end - t_beg)
    
    df_res.to_csv("%s_result_randomized_weights(-3_3).csv"%(mdata.abbr.lower()))
    df_desc.to_csv("%s_desc_result_randomized_weights(-3_3).csv"%(mdata.abbr.lower()))
