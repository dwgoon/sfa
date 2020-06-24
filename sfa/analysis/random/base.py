# -*- coding: utf-8 -*-

from multiprocessing import Pool

import numpy as np
import pandas as pd

import sfa


class BaseRandomSimulator(object):

    def __init__(self):
        # Consider RuntimeWarning from NumPy as an error
        np.seterr(all='raise')

    def _initialize(self, alg):
        self._S = np.sign(alg.data.A)  # Sign matrix
        self._ir, self._ic = alg.data.A.to_numpy().nonzero()
        self._num_links = self._ir.size
        self._A = np.array(alg.data.A)
        self._W = np.zeros_like(self._A, dtype=np.float)

    def _randomize(self):
        raise NotImplementedError()

    def _apply_norm(self, alg, use_norm):
        """
        Apply normalization
        """
        if use_norm:
            alg.W = sfa.normalize(self._W)
        else:
            alg.W = self._W

    # end of def

    def _need_to_print(self, use_print, freq_print, cnt):
        return use_print and (cnt % freq_print == 0)

    def simulate_single(*args, **kwargs):
        raise NotImplementedError()

    def simulate_multiple(*args, **kwargs):
        raise NotImplementedError()

    """
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
            print("%s simulation for %s starts..." % (alg.abbr, data.abbr))

        while cnt < num_samp:
            self._randomize()
            self._apply_norm(alg, use_norm)
            try:
                alg.compute_batch()
                acc = sfa.calc_accuracy(self._alg.result.df_sim,
                                        self._alg.data.df_exp)
            except FloatingPointError as pe:
                # Skip this condition
                if self._need_to_print(use_print, freq_print, cnt):
                    print("%s: skipped..." % (pe))
                continue
            except RuntimeWarning as rw:
                # Skip these weights
                if self._need_to_print(use_print, freq_print, cnt):
                    print("%s: skipped..." % (rw))
                continue

                # Skip these weights assuming acc cannot be exactly 0.
            if acc == 0:
                if self._need_to_print(use_print, freq_print, cnt):
                    print("Zero accuracy: skipped...")
                continue

            results[cnt] = acc
            cnt += 1
            if self._need_to_print(use_print, freq_print, cnt):
                print("[Iteration #%d] acc: %f" % (cnt, acc))
        # end of loop
        df = pd.DataFrame(results)
        df.index = range(1, num_samp + 1)
        df.columns = [data.abbr]
        return df
    # end of def

    def simulate_single(self, num_samp, alg, data, use_norm=False,
                        use_print=False, freq_print=100):
        self._alg = alg
        self._alg.data = data
        self._alg.initialize()
        self._initialize(alg)

        df = self._simulate_single((num_samp, alg, data, use_norm,
                                    use_print, freq_print))
        return df

    # end of def

    def simulate_multiple(self, num_samp, alg, mdata, use_norm=False,
                          use_print=False, freq_print=100,
                          max_workers=1):
        self._alg = alg

        # Initialize network information only
        self._alg.data = sfa.get_avalue(mdata)
        self._alg.initialize()
        self._initialize(alg)

        if isinstance(mdata, list):
            list_data = [(data.abbr, data) for data in mdata]
        elif isinstance(mdata, dict):
            list_data = [(abbr, mdata[abbr]) for abbr in mdata]

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
            pool = Pool(processes=max_workers)
            dfs = list(pool.map(self._simulate_single, args))
            pool.close()
            pool.join()
        else:
            raise ValueError("max_workers should be a positive integer.")

        df_res = pd.concat(dfs, axis=1)
        return df_res
    # end of def
    """

# end of class


class BaseRandomBatchSimulator(BaseRandomSimulator):
    def __init__(self):
        super().__init__()

    def _randomize(self):
        raise NotImplementedError()

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
            print("%s simulation for %s starts..." % (alg.abbr, data.abbr))

        while cnt < num_samp:
            self._randomize()
            self._apply_norm(alg, use_norm)
            try:
                alg.compute_batch()
                acc = sfa.calc_accuracy(self._alg.result.df_sim,
                                        self._alg.data.df_exp)
            except FloatingPointError as pe:
                # Skip this condition
                if self._need_to_print(use_print, freq_print, cnt):
                    print("%s: skipped..." % (pe))
                continue
            except RuntimeWarning as rw:
                # Skip these weights
                if self._need_to_print(use_print, freq_print, cnt):
                    print("%s: skipped..." % (rw))
                continue

                # Skip these weights assuming acc cannot be exactly 0.
            if acc == 0:
                if self._need_to_print(use_print, freq_print, cnt):
                    print("Zero accuracy: skipped...")
                continue

            results[cnt] = acc
            cnt += 1
            if self._need_to_print(use_print, freq_print, cnt):
                print("[Iteration #%d] acc: %f" % (cnt, acc))
        # end of loop
        df = pd.DataFrame(results)
        df.index = range(1, num_samp + 1)
        df.columns = [data.abbr]
        return df
    # end of def

    def simulate_single(self, num_samp, alg, data, use_norm=False,
                        use_print=False, freq_print=100):
        self._alg = alg
        self._alg.data = data
        self._alg.initialize()
        self._initialize(alg)

        df = self._simulate_single((num_samp, alg, data, use_norm,
                                    use_print, freq_print))
        return df

    # end of def

    def simulate_multiple(self, num_samp, alg, mdata, use_norm=False,
                          use_print=False, freq_print=100,
                          max_workers=1):
        self._alg = alg

        # Initialize network information only
        self._alg.data = sfa.get_avalue(mdata)
        self._alg.initialize()
        self._initialize(alg)

        if isinstance(mdata, list):
            list_data = [(data.abbr, data) for data in mdata]
        elif isinstance(mdata, dict):
            list_data = [(abbr, mdata[abbr]) for abbr in mdata]

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
            pool = Pool(processes=max_workers)
            dfs = list(pool.map(self._simulate_single, args))
            pool.close()
            pool.join()
        else:
            raise ValueError("max_workers should be a positive integer.")

        df_res = pd.concat(dfs, axis=1)
        return df_res
    # end of def
# end of class
