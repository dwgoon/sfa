
import sys
if sys.version_info <= (2, 8):
    from builtins import super

import sfa

from sfa.algorithms.gs import GaussianSmoothing

class NormGaussianSmoothing(GaussianSmoothing):
    def __init__(self, abbr="nGS"):
        super().__init__(abbr)
        self._name = "Gaussian smoothing algorithm with matrix normalization"

    def normalize(self, A):
        # Call normalize of GaussianSmoothing's parent (i.e, SignalPropagation)
        return super(GaussianSmoothing, self).normalize(A)
# end of def class


if __name__ == "__main__":

    algs = sfa.AlgorithmSet()
    algs.create(["GS", "SP"])

    algs["nGS"] = NormGaussianSmoothing()

    ds = sfa.DataSet()
    ds.create("BORISOV_2009")
    data = ds["BORISOV_2009"]["AUC_EGF=1+I=100"]


    for abbr, alg in algs.items():
        alg.params.is_rel_change = True
        alg.data = data
        alg.initialize()
        alg.compute_batch()
        acc = sfa.calc_accuracy(alg.result.df_sim, data.df_exp)
        print("%s: %f"%(abbr, acc))
    # end of for
# end of main