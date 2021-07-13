import time

import sfa
from sfa.analysis import RandomStructureBatchSimulator

if __name__ == "__main__":
    aset = sfa.AlgorithmSet()
    aset.create("SP")
    alg = aset["SP"]
    alg.params.initialize()
    alg.params.exsol_forbidden = True
    alg.params.use_rel_change = True
    alg.params.alpha = 0.5

    abbr = "BORISOV_2009"
    ds = sfa.DataSet()
    mdata = ds.create(abbr)

    rs = RandomStructureBatchSimulator(nswap=200, noself=True)

    t_beg = time.time()
    df_res = rs.simulate_multiple(10000,
                                  alg,
                                  mdata,
                                  use_norm=False,
                                  use_print=True,
                                  freq_print=100,
                                  max_workers=4)

    df_desc = df_res.describe().T
    t_end = time.time()

    print(df_desc)
    print("Time elapsed: ", t_end - t_beg)

    df_res.to_csv(
        "%s_result_randomized_structs.csv" % (abbr.lower()))
    df_desc.to_csv(
        "%s_desc_result_randomized_structs.csv" % (abbr.lower()))
