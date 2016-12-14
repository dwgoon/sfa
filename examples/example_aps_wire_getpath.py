



import sfa

if __name__ == "__main__":

    algs = sfa.AlgorithmSet()
    algs.create("APS")
    alg = algs["APS"]

    ds = sfa.DataSet()
    ds.create("NELANDER_2008")
    data = ds["NELANDER_2008"]

    alg.data = data
    CE, paths = alg.wire(["EGFR"], [1.0], getpath=True)

    print("Combined effect: ", CE)
    print("Paths identified by pathway wiring algorithm:")
    for i, p in enumerate(paths):
        print ("[#{}] {}".format(i+1, p))


    alg.params.max_path_length = 6
    CE, paths = alg.wire(["EGFR"], [1.0], getpath=True)
    print("Combined effect: ", CE)
    print("Paths identified by pathway wiring algorithm")
    print("(max. path length: {})".format(alg.params.max_path_length))
    for i, p in enumerate(paths):
        print("[#{}] {}".format(i+1, p))