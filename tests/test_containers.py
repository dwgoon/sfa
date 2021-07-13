import sfa

# Test AlgorithmSet
algs = sfa.AlgorithmSet()
keys = algs.get_all_keys()
assert len(list(keys)) != 0

keys = algs.get_all_keys()
assert len(list(keys)) != 0