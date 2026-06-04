# Simulation

The `sfa.analysis` module provides tools for running large simulations on
randomized variants of a network. Each randomization perturbs either the
topology (which links exist) or the link weights, runs the chosen
algorithm against the experimental data, and records the accuracy.

## Randomizing network structure

The low-level functions in `sfa.utils` operate directly on adjacency
matrices and are used by the batch simulators below.

| Function                                        | What it does                                        |
|-------------------------------------------------|-----------------------------------------------------|
| `sfa.rand_flip(A, nsamp, pivots, inplace)`      | Flip the sign of `nsamp` randomly chosen links.     |
| `sfa.rand_swap(A, nsamp, noself, pivots, ...)`  | Rewire `nsamp` links by swapping endpoints.         |
| `sfa.rand_structure(A, nswap, nflip, ...)`      | Combine flipping and swapping in one call.          |
| `sfa.rand_weights(W, lb, ub, inplace)`          | Sample link magnitudes from $10^{[lb,\, ub]}$.      |

```python
import numpy as np
import sfa

A = data.A.astype(np.float64)
B = sfa.rand_structure(A, nswap=10, nflip=5)  # 10 swaps + 5 sign flips
W = sfa.rand_weights(B, lb=-3, ub=0)          # |w| ~ 10^U(-3, 0)
```

`pivots` restricts the operation to a set of node indices; `noself`
forbids self-loops; `inplace=True` mutates the input matrix instead of
allocating a new one.

## Batch simulators

The simulators in `sfa.analysis` wrap an algorithm-data combination and
run many randomized iterations against it, computing accuracy with
`sfa.calc_accuracy` and skipping iterations whose simulation overflows
or produces zero accuracy.

### Randomizing structure

`sfa.RandomStructureBatchSimulator` rewires the network on every
iteration using `rand_flip` + `rand_swap`.

```python
import sfa
from sfa.analysis import RandomStructureBatchSimulator

sim = RandomStructureBatchSimulator(nswap=10, nflip=5, noself=True)
df = sim.simulate_single(
    num_samp=1000,
    alg=alg,
    data=data,
    use_norm=True,      # Re-normalize the rewired matrix each iter.
    use_print=True,
    freq_print=100,
)
```

### Randomizing weights

`sfa.RandomWeightBatchSimulator` keeps the topology fixed and samples
new link magnitudes on every iteration.

```python
from sfa.analysis import RandomWeightBatchSimulator

sim = RandomWeightBatchSimulator(bounds=(-3, 0))
df = sim.simulate_single(1000, alg, data, use_norm=True)
```

The returned `df` is a `pandas.DataFrame` of accuracies indexed by
iteration number; pass it to `pandas` plotting or your favorite stats
tool.

## Multiple datasets

`simulate_multiple` runs the same algorithm against several datasets and
concatenates the per-dataset accuracy columns into one DataFrame.

```python
mdata = sfa.DataSet().create('BORISOV_2009')   # dict of data objects

df_res = sim.simulate_multiple(
    num_samp=1000,
    alg=alg,
    mdata=mdata,
    use_norm=True,
    use_print=True,
    freq_print=100,
)
```

The result has one column per dataset abbreviation, so you can compare
accuracy distributions across conditions directly.

## Parallel processing

`simulate_multiple` supports `max_workers > 1` to run datasets in
parallel using `multiprocessing.Pool`.

```python
df_res = sim.simulate_multiple(
    num_samp=1000,
    alg=alg,
    mdata=mdata,
    max_workers=4,
)
```

A few practical notes:

- Each worker re-runs the simulator on its own dataset, so the speedup
  scales with the number of datasets, not the number of iterations.
- The pool is created and joined inside `simulate_multiple`; nothing has
  to be done at the call site beyond setting `max_workers`.
- On Windows, the standard `if __name__ == "__main__":` guard is
  required when invoking parallel simulations from a script.

## Perturbation-only analysis

If you only need a single before/after comparison rather than a random
ensemble, `sfa.analyze_perturb` returns the activity change, the signal
flow change, and (optionally) the activity trajectory for a given list
of perturbation targets.

```python
from sfa.analysis import analyze_perturb

act, F = analyze_perturb(alg, data, targets=['MEK'])
```
