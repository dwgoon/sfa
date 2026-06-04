# Simulation

`sfa.analysis` provides tools for running large simulations on
randomized network variants. Each randomization perturbs either the
topology or the link weights, runs the chosen algorithm against the
experimental data, and records the accuracy.

## Setup

The examples below assume `alg` and `data` are prepared as in
[Signal flow analysis](tutorial_sfa.md):

```python
import sfa

algs = sfa.AlgorithmSet()
ds = sfa.DataSet()
data = sfa.get_avalue(ds.create('BORISOV_2009'))

alg = algs.create('SP')
alg.params.apply_weight_norm = True
alg.data = data
alg.initialize()
```

## Randomizing network structure

The helpers in `sfa.utils` operate directly on adjacency matrices:

| Function                                        | What it does                                        |
|-------------------------------------------------|-----------------------------------------------------|
| `sfa.rand_flip(A, nsamp, pivots, inplace)`      | Flip the sign of `nsamp` randomly chosen links.     |
| `sfa.rand_swap(A, nsamp, noself, pivots, ...)`  | Rewire `nsamp` links by swapping endpoints.         |
| `sfa.rand_structure(A, nswap, nflip, ...)`      | Combine flipping and swapping in one call.          |
| `sfa.rand_weights(W, lb, ub, inplace)`          | Sample link magnitudes from $10^{[lb,\, ub]}$.      |

```python
import numpy as np
import sfa

A = data.A.astype(np.float)
B = sfa.rand_structure(A, nswap=10, nflip=5)
W = sfa.rand_weights(B, lb=-3, ub=0)
```

!!! warning "v0.1.0 quirks"
    - `rand_swap(..., inplace=True)` falls through without an explicit
      return, so `rand_structure(..., inplace=True)` also returns
      `None`. Use `inplace=False` (default) and capture the return.
    - `rand_flip(..., pivots=...)` treats `pivots` (node indices) as
      edge indices, which can produce wrong flips or `IndexError`.
      Drop the `pivots` argument or upgrade to v0.2.x for the fix.

## Batch simulators

`sfa.analysis.RandomStructureBatchSimulator` and
`sfa.analysis.RandomWeightBatchSimulator` wrap an algorithm-data
combination and run many randomized iterations against it,
computing accuracy with `sfa.calc_accuracy` and skipping iterations
whose simulation overflows or produces zero accuracy.

```python
from sfa.analysis import RandomStructureBatchSimulator

sim = RandomStructureBatchSimulator(nswap=10, nflip=5, noself=True)
df = sim.simulate_single(
    num_samp=1000,
    alg=alg,
    data=data,
    use_norm=True,
    use_print=True,
    freq_print=100,
)
```

!!! warning "v0.1.0 known issues"
    - The simulator starts with an all-zero internal `_W`. As a
      result, `RandomWeightBatchSimulator` finds no nonzero positions
      to sample, so weight randomization runs as a no-op.
    - `RandomStructureBatchSimulator` writes new nonzeros into `_W`
      without clearing stale entries, so link patterns accumulate
      across iterations.
    These were fixed in v0.2.x by seeding `_W` from the signed
    adjacency and replacing `_W` in place on each structure rewire.

## Multiple datasets

`simulate_multiple` runs the same algorithm against several datasets:

```python
mdata = sfa.DataSet().create('BORISOV_2009')

df_res = sim.simulate_multiple(
    num_samp=1000,
    alg=alg,
    mdata=mdata,
    use_norm=True,
)
```

!!! warning "v0.1.0 serial path is broken"
    `simulate_multiple(..., max_workers=1)` (the default) unpacks
    arguments positionally into `_simulate_single(num_samp, alg, ...)`
    while the method signature expects a single tuple. This raises
    `TypeError`. Workarounds in v0.1.0: call `simulate_single` per
    dataset and concatenate manually, or pass `max_workers > 1` to
    take the parallel branch (which uses `pool.map` correctly).

## Parallel processing

```python
df_res = sim.simulate_multiple(
    num_samp=1000,
    alg=alg,
    mdata=mdata,
    max_workers=4,
)
```

Speedup scales with the number of datasets, not iterations per
dataset.

## Perturbation-only analysis

For a single before/after comparison, use
`sfa.analysis.analyze_perturb`:

```python
from sfa.analysis import analyze_perturb

act, F = analyze_perturb(alg, data, targets=['MEK'])
```

It returns the activity change, the signal-flow change, and, with
`get_trj=True`, the activity trajectory.
