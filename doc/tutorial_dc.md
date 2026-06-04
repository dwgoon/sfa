# Discovery of control targets

This tutorial walks through finding *control targets* — nodes whose
perturbation is most likely to push a chosen output in a chosen
direction — using only the network topology. The approach is described
in [Lee & Cho, *Scientific Reports* 2019, 9:14289](https://www.nature.com/articles/s41598-019-50790-0).
For the conceptual background and the individual function signatures,
see the [Control](control.md) page.

## Setup

We use one of the bundled datasets and the `SP` algorithm.

```python
import sfa
import numpy as np

algs = sfa.AlgorithmSet()
ds = sfa.DataSet()

mdata = ds.create('BORISOV_2009')
data = sfa.get_avalue(mdata)

alg = algs.create('SP')
alg.params.alpha = 0.9
alg.params.apply_weight_norm = True
alg.data = data
alg.initialize()
```

After `initialize()`, `alg.W` is the normalized weight matrix that
`compute_influence` operates on.

## Step 1 — Compute the influence matrix

```python
outputs = ['ERK', 'AKT']
df_inf = sfa.compute_influence(
    alg.W,
    alpha=0.9,
    beta=0.1,
    rtype='df',
    outputs=outputs,
    n2i=data.n2i,
)
df_inf.head()
```

Each row is a candidate source node; each column is an output. A
positive entry means activating the source increases the output;
negative means it decreases the output.

## Step 2 — Compute SPLO

```python
df_splo = sfa.splo(
    nxdg=data.dg,
    sources=list(data.n2i),
    outputs=outputs,
    rtype='df',
)
```

`df_splo` reports the shortest path length from every source to every
output. Sources unreachable from a given output are dropped.

`sfa.max_spl(data.dg)` returns the longest such path in the network,
which is a useful upper bound when choosing `max_splo` in the next
step.

## Step 3 — Prioritize candidates

We want to *down-regulate* ERK, so `dac = -1`. We keep the top-3
candidates per SPLO bucket.

```python
targets = sfa.prioritize(
    df_splo=df_splo,
    df_inf=df_inf,
    output='ERK',
    dac=-1,
    thr_rank=3,
    min_group_size=0,
    thr_inf=1e-10,
)

print(targets)
```

`thr_rank` can also be a fraction in $(0, 1)$ — for example,
`thr_rank=0.1` keeps the top 10% of each SPLO bucket.

## Step 4 — Visualize the SPLO–Influence layout

```python
import matplotlib.pyplot as plt
from sfa.plot import siplot

fig = siplot(
    df_splo['ERK'],
    df_inf,
    output='ERK',
    designated=targets,   # Highlight the targets selected above.
)
plt.show()
```

`siplot` lays out a panel per SPLO bucket. Bars are sorted by
influence; the names in `designated` are colored red so you can confirm
the selection visually.

## Step 5 — Validate the candidates by perturbation

Sanity-check the selection by perturbing each target in turn and
inspecting the simulated activity change at the output. The example
below shows the perturbation of one candidate; loop over `targets` to
evaluate the full selection.

```python
N = data.A.shape[0]
b = np.zeros((N,), dtype=np.float64)
b[data.n2i['EGF']] = 1
x_ctrl = alg.compute(b)

target = targets[0]
b[data.n2i[target]] = -1   # Inhibit the candidate.
x_ptb = alg.compute(b)
b[data.n2i[target]] = 0    # Restore for the next loop iteration.

dx_erk = x_ptb[data.n2i['ERK']] - x_ctrl[data.n2i['ERK']]
print(f"{target}: Δ(ERK) = {dx_erk:+.4f}")
```

A negative `Δ(ERK)` confirms the candidate down-regulates ERK as
intended. Combined with the influence ranking and the SPLO bucketing,
this gives a short shortlist of control targets to test
experimentally.
