# Discovery of control targets

This tutorial reproduces the main finding of
[Lee & Cho, 2019](https://www.nature.com/articles/s41598-019-50790-0):
when both ERK and AKT must be suppressed in EGF/insulin co-stimulated
signaling, **inhibition of GAB1 or IRS** is predicted to work, while
GS and PDK1 fail because their influences on the two outputs have
opposite signs.

We use the bundled `BORISOV_2009` dataset, which has the same EGFR + IR
topology as in the paper.

## Setup

```python
import numpy as np
import pandas as pd

import sfa
from sfa.control import compute_influence, prioritize

algs = sfa.AlgorithmSet()
ds = sfa.DataSet()

mdata = ds.create('BORISOV_2009')
data = sfa.get_avalue(mdata)

alg = algs.create('SP')
alg.params.apply_weight_norm = True
alg.data = data
alg.initialize()
```

## Compute the influence matrix

Following the paper, use $\alpha = 0.9$ and $\beta = 0.1$:

```python
df_inf = compute_influence(
    alg.W,
    alpha=0.9, beta=0.1,
    rtype='df',
    outputs=['ERK', 'AKT'],
    n2i=data.n2i,
)
df_inf = df_inf.apply(pd.to_numeric, errors='coerce')
```

In v0.1.0 the DataFrame ends up with an `object` dtype (the inner
loop assigns `np.inf` to the diagonal as a placeholder); the
`pd.to_numeric` cast normalizes it.

## Inspect the candidates

```python
nodes = ['GAB1', 'IRS', 'GS', 'PDK1']
print(df_inf.loc[nodes])
```

```text
        ERK       AKT
GAB1   +0.00496  +0.00904
IRS    +0.00756  +0.01163
GS     +0.02465  -0.00072
PDK1   -0.02579  +0.08779
```

- **GAB1, IRS**: positive influence on both ERK and AKT, so inhibiting
  them flips both outputs negative.
- **GS**: positive on ERK, negative on AKT.
- **PDK1**: negative on ERK, positive on AKT.

GS and PDK1 cannot suppress both outputs with a single perturbation,
which reproduces the paper's conclusion.

## Find dual-output targets programmatically

```python
df_splo = sfa.splo(
    nxdg=data.dg,
    sources=list(data.n2i),
    outputs=['ERK', 'AKT'],
    rtype='df',
)

targets_erk = prioritize(
    df_splo['ERK'], df_inf, output='ERK', dac=+1,
    thr_rank=0.5, min_group_size=0, thr_inf=1e-10,
)
targets_akt = prioritize(
    df_splo['AKT'], df_inf, output='AKT', dac=+1,
    thr_rank=0.5, min_group_size=0, thr_inf=1e-10,
)

dual = sorted(set(targets_erk) & set(targets_akt))
print(dual)
```

Pass `dac=+1` to keep sources with positive influence on the output;
inhibiting them then drives the output negative. The intersection
contains GAB1 and IRS; GS and PDK1 are filtered out by the sign
mismatch.

## Validate by perturbation

```python
N = data.A.shape[0]
b0 = np.zeros((N,), dtype=np.float)
b0[data.n2i['EGF']] = 1
b0[data.n2i['I']]   = 1                # EGF + insulin co-stimulation.
x_ctrl = alg.compute(b0)

for tgt in ['GAB1', 'IRS', 'GS', 'PDK1']:
    b = b0.copy()
    b[data.n2i[tgt]] = -1
    x = alg.compute(b)
    d_erk = x[data.n2i['ERK']] - x_ctrl[data.n2i['ERK']]
    d_akt = x[data.n2i['AKT']] - x_ctrl[data.n2i['AKT']]
    print("%-5s  dERK=%+0.4f  dAKT=%+0.4f" % (tgt, d_erk, d_akt))
```

Both deltas are negative for GAB1 and IRS; GS and PDK1 produce
mixed-sign changes.

## Visualize

```python
import matplotlib.pyplot as plt
from sfa.plot import siplot

fig = siplot(df_splo['ERK'], df_inf, output='ERK', designated=dual)
plt.show()
```

For the mutation-context analyses (SFK for activated RAS, PIP3 for
activated PI3K), see [Lee & Cho, 2019](https://www.nature.com/articles/s41598-019-50790-0).
