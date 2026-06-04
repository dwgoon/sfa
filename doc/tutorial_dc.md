# Discovery of control targets

This tutorial reproduces the main finding of
[Lee & Cho, *Scientific Reports* 2019, 9:14289](https://www.nature.com/articles/s41598-019-50790-0)
using SFA: when both ERK and AKT must be suppressed in an EGF/insulin
co-stimulated network, **inhibition of GAB1 or IRS** is predicted to
work, while inhibition of GS or PDK1 fails — because GS and PDK1 have
*opposite-sign* influences on ERK and AKT and so cannot move both in the
same direction with a single perturbation.

The bundled `BORISOV_2009` dataset has the same EGFR + IR signaling
topology used in the paper (22 nodes, 46 interactions), so we can
reproduce the result in a few dozen lines of Python.

## Setup

```python
import numpy as np
import pandas as pd

import sfa
from sfa.control import compute_influence, prioritize

algs = sfa.AlgorithmSet()
ds = sfa.DataSet()

mdata = ds.create('BORISOV_2009')
data = sfa.get_avalue(mdata)   # Any condition; we only need the topology.

alg = algs.create('SP')
alg.params.apply_weight_norm = True
alg.data = data
alg.initialize()
```

After `initialize()`, `alg.W` is the normalized weight matrix that the
influence computation operates on.

## Compute the influence matrix

Following the paper, we use $\alpha = 0.9$ and $\beta = 0.1$ for the
influence computation:

```python
df_inf = compute_influence(
    alg.W,
    alpha=0.9,
    beta=0.1,
    rtype='df',
    outputs=['ERK', 'AKT'],
    n2i=data.n2i,
)
df_inf = df_inf.apply(pd.to_numeric, errors='coerce')
```

`compute_influence` marks self-loops with `np.inf` and returns an
`object`-dtype frame; `pd.to_numeric(errors='coerce')` converts it to a
clean numeric frame.

## Inspect the candidates

The paper's key observation is best seen by reading off the influence
sign on both outputs:

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

- **GAB1, IRS**: positive influence on both ERK and AKT.
  → Inhibiting them flips both outputs negative, suppressing both
    simultaneously.
- **GS**: positive on ERK, *negative* on AKT.
  → Inhibition would suppress ERK but *up*-regulate AKT — opposite
    direction.
- **PDK1**: negative on ERK, positive on AKT.
  → Inhibition would *up*-regulate ERK while suppressing AKT — again
    opposite directions.

This reproduces the paper's conclusion that *GS and PDK1 fail to
suppress the two outputs simultaneously*.

## Find dual-output targets programmatically

For a single output, `prioritize` groups candidates by SPLO and returns
the top sources whose influence has the requested sign. To find
candidates whose **inhibition** suppresses an output, ask for positive
influence (`dac=+1`) — then the same negative perturbation drives the
output negative.

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

dual_targets = sorted(set(targets_erk) & set(targets_akt))
print('Inhibition candidates for both ERK and AKT:', dual_targets)
```

`thr_rank` can be an integer (top-N per SPLO bucket) or a fraction in
$(0, 1)$ (top fraction). Taking the intersection of the per-output
shortlists yields the nodes that can suppress both outputs with a
single perturbation. GAB1 and IRS appear in the intersection; GS and
PDK1 are filtered out — exactly the paper's finding.

## Validate by perturbation

We confirm the prediction by simulating each candidate's inhibition and
measuring the change at ERK and AKT.

```python
N = data.A.shape[0]
b0 = np.zeros((N,), dtype=np.float64)
b0[data.n2i['EGF']] = 1
b0[data.n2i['I']]   = 1   # EGF + insulin co-stimulation as in the paper.
x_ctrl = alg.compute(b0)

for tgt in ['GAB1', 'IRS', 'GS', 'PDK1']:
    b = b0.copy()
    b[data.n2i[tgt]] = -1                  # Inhibit the candidate.
    x = alg.compute(b)
    d_erk = x[data.n2i['ERK']] - x_ctrl[data.n2i['ERK']]
    d_akt = x[data.n2i['AKT']] - x_ctrl[data.n2i['AKT']]
    print(f"{tgt:<5s}  ΔERK={d_erk:+.4f}  ΔAKT={d_akt:+.4f}")
```

The signs of `ΔERK` and `ΔAKT` match the prediction read off the
influence table: both negative for GAB1 and IRS, mixed for GS and
PDK1.

## Visualizing the SPLO–Influence layout

`sfa.plot.siplot` draws a panel per SPLO bucket with sources sorted by
influence on the output of interest. The `designated` argument
highlights the names returned by `prioritize`, making the selection
easy to confirm visually.

```python
import matplotlib.pyplot as plt
from sfa.plot import siplot

fig = siplot(df_splo['ERK'], df_inf, output='ERK', designated=dual_targets)
plt.show()
```

For the original analysis — including the mutation-context cases
(`SFK` for constitutively active RAS, `PIP3` for activated PI3K) — see
the paper.
