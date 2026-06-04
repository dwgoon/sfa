# Control

The control module provides utilities for *discovery of control targets*:
nodes whose perturbation is most likely to push a chosen output in a chosen
direction. The approach is described in
[Lee & Cho, *Scientific Reports* 2019, 9:14289](https://www.nature.com/articles/s41598-019-50790-0).

## Influence matrix

`sfa.compute_influence` estimates the partial-derivative-based influence
of every source node on every target, using only the network topology.

Given the propagation update

$$
x(t+1) = \alpha W x(t) + (1 - \alpha) b,
$$

the influence matrix $S$ satisfies

$$
S_{ij} = \frac{\partial x_i}{\partial x_j}
      = \big(I + \alpha W + \alpha^2 W^2 + \cdots\big)_{ij},
$$

which is approximated by truncating the series. SFA uses the iteration

$$
S(t+1) = \alpha W S(t) + I, \qquad S(0) = \beta I,
$$

and the iteration stops when $\lVert S(t+1) - S(t) \rVert \le \mathrm{tol}$.

```python
import sfa
import numpy as np

data = sfa.get_avalue(sfa.DataSet().create('BORISOV_2009'))
alg = sfa.AlgorithmSet().create('SP')
alg.params.apply_weight_norm = True
alg.data = data
alg.initialize()

df_inf = sfa.compute_influence(
    alg.W,
    alpha=0.9,
    beta=0.1,
    rtype='df',
    outputs=['ERK', 'AKT'],
    n2i=data.n2i,
)
```

| Parameter   | Default | Description                                                   |
|-------------|---------|---------------------------------------------------------------|
| `W`         | —       | Weight matrix (output of `alg.W`).                            |
| `alpha`     | `0.9`   | Signal-flow contribution.                                     |
| `beta`      | `0.1`   | Basal-activity contribution; scales the final `S`.            |
| `S`         | `None`  | Initial influence matrix; defaults to identity.               |
| `rtype`     | `'df'`  | `'df'` for `pandas.DataFrame`, `'array'` for `numpy.ndarray`. |
| `outputs`   | `None`  | Required when `rtype='df'`; output node names.                |
| `n2i`       | `None`  | Required when `rtype='df'`; the data's name-to-index map.     |
| `max_iter`  | `1000`  | Iteration cap.                                                |
| `tol`       | `1e-7`  | Tolerance for the stopping criterion.                         |
| `device`    | `'cpu'` | `'cpu'` or `'gpu:<id>'` (requires CuPy).                      |
| `sparse`    | `False` | Use SciPy sparse matrices for the CPU path.                   |

## Shortest path length to output (SPLO)

`sfa.splo` computes, for each `(source, output)` pair, the shortest path
length in the directed network. This is used to bucket candidate sources by
how "close" they are to the output.

```python
df_splo = sfa.splo(
    nxdg=data.dg,
    sources=list(data.n2i),
    outputs=['ERK', 'AKT'],
    rtype='df',
)
```

`sfa.max_spl(nxdg)` is also available to inspect the diameter-like
quantity (the maximum shortest path length in the network) when choosing
SPLO bounds.

## Prioritizing control candidates

Once you have both influence and SPLO, `sfa.prioritize` groups candidates
by SPLO and selects, within each group, the top-ranked sources that move
the output in the requested *direction of activity change* (`dac`):
`+1` to up-regulate, `-1` to down-regulate.

```python
targets = sfa.prioritize(
    df_splo=df_splo,
    df_inf=df_inf,
    output='ERK',
    dac=-1,         # We want to down-regulate ERK.
    thr_rank=3,     # Top-3 per SPLO group; or a fraction in (0, 1).
    min_group_size=0,
    thr_inf=1e-10,
)
```

`sfa.control.arrange_si` is the lower-level helper used by `prioritize`;
call it directly when you need the grouped SPLO–Influence DataFrames
rather than just the target list.

## Visualizing SPLO–Influence

`sfa.plot.siplot` draws a grid of horizontal bar charts, one panel per
SPLO bucket, with the candidate sources sorted by influence on the
output. This makes the relative ranking inside each SPLO group easy to
read.

```python
import matplotlib.pyplot as plt
from sfa.plot import siplot

fig = siplot(df_splo, df_inf, output='ERK', designated=targets)
plt.show()
```

The `designated` argument highlights the names returned by
`prioritize`, so you can confirm the selection visually before applying
the perturbations.
