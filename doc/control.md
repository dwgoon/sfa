# Control

`sfa.control` provides utilities for *discovery of control targets*:
nodes whose perturbation is most likely to push a chosen output in a
desired direction. The approach is from
[Lee & Cho, 2019](https://www.nature.com/articles/s41598-019-50790-0).

## Influence matrix

`sfa.control.compute_influence` estimates the partial-derivative-based
influence of every source on every target using only the network
topology.

The signal-flow propagation is

$$
x(t+1) = \alpha W x(t) + \beta b,
$$

with hyperparameters $\alpha$ and $\beta$. At steady state, $x^* =
\beta (I - \alpha W)^{-1} b$, and the influence of a basal-activity
change in node $j$ on node $i$ is

$$
S_{ij} = \frac{dx_i}{db_j}
       = \beta \bigl(I + \alpha W + \alpha^2 W^2 + \cdots\bigr)_{ij}.
$$

v0.1.0 approximates the series with the iteration

$$
S(t+1) = \alpha W S(t) + \beta I, \qquad S(0) = \beta I,
$$

and stops when $\lVert S(t+1) - S(t) \rVert_F$ falls below `tol`.

```python
import sfa
import pandas as pd
from sfa.control import compute_influence

data = sfa.get_avalue(sfa.DataSet().create('BORISOV_2009'))
alg = sfa.AlgorithmSet().create('SP')
alg.params.apply_weight_norm = True
alg.data = data
alg.initialize()

df_inf = compute_influence(
    alg.W,
    alpha=0.9, beta=0.1,
    rtype='df',
    outputs=['ERK', 'AKT'],
    n2i=data.n2i,
)
df_inf = df_inf.apply(pd.to_numeric, errors='coerce')
```

| Parameter   | Default    | Description                                                       |
|-------------|------------|-------------------------------------------------------------------|
| `W`         | (required) | Weight matrix (output of `alg.W`).                                |
| `alpha`     | `0.9`      | Signal-flow contribution.                                         |
| `beta`      | `0.1`      | Basal-activity contribution; scales the final $S$.                |
| `S`         | `None`     | Initial influence matrix; defaults to identity.                   |
| `rtype`     | `'df'`     | `'df'` for `pandas.DataFrame`, `'array'` for `numpy.ndarray`.     |
| `outputs`   | `None`     | Required when `rtype='df'`; output node names.                    |
| `n2i`       | `None`     | Required when `rtype='df'`; the data's name-to-index map.         |
| `max_iter`  | `1000`     | Iteration cap.                                                    |
| `tol`       | `1e-7`     | Tolerance for the stopping criterion.                             |
| `get_iter`  | `False`    | Also return the iteration count (note: 0-based in v0.1.0).        |
| `device`    | `'cpu'`    | `'cpu'` or `'gpu:<id>'` (requires CuPy).                          |
| `sparse`    | `False`    | Use SciPy sparse matrices for the CPU path.                       |

!!! warning "v0.1.0 quirks"
    - `get_iter` returns a 0-based count of completed iterations in
      v0.1.0; fixed to a 1-based count in v0.2.x.
    - The DataFrame return is built cell by cell with `np.inf`
      placeholders, leading to an `object` dtype; cast with
      `pd.to_numeric(errors='coerce')` before sorting/arithmetic.
    - The sparse path inherits the same iteration but feeds back a
      SciPy sparse matrix when `rtype='array'`.

## Shortest path length to output (SPLO)

`sfa.splo` computes, for each `(source, output)` pair, the shortest
path length in the directed network. SPLO is used to bucket candidate
sources by topological distance to the output.

```python
df_splo = sfa.splo(
    nxdg=data.dg,
    sources=list(data.n2i),
    outputs=['ERK', 'AKT'],
    rtype='df',
)
```

`sfa.max_spl(nxdg)` returns the longest such path in the network.

## Prioritizing control candidates

`sfa.control.prioritize` groups candidates by SPLO and selects the
top-ranked sources whose influence on the output has the requested
sign (`dac`):

- `dac=+1`: positive-influence sources; inhibiting them drives the
  output negative.
- `dac=-1`: negative-influence sources; activating them drives the
  output negative.

```python
from sfa.control import prioritize

targets = prioritize(
    df_splo=df_splo['ERK'],
    df_inf=df_inf,
    output='ERK',
    dac=+1,
    thr_rank=3,
    min_group_size=0,
    min_splo=None,
    max_splo=None,
    thr_inf=1e-10,
)
```

See [Discovery of control targets](tutorial_dc.md) for the worked
example reproducing the ERK + AKT dual-output finding.

## Visualizing SPLO-Influence

`sfa.plot.siplot` draws a panel-grid of horizontal bar charts, one per
SPLO bucket, with sources sorted by influence on the output.

```python
import matplotlib.pyplot as plt
from sfa.plot import siplot

fig = siplot(df_splo['ERK'], df_inf, output='ERK', designated=targets)
plt.show()
```
