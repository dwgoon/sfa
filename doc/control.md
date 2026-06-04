# Control

The control module provides utilities for *discovery of control targets*:
nodes whose perturbation is most likely to push a chosen output in a chosen
direction. The approach is described in
[Lee & Cho, 2019](https://www.nature.com/articles/s41598-019-50790-0).

## Influence matrix

`sfa.control.compute_influence` estimates the partial-derivative-based
influence of every source node on every target, using only the network
topology.

The signal flow propagation is

$$
x(t+1) = \alpha W x(t) + \beta b,
$$

with two hyperparameters $\alpha$ and $\beta$. At steady state the
activity vector is $x^* = \beta (I - \alpha W)^{-1} b$, so the influence
of a basal-activity change in node $j$ on node $i$ is

$$
S_{ij} = \frac{dx_i}{db_j} = \frac{\partial x_j}{\partial b_j}\,\frac{dx_i}{dx_j}
       = \beta\,\bigl(I + \alpha W + \alpha^2 W^2 + \cdots\bigr)_{ij}.
$$

SFA approximates the series with the iteration

$$
S(t+1) = \alpha W S(t) + \beta I, \qquad S(0) = \beta I,
$$

and terminates when the Frobenius norm of the unscaled update falls
below the tolerance: $\lVert S(t+1) - S(t) \rVert_F \le \mathrm{tol}$.
The shipped CPU implementation checks the tolerance on the unscaled
series and multiplies by $\beta$ at the end (see the implementation
note at the bottom of this page).

```python
import sfa
import numpy as np
import pandas as pd
from sfa.control import compute_influence

data = sfa.get_avalue(sfa.DataSet().create('BORISOV_2009'))
alg = sfa.AlgorithmSet().create('SP')
alg.params.apply_weight_norm = True
alg.data = data
alg.initialize()

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

Because of how the DataFrame is populated (each cell is assigned in a
loop with a transient `np.inf` placeholder on the diagonal), the
returned `DataFrame` has an `object` dtype; cast it with
`pd.to_numeric(errors='coerce')` before sorting or arithmetic.

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
| `get_iter`  | `False`    | Also return the actual iteration count.                           |
| `device`    | `'cpu'`    | `'cpu'` or `'gpu:<id>'` (requires CuPy).                          |
| `sparse`    | `False`    | Use SciPy sparse matrices for the CPU path.                       |

## Shortest path length to output (SPLO)

`sfa.splo` computes, for each `(source, output)` pair, the shortest path
length in the directed network. This is used to bucket candidate sources
by how "close" they are to the output.

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

Once you have both influence and SPLO, `sfa.control.prioritize` groups
candidates by SPLO and selects, within each group, the top-ranked
sources whose influence on the output has the requested sign (`dac`):

- `dac=+1`: sources with **positive** influence on the output. Their
  *inhibition* (negative perturbation) drives the output negative;
  their activation drives it positive.
- `dac=-1`: sources with **negative** influence. Their *activation*
  drives the output negative.

```python
from sfa.control import prioritize

targets = prioritize(
    df_splo=df_splo['ERK'],   # SPLO series for the output of interest.
    df_inf=df_inf,
    output='ERK',
    dac=+1,                   # Inhibit these to suppress ERK.
    thr_rank=3,               # Top-3 per SPLO group; or a fraction in (0, 1).
    min_group_size=0,
    min_splo=None,            # Optional lower bound on SPLO bucket value.
    max_splo=None,            # Optional upper bound on SPLO bucket value.
    thr_inf=1e-10,
)
```

`min_splo` and `max_splo` restrict the candidate pool to a SPLO window
(useful for screening only "distant" or only "proximal" sources).

`sfa.control.arrange_si` is the lower-level helper used by `prioritize`;
call it directly when you need the grouped SPLO-Influence DataFrames
rather than just the target list. See
[Discovery of control targets](tutorial_dc.md) for a worked example
that reproduces the dual-output (ERK and AKT) finding from
[Lee & Cho, 2019](https://www.nature.com/articles/s41598-019-50790-0).

## Visualizing SPLO-Influence

`sfa.plot.siplot` draws a grid of horizontal bar charts, one panel per
SPLO bucket, with the candidate sources sorted by influence on the
output. This makes the relative ranking inside each SPLO group easy to
read.

```python
import matplotlib.pyplot as plt
from sfa.plot import siplot

fig = siplot(df_splo['ERK'], df_inf, output='ERK', designated=targets)
plt.show()
```

The `designated` argument highlights the names returned by
`prioritize`, so you can confirm the selection visually before applying
the perturbations.

!!! note "Implementation note"
    The shipped CPU implementation iterates as
    $S(t+1) = S(t)\,\alpha W + I$ with $S(0) = I$ and applies the $\beta$
    factor only at the final step. Starting from the identity, this is
    mathematically equivalent to the paper's iteration above; both
    converge to $\beta(I - \alpha W)^{-1}$ when the spectral radius of
    $\alpha W$ is less than one.
