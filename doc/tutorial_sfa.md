# Signal flow analysis

This brief tutorial walks through the v0.1.0 API. It assumes overall
familiarity with [Lee & Cho, 2018](https://www.nature.com/articles/s41598-018-23643-5).

## Creating algorithm object

`sfa.AlgorithmSet` creates and manages algorithm objects.

```python
>>> import sfa
>>> algs = sfa.AlgorithmSet()
```

Create the *Signal Propagation* (`SP`) algorithm.

```python
>>> alg = algs.create('SP')
SP algorithm has been created.
>>> alg.abbr
'SP'
>>> alg
SignalPropagation object
```

`sfa.AlgorithmSet` behaves like a dictionary:

```python
>>> algs['SP']
SignalPropagation object
```

## Setting hyperparameter values

Algorithm hyperparameters live on `alg.params`, a `FrozenClass`
instance that only allows the predefined attributes.

```python
>>> alg.params.alpha
0.5
```

`alpha` controls the proportion of signal flow in determining the
next system state, $x(t+1)$:

$$
x(t+1) = \alpha W x(t) + (1 - \alpha) b
$$

Change `alpha` by assigning a value in $(0, 1)$:

```python
>>> alg.params.alpha = 0.9
>>> alg.params.alpha
0.9
```

`apply_weight_norm` is `False` by default but is usually set to
`True`:

```python
>>> alg.params.apply_weight_norm = True
```

## Creating data object

Bundled datasets are loaded through `sfa.DataSet`. For example,
the [Borisov et al.](http://msb.embopress.org/content/5/1/256)
dataset is keyed `BORISOV_2009`:

```python
>>> ds = sfa.DataSet()
>>> mdata = ds.create('BORISOV_2009')
BORISOV_2009 data has been created.
>>> mdata['120m_AUC_EGF=0.001+I=0.1']
BorisovData object
```

The container is a `dict` keyed by condition; pick one or grab the
first with the helper:

```python
>>> data = sfa.get_avalue(mdata)
>>> data.abbr
'120m_AUC_EGF=0.001+I=0.1'
```

## Accessing the members of a data object

Each `sfa.base.Data` instance exposes:

- `A` — adjacency matrix (`numpy.ndarray`)
- `dg` — directed graph (`networkx.DiGraph`)
- `n2i` — name-to-index `dict`
- `i2n` — index-to-name `dict`

```python
>>> data.A[data.n2i['ERK'], data.n2i['MEK']]    # MEK -> ERK
1
>>> data.A[data.n2i['GAB1'], data.n2i['ERK']]   # ERK -| GAB1
-1
>>> data.A[data.n2i['ERK'], data.n2i['EGFR']]   # no link
0
```

In v0.1.0, signs on the graph object are stored under the
`SIGN` edge attribute (set by `sfa.read_sif`):

```python
>>> for src, trg, attr in data.dg.edges(data=True):
...     if attr['SIGN'] > 0:
...         print('%s -> %s' % (src, trg))
...     elif attr['SIGN'] < 0:
...         print('%s -| %s' % (src, trg))
...
EGF -> EGFR
ERK -| GAB1
...
```

## Analyzing data with an algorithm

Bind the data, initialize, and use the algorithm:

```python
>>> alg.params.alpha = 0.5
>>> alg.params.apply_weight_norm = True
>>> alg.data = data
>>> alg.initialize()
```

`initialize` builds `alg.W` from `data.A`, normalizing if requested:

```python
>>> data.A[data.n2i['GAB1'], data.n2i['EGFR']]
1
>>> alg.W[data.n2i['GAB1'], data.n2i['EGFR']]
0.1889822365046136
```

Set up a basal-activity vector and apply EGF stimulation:

```python
>>> import numpy as np
>>> N = data.dg.number_of_nodes()
>>> b = np.zeros((N,), dtype=np.float)   # v0.1.0: np.float is OK on NumPy < 1.20
>>> b[data.n2i['EGF']] = 1
```

Compute the steady-state activity:

```python
>>> xs1 = alg.compute(b)
>>> xs1[data.n2i['ERK']]
0.0016554557287082902
>>> xs1[data.n2i['AKT']]
0.0015562514037656679
```

Apply an inhibitory perturbation to MEK:

```python
>>> b[data.n2i['MEK']] = -1
>>> xs2 = alg.compute(b)
>>> xs2[data.n2i['ERK']]
-0.24735422595037565
>>> xs2[data.n2i['AKT']]
0.001836795161913794
```

Compare the two conditions:

```python
>>> dxs = xs2 - xs1
>>> ind_up = np.where(dxs > 0)[0]
>>> ind_dn = np.where(dxs < 0)[0]
>>> [data.i2n[i] for i in ind_dn]
['ERK', 'MEK']
```

## Applying perturbation to a link

In v0.1.0, `propagate_exact` reuses a cached matrix `M` and does
**not** detect in-place edits to `alg.W`. Assign through the
property setter so the cache is rebuilt:

```python
>>> b = np.zeros((N,), dtype=np.float)
>>> b[data.n2i['EGF']] = 1
>>> W_ptb = alg.W.copy()
>>> W_ptb[:, data.n2i['PI3K']] = 0      # remove all PI3K out-links
>>> alg.W = W_ptb                       # triggers M re-computation
>>> xs3 = alg.compute(b)
>>> xs3[data.n2i['AKT']]
0.0
```

## Estimating signal flow

Signal flow is the element-wise product of link weight and source
activity:

$$
F(t)_{ij} = W_{ij} \cdot x(t)_{j}
$$

```python
>>> alg.initialize()         # restore the intact W
>>> W1 = alg.W.copy()
>>> F1 = W1 * xs1
>>> F1[data.n2i['PIP3'], data.n2i['PI3K']]
0.02780066830505488
```

For two conditions, the net signal flow is

$$
F_{net} = F_{c2} - F_{c1}
$$
