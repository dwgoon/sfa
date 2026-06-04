# Data

A v0.1.0 dataset is a subclass of `sfa.base.Data`. The `__init__`
should populate four members:

| # | Member | Description                                          |
|---|--------|------------------------------------------------------|
| 1 | `_A`   | 2-D `numpy.ndarray` adjacency matrix.                |
| 2 | `_dg`  | `networkx.DiGraph` instance (signs on edges).        |
| 3 | `_n2i` | `dict` mapping node name to row/column index.        |
| 4 | `_i2n` | `dict` mapping index to node name.                   |

The underscored members are protected; access them through the
`property` of the same name without the underscore.

## Toy example: 3-node cascade

```python
import numpy as np
import networkx as nx

import sfa


class ThreeNodeCascade(sfa.base.Data):
    def __init__(self):
        super().__init__()
        self._abbr = "TNC"
        self._name = "A simple three node cascade"

        self._n2i = {"A": 0, "B": 1, "C": 2}
        self._i2n = {idx: name for name, idx in self._n2i.items()}

        # SIGN is the edge attribute key used by sfa.read_sif.
        self._dg = nx.DiGraph()
        self._dg.add_edge('A', 'B', SIGN=+1)
        self._dg.add_edge('B', 'C', SIGN=+1)

        n = self._dg.number_of_nodes()
        self._A = np.zeros((n, n), dtype=np.float)
        for src, tgt, attr in self._dg.edges(data=True):
            isrc, itgt = self._n2i[src], self._n2i[tgt]
            self._A[itgt, isrc] = attr['SIGN']
```

Run signal propagation with EGF replaced by node `A`:

```python
if __name__ == "__main__":
    data = ThreeNodeCascade()
    algs = sfa.AlgorithmSet()
    alg = algs.create('SP')
    alg.data = data
    alg.params.apply_weight_norm = True
    alg.initialize()

    b = np.zeros(data.A.shape[0])
    b[data.n2i['A']] = 1
    x = alg.compute(b)
    print(x)              # -> [0.5, 0.25, 0.125]
```

## Loading from a SIF file

`sfa.read_sif` reads a tab- or space-separated SIF file and returns
`(A, n2i, dg)`. For the toy network above the file is:

```text
A   +   B
B   +   C
```

```python
import os
import sfa


class ThreeNodeCascade(sfa.base.Data):
    def __init__(self):
        super().__init__()
        self._abbr = "TNC"
        self._name = "A simple three node cascade"

        dpath = os.path.dirname(__file__)
        A, n2i, dg = sfa.read_sif(os.path.join(dpath, 'network.sif'),
                                  as_nx=True)
        self._A = A
        self._n2i = n2i
        self._dg = dg
        self._i2n = {idx: name for name, idx in n2i.items()}
```

Non-default sign labels can be supplied through the `signs` keyword:

```python
>>> sfa.read_sif("network.sif",
...              signs={'activates': 1, 'inhibits': -1},
...              as_nx=True)
```

## Defining a dataset for algorithm validation

To validate an algorithm against perturbation/condition tables, also
populate `_df_conds`, `_df_exp`, and `_df_ptb` (with `pd.read_table`
in v0.1.0):

```python
self._df_conds = pd.read_table(os.path.join(dpath, "conds.tsv"),
                               header=0, index_col=0)
self._df_exp = pd.read_table(os.path.join(dpath, "exp.tsv"),
                             header=0, index_col=0)
self._df_ptb = pd.read_table(os.path.join(dpath, "ptb.tsv"),
                             index_col=0)
```

See `sfa/data/borisov_2009/__init__.py` and
`sfa/data/korkut_2015a/__init__.py` for working examples.
