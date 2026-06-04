# Data

## Defining a new data with network structure

To create a new data with a network structure,
it is required to define a derived class of the `sfa.base.Data` class.
In `__init__` of the class, we need to create four objects.

| # | Member | Description                                          |
|---|--------|------------------------------------------------------|
| 1 | `_A`   | 2-dimensional `numpy.ndarray` for adjacency matrix.  |
| 2 | `_dg`  | `NetworkX.DiGraph` object.                           |
| 3 | `_n2i` | `dict` for mapping name to index.                    |
| 4 | `_i2n` | `dict` for mapping index to name.                    |

The underscore `_` of a member name means
the member is a protected member,
which is defined not to be directly accessed
(refer to [this Stack Overflow answer](https://stackoverflow.com/a/797814)).
Instead, the four members can be accessed through
[`property`](https://docs.python.org/3/library/functions.html?highlight=property#property).
The underscored members (protected members) should be used
only in the methods of the class such as `__init__`.

```python
>>> obj._A  # We can access like this, but defined not to.
>>> obj.A   # Instead, access the member like this.
```

Now, let's define a child class of `sfa.base.Data`
for a simple 3-node cascade as a toy example.

```python
import numpy as np
import networkx as nx

import sfa

class ThreeNodeCascade(sfa.base.Data):
    def __init__(self):
        super().__init__()
        self._abbr = "TNC"  # Abbreviation for this data.
        self._name = "A simple three node cascade"  # Full name

        # Create name to index mapper and index to name mapper.
        self._n2i = {"A": 0, "B": 1, "C": 2}  # Name to index
        self._i2n = {idx: name for name, idx in self._n2i.items()}

        # Create Directed graph object of NetworkX.
        # SIGN matches the attribute key used by sfa.read_sif.
        self._dg = nx.DiGraph()
        self._dg.add_edge('A', 'B', SIGN=+1)
        self._dg.add_edge('B', 'C', SIGN=+1)

        # Create adjacency matrix with signs.
        n = self._dg.number_of_nodes()
        self._A = np.zeros((n, n), dtype=np.float64)

        for (src, tgt, attr) in self._dg.edges(data=True):
            isrc = self._n2i[src]
            itgt = self._n2i[tgt]
            self._A[itgt, isrc] = attr['SIGN']


if __name__ == "__main__":
    data = ThreeNodeCascade()

    algs = sfa.AlgorithmSet()
    alg = algs.create('SP')
    alg.data = data
    alg.params.apply_weight_norm = True
    alg.initialize()

    n = data.A.shape[0]
    b = np.zeros((n,))  # Basal activity

    # Set node A to be activated
    # by assigning 1 to its basal activity.
    b[data.n2i['A']] = 1

    # Compute the activity at steady-state.
    x = alg.compute(b)

    print("[Activity at steady-state]")
    print(x)
```

The above code is very straightforward
if you are familiar with the functionalities of
the `numpy` and `networkx` packages.
The following is the result of executing the code.

```text
[Activity at steady-state]
[0.5   0.25  0.125]
```

If you want to see both the name and its activity,
use `n2i` or `i2n`.

```python
>>> for i, act in enumerate(x):
...     print(data.i2n[i], act)
A 0.5
B 0.25
C 0.125
>>> idx = data.n2i['B']
>>> x[idx]
0.25
```

Now, it is a little bit easier to read.

For a large-scale network,
it is almost impossible to write node names
and their relationships one by one in code.
Thus, `sfa` provides utility functions
to create the data class.

If network structure information is defined
in a text file such as a SIF file,
we can utilize the `sfa.read_sif` function.
`sfa.read_sif` reads the text file and returns
`A`, `n2i`, and `dg` objects
that are required to define the data class.

Let's go back to the toy example.

```text
A   +   B
B   +   C
```

The network structure can be described in SIF format as above.

```python
import os
import numpy as np
import sfa

class ThreeNodeCascade(sfa.base.Data):
    def __init__(self):
        super().__init__()
        self._abbr = "TNC"
        self._name = "A simple three node cascade"

        # Specify the file path for the network file in SIF.
        dpath = os.path.dirname(__file__)
        fpath = os.path.join(dpath, 'network.sif')

        # Use the read_sif function.
        A, n2i, dg = sfa.read_sif(fpath, as_nx=True)
        self._A = A
        self._n2i = n2i
        self._dg = dg
        self._i2n = {idx: name for name, idx in n2i.items()}


if __name__ == "__main__":
    data = ThreeNodeCascade()

    algs = sfa.AlgorithmSet()
    alg = algs.create('SP')
    alg.data = data
    alg.params.apply_weight_norm = True
    alg.initialize()

    n = data.A.shape[0]
    b = np.zeros((n,))

    # Activate node B at this time.
    b[data.n2i['B']] = 1
    x = alg.compute(b)

    print("[Activity at steady-state]")
    for i, act in enumerate(x):
        print("[Node %s] %f" % (data.i2n[i], act))
```

In the above code, all you need to do is
just put the file path of the network in `sfa.read_sif`.
We recommend utilizing the above code snippet as a template
for creating your own network structure data.
The result of the above code is as follows.

```text
[Activity at steady-state]
[Node A] 0.000000
[Node B] 0.500000
[Node C] 0.250000
```

In this example, positive and negative signs of links are
defined as `+` and `-`, respectively, in the file.
However, if the sign or interaction information is defined differently,
you can specify it with the `signs` keyword argument
of `sfa.read_sif`.
For example, if a network file has `activates` and `inhibits`
as the signs for positive and negative links, respectively,
we can call `sfa.read_sif` as follows.

```python
>>> signs = {'activates': 1, 'inhibits': -1}
>>> sfa.read_sif("network.sif", signs=signs, as_nx=True)
(array([[ 0,  0,  0],
        [ 1,  0,  0],
        [ 0, -1,  0]]),
 {'A': 0, 'B': 1, 'C': 2},
 <networkx.classes.digraph.DiGraph at 0x2ce1503c7b8>)
```

If your network structure is defined
in a different file format (not SIF),
you will need to write some code lines
to parse the network structure data.

## Defining a new data for validating an algorithm

- Describe how to define your own datasets only with experimental data.
- Explanation for the members of the `Data` class for validation.
