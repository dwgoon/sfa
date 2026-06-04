# Algorithm

A v0.1.0 algorithm is a subclass of `sfa.base.Algorithm`. The shipped
network-propagation algorithms inherit from
`sfa.algorithms.np.NetworkPropagation`.

## The `Algorithm` base class

| Attribute  | Description                                                   |
|------------|---------------------------------------------------------------|
| `abbr`     | Short symbol for the algorithm (e.g. `"SP"`).                 |
| `name`     | Full name of the algorithm.                                   |
| `data`     | Currently bound `sfa.base.Data` object.                       |
| `params`   | `ParameterSet` of hyperparameters for the algorithm.          |
| `result`   | `sfa.base.Result` populated by `compute_batch()`.             |

Concrete algorithms must implement:

- `compute(b)`: return the steady-state activity vector $x$ given a
  basal-activity vector $b$.
- `compute_batch()`: iterate over every perturbation condition in
  `self.data` and store per-condition results in `self.result.df_sim`.

`initialize()` is split into `initialize_network()` and
`initialize_basal_activity()` so subclasses can override one without
re-implementing the other.

## The `NetworkPropagation` base

`sfa.algorithms.np.NetworkPropagation` is the abstract framework;
concrete subclasses (`SignalPropagation`) implement the specific
propagation update. The canonical SP form is

$$
x(t+1) = \alpha W x(t) + (1 - \alpha) b,
$$

which is the $\beta = 1 - \alpha$ special case of the general
[Lee & Cho, 2019](https://www.nature.com/articles/s41598-019-50790-0)
form $x(t+1) = \alpha W x(t) + \beta b$.

The base class:

- Builds `W` from `data.A` in `initialize_network()`, optionally
  normalizing via `sfa.normalize()` when
  `params.apply_weight_norm` is `True`.
- Tries `prepare_exact_solution()` first; if it raises
  `numpy.linalg.LinAlgError`, falls back to the iterative solver.
- Drives the per-condition simulation in `compute_batch()` using
  `apply_inputs` and `apply_perturbations`.

### `NetworkPropagationParameterSet`

| Parameter            | Default | Description                                                    |
|----------------------|---------|----------------------------------------------------------------|
| `alpha`              | `0.5`   | Signal-flow weight in $x(t+1)$, in $(0, 1)$.                   |
| `lim_iter`           | `1000`  | Maximum iterations for the iterative solver.                   |
| `apply_weight_norm`  | `False` | Apply `sfa.normalize()` to the adjacency matrix.               |
| `use_rel_change`     | `False` | Subtract the control-state activity from the perturbed state. |
| `exsol_forbidden`    | `False` | Force the iterative solver even when an exact solution exists. |
| `no_inputs`          | `False` | Skip applying inputs in `apply_inputs()`.                      |

The parameter set is a `FrozenClass`: new attributes cannot be added,
so typos like `alg.params.alphaa = 0.9` raise `TypeError`.

## Signal Propagation (`SP`)

`sfa.algorithms.sp.SignalPropagation` provides both solvers. The exact
solution is

$$
M = (1 - \alpha)(I - \alpha W)^{-1}, \qquad x_\infty = M b,
$$

and the iterative form is the fixed-point iteration on
$x(t+1) = \alpha W x(t) + (1-\alpha) b$ with the Euclidean ($L_2$)
norm tolerance on $x(t+1) - x(t)$.

```python
>>> import sfa
>>> alg = sfa.AlgorithmSet().create('SP')
>>> alg.params.alpha = 0.9
>>> alg.params.apply_weight_norm = True
```

!!! warning "v0.1.0 caveat"
    `SignalPropagation.prepare_exact_solution()` caches the exact
    matrix $M$ but **does not** detect in-place edits to `alg.W`.
    When mutating the weight matrix between calls, assign through
    the setter (`alg.W = W_new`) to invalidate the cache. The v0.2.x
    series fixes this so the cache resets after each
    `prepare_exact_solution()` call.

## Defining a new algorithm

Create a new module under `sfa/algorithms/` named after the
abbreviation in lowercase (e.g. `sfa/algorithms/myalg.py` for
`MYALG`). The module must expose a `create_algorithm(abbr)` factory:

```python
# sfa/algorithms/myalg.py
import numpy as np
from .np import NetworkPropagation, NetworkPropagationParameterSet


def create_algorithm(abbr):
    return MyAlgorithm(abbr)


class MyAlgorithmParameterSet(NetworkPropagationParameterSet):
    def initialize(self):
        super().initialize()


class MyAlgorithm(NetworkPropagation):
    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "My custom algorithm"
        self._params = MyAlgorithmParameterSet()
        # Without an exact-solution override, force the iterative path
        # so that initialize_network() does not call the empty base hook.
        self._params.exsol_forbidden = True

    def propagate_iterative(self, W, xi, b, a=0.5, lim_iter=1000,
                            tol=1e-5, get_trj=False):
        x_t1 = np.array(xi, dtype=np.float)
        num_iter = 0
        for num_iter in range(1, lim_iter + 1):
            x_t2 = a * W.dot(x_t1) + (1 - a) * b
            if np.linalg.norm(x_t2 - x_t1) <= tol:
                break
            x_t1 = x_t2.copy()
        return x_t2, num_iter
```

`AlgorithmSet` discovers the module by scanning `sfa/algorithms/`;
the filename uppercased becomes the key (`myalg.py` → `MYALG`).
