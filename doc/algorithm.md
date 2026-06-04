# Algorithm

SFA models a signal flow algorithm as a subclass of `sfa.base.Algorithm`.
The shipped network-propagation algorithms (e.g. *Signal Propagation*, `SP`)
inherit from a common base class, `sfa.algorithms.np.NetworkPropagation`,
which already implements the simulation loop, perturbation handling,
and basal-activity bookkeeping.

## The `Algorithm` base class

`sfa.base.Algorithm` defines the minimal interface for an algorithm.

| Attribute  | Description                                                   |
|------------|---------------------------------------------------------------|
| `abbr`     | Short symbol for the algorithm (e.g. `"SP"`).                 |
| `name`     | Full name of the algorithm.                                   |
| `data`     | The currently bound `sfa.base.Data` object.                   |
| `params`   | A `ParameterSet` of hyperparameters for the algorithm.        |
| `result`   | A `sfa.base.Result` populated by `compute_batch()`.           |

Two methods must be implemented by every concrete algorithm:

- `compute(b)` — return the steady-state activity vector $x$
  given a basal-activity vector $b$.
- `compute_batch()` — iterate over every perturbation condition in
  `self.data` and store the per-condition results in `self.result.df_sim`.

`initialize()` is called once after `data` and `params` are set.
It is split into `initialize_network()` (build the weight matrix `W`) and
`initialize_basal_activity()` (build the default `b`), so subclasses can
override one without re-implementing the other.

## The `NetworkPropagation` base

`sfa.algorithms.np.NetworkPropagation` implements the propagation framework

$$
x(t+1) = \alpha W x(t) + (1 - \alpha) b
$$

The class:

- Builds `W` from `data.A` in `initialize_network()`, optionally normalizing
  it via `sfa.normalize()` when `params.apply_weight_norm` is `True`.
- Attempts to prepare an exact, closed-form solution in
  `prepare_exact_solution()`; if that raises a `LinAlgError`, it falls back
  to the iterative solver from `prepare_iterative_solution()`.
- Drives the per-condition simulation in `compute_batch()`, applying inputs
  (`apply_inputs`) and perturbations (`apply_perturbations`) for each row in
  `data.df_conds`.

### `NetworkPropagationParameterSet`

| Parameter            | Default | Description                                                    |
|----------------------|---------|----------------------------------------------------------------|
| `alpha`              | `0.5`   | Weight of the signal flow term in $x(t+1)$, in $(0, 1)$.       |
| `lim_iter`           | `1000`  | Maximum iterations for the iterative solver.                   |
| `apply_weight_norm`  | `False` | Apply `sfa.normalize()` to the adjacency matrix.               |
| `use_rel_change`     | `False` | Subtract the control-state activity from the perturbed state.  |
| `exsol_forbidden`    | `False` | Force the iterative solver even if an exact solution exists.   |
| `no_inputs`          | `False` | Skip applying inputs in `apply_inputs()`.                      |

The parameter set is a `FrozenClass`: new attributes cannot be added after
construction, so typos like `alg.params.alphaa = 0.9` raise `TypeError`.

## Signal Propagation (`SP`)

`sfa.algorithms.sp.SignalPropagation` provides both the exact and iterative
solvers for the propagation update above. The exact solution is

$$
M = (1 - \alpha)(I - \alpha W)^{-1}, \qquad x_\infty = M b
$$

and the iterative form is the fixed-point iteration on
$x(t+1) = \alpha W x(t) + (1-\alpha) b$ with a Frobenius-norm tolerance.

```python
>>> import sfa
>>> alg = sfa.AlgorithmSet().create('SP')
>>> alg.params.alpha = 0.9
>>> alg.params.apply_weight_norm = True
```

## Defining a new algorithm

To add a new algorithm, create a new module under `sfa/algorithms/`
named after the abbreviation in lowercase
(e.g. `sfa/algorithms/myalg.py` for `MYALG`).
The module must expose a `create_algorithm(abbr)` factory.

```python
# sfa/algorithms/myalg.py
import numpy as np
from .np import NetworkPropagation, NetworkPropagationParameterSet


def create_algorithm(abbr):
    return MyAlgorithm(abbr)


class MyAlgorithmParameterSet(NetworkPropagationParameterSet):
    def initialize(self):
        super().initialize()
        # Add custom parameters here.


class MyAlgorithm(NetworkPropagation):
    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "My custom algorithm"
        self._params = MyAlgorithmParameterSet()

    def propagate_iterative(self, W, xi, b, a=0.5, lim_iter=1000,
                            tol=1e-5, get_trj=False):
        x_t1 = np.array(xi, dtype=np.float64)
        for _ in range(lim_iter):
            x_t2 = a * W.dot(x_t1) + (1 - a) * b
            if np.linalg.norm(x_t2 - x_t1) <= tol:
                break
            x_t1 = x_t2.copy()
        return x_t2, _
```

`AlgorithmSet` discovers algorithms by scanning the `sfa/algorithms/`
directory. Any file whose name does not start with `_` and is not in the
internal `excluded` list is treated as an algorithm module; the file
basename uppercased becomes the key (`myalg.py` → `MYALG`).

```python
>>> algs = sfa.AlgorithmSet()
>>> alg = algs.create('MYALG')
>>> algs['MYALG'] is alg
True
```

If your algorithm has a closed-form steady state, override
`prepare_exact_solution()` and `propagate_exact(b)` as `SignalPropagation`
does; otherwise leave them unimplemented and rely on the iterative path.
