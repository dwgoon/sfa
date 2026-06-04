# Development

Reference for contributors who want to extend v0.1.0 with new
algorithms, datasets, or analyses.

## Package layout

| Path                        | Purpose                                                   |
|-----------------------------|-----------------------------------------------------------|
| `sfa/base.py`               | `Algorithm`, `Data`, `Result`, `ContainerItem` base types.|
| `sfa/containers.py`         | `AlgorithmSet` and `DataSet` singletons.                  |
| `sfa/fileio.py`             | `read_sif`, `read_inputs`, `create_from_sif`.             |
| `sfa/utils.py`              | Matrix utilities, randomization, singleton helper.        |
| `sfa/topology.py`           | `max_spl`, `splo` (shortest path lengths).                |
| `sfa/stats.py`              | Accuracy helpers (`calc_accuracy`).                       |
| `sfa/algorithms/`           | One module per algorithm; `np.py` and `sp.py` shipped.    |
| `sfa/data/<dataset>/`       | One subpackage per dataset, with `network.sif`, etc.      |
| `sfa/analysis/`             | Perturbation analysis and random batch simulators.        |
| `sfa/control/`              | Influence matrix and target prioritization.               |
| `sfa/plot/`                 | Matplotlib-based plotters.                                |
| `sfa/vis/`                  | Graph annotation and the optional SFV integration.        |

## Adding a new algorithm

1. Create `sfa/algorithms/<name>.py`. The filename, uppercased,
   becomes the key used by `AlgorithmSet`.
2. Define a `create_algorithm(abbr)` factory that returns an instance
   of a subclass of `sfa.base.Algorithm` (typically
   `NetworkPropagation`).
3. Implement `compute(b)` and `compute_batch()`. For
   network-propagation variants, override `propagate_iterative`. If
   you do not also provide
   `prepare_exact_solution`/`propagate_exact`, set
   `self._params.exsol_forbidden = True` so the base class skips its
   closed-form branch during `initialize_network()`.
4. Reuse `NetworkPropagationParameterSet` and add custom
   hyperparameters as properties on a `FrozenClass` subclass.

`AlgorithmSet().create('YOURALG')` discovers the module automatically.

## Adding a new dataset

1. Create `sfa/data/<name>/__init__.py` and add at least
   `network.sif` and any `conds.tsv`, `exp.tsv`, `ptb.tsv` files you
   want `DataSet` to load. The directory name, uppercased, becomes the
   key.
2. The `__init__.py` must expose a `create_data()` factory. It may
   return:
    - a single `sfa.base.Data` subclass instance,
    - a `list` of instances (will be keyed by each
      `data.abbr.upper()`),
    - or a `dict` of pre-built instances.
3. Use `sfa.read_sif(fpath, as_nx=True)` to populate `_A`, `_n2i`,
   and `_dg`; use `pd.read_table(..., header=0, index_col=0)` for the
   TSV files.

See `sfa/data/borisov_2009/__init__.py` for a working example.

## Coding conventions (v0.1.0)

- Target both Python 2 and Python 3; use `from builtins import super`
  guarded by `sys.version_info <= (2, 8)` and `six.add_metaclass`
  where the metaclass syntax differs.
- NumPy aliases such as `np.float`, `np.int`, `np.mat` are used
  throughout. They are deprecated in NumPy 1.20 and removed in
  later versions; pin `numpy<1.20` to run v0.1.0 as shipped, or use
  v0.2.x for current NumPy.
- `pd.read_table(...)` is the file-loading idiom. Pandas deprecated
  it later in favor of `pd.read_csv(..., sep='\t')`.

## Known issues in v0.1.0

The following bugs were identified and fixed in v0.2.x. They are
present in v0.1.0 as shipped:

- `rand_swap(..., inplace=True)` returns `None`.
- `rand_flip(..., pivots=...)` indexes edges by node index.
- `np.eye(N, dtype=np.float)`, `dtype=np.int`, `np.mat(...).T`,
  `pd.read_table` deprecations on modern stacks.
- `nx.Digraph` typo in `sfa.utils.to_networkx_digraph`
  (correct is `nx.DiGraph`).
- `.to_numpy()` is called on `ndarray` in `sfa.utils.rand_*`,
  `sfa.vis.utils._compute_graphics_links`, and the SFV update path.
- `RandomWeightBatchSimulator` operates on an all-zero `_W` and never
  samples any weights.
- `RandomStructureBatchSimulator` does not clear stale entries
  between iterations.
- `simulate_multiple(..., max_workers=1)` raises `TypeError` because
  the serial branch unpacks args positionally.
- `SignalPropagation.prepare_exact_solution` never clears
  `_weight_matrix_invalidated`, so the inverse is recomputed on every
  `propagate_exact` call.
- `compute_influence(..., get_iter=True)` returns a 0-based iteration
  count.

## Building the documentation

The v0.1.0 documentation is written in Markdown and built with
[MkDocs](https://www.mkdocs.org) +
[Material](https://squidfunk.github.io/mkdocs-material/) +
[mkdocstrings](https://mkdocstrings.github.io/). The original Sphinx
sources live in `_doc/`.

```bash
$ pip install -r docs-requirements.txt
$ mkdocs serve         # local preview on http://127.0.0.1:8000/
$ mkdocs build         # static site in ./site/
```

!!! note "API docstring rendering"
    Several v0.1.0 docstrings use RST inline math (`:math:` roles)
    and `\begin{align}` blocks. These render as raw text in MkDocs
    because the toolchain does not pre-translate RST. The v0.2.x
    docstrings convert these to `$...$` and `$$...$$` so MathJax
    handles them.
