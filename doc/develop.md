# Development

This page is a quick reference for contributors who want to extend SFA
with new algorithms, datasets, or analyses.

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

1. Create `sfa/algorithms/<name>.py`. The filename, uppercased, becomes
   the key used by `AlgorithmSet`.
2. Define a `create_algorithm(abbr)` factory that returns an instance of
   a subclass of `sfa.base.Algorithm` (typically `NetworkPropagation`).
3. Implement `compute(b)` and `compute_batch()`. For network-propagation
   variants, override `propagate_iterative` and optionally
   `propagate_exact`.
4. Reuse `NetworkPropagationParameterSet` and add any custom
   hyperparameters as properties on a `FrozenClass` subclass.

`AlgorithmSet().create('YOURALG')` will discover the module
automatically. See the [Algorithm](algorithm.md) page for a full
template.

## Adding a new dataset

1. Create `sfa/data/<name>/__init__.py` and add at least
   `network.sif` and any `conds.tsv`, `exp.tsv`, `ptb.tsv` files you
   want `DataSet` to load. The directory name, uppercased, becomes the
   key (`borisov_2009` → `BORISOV_2009`).
2. The `__init__.py` must expose a `create_data()` factory. It may
   return:
    - a single `sfa.base.Data` subclass instance,
    - a `list` of instances (will be keyed by each `data.abbr.upper()`),
    - or a `dict` of pre-built instances.
3. Use `sfa.read_sif(fpath, as_nx=True)` to populate `_A`, `_n2i`, and
   `_dg`; use `pd.read_csv(..., sep='\t')` for the TSV files.

See `sfa/data/borisov_2009/__init__.py` and
`sfa/data/korkut_2015a/__init__.py` for working examples.

## Coding conventions

- Target Python 3.7 and newer; the codebase no longer carries Python 2
  compatibility shims.
- Prefer explicit dtypes (`np.float64`, `int`) over the legacy aliases
  `np.float` / `np.int`, which were removed in NumPy 1.20.
- Use `pd.read_csv(..., sep='\t')` instead of the deprecated
  `pd.read_table`.
- For matrix-shape operations on adjacency matrices, work with NumPy
  `ndarray`s directly: `.to_numpy()` is a `pandas` method and does not
  exist on `ndarray`.

## Building the documentation

Documentation is written in Markdown and built by
[MkDocs](https://www.mkdocs.org) with the
[Material](https://squidfunk.github.io/mkdocs-material/) theme and
[mkdocstrings](https://mkdocstrings.github.io/) for the API reference.

```bash
$ pip install -r docs-requirements.txt
$ mkdocs serve         # Local preview on http://127.0.0.1:8000/
$ mkdocs build         # Static site in ./site/
```

Read the Docs picks up `.readthedocs.yaml` automatically; pushing to the
default branch triggers a fresh build at <https://sfa.readthedocs.io>.
