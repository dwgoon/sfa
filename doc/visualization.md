# Visualization

`sfa.vis` provides small helpers for turning simulation outputs into
visual representations of signal flow.

## Computing graphics from a simulation

`sfa.vis.compute_graphics` annotates a NetworkX `DiGraph` with
per-node fill colors and per-edge widths/signs/colors derived from an
activity change and a signal-flow matrix.

The example below assumes a configured `alg` and `data` as in
[Signal flow analysis](tutorial_sfa.md):

```python
import sfa
from sfa.vis import compute_graphics
from sfa.analysis import analyze_perturb

act, F = analyze_perturb(alg, data, targets=['MEK'])

dg = compute_graphics(
    F=F,
    act=act,
    A=data.A,
    n2i=data.n2i,
    lw_min=1.0,
    lw_max=10.0,
    pct_link=90,
    pct_act=50,
)
```

`dg` is a `networkx.DiGraph`. Each node has `FILL_COLOR`,
`BORDER_WIDTH`, and `BORDER_COLOR` attributes; each edge has `SIGN`,
`FILL_COLOR`, and `WIDTH`. The fill color blends from white toward red
for up-regulated nodes and toward blue for down-regulated nodes,
scaled by the percentile of activity change. Edge widths are mapped
from $\log_{10}|F|$ clipped at the `pct_link` percentile.

!!! warning "v0.1.0 quirks"
    - `compute_graphics` calls `F.to_numpy().nonzero()` and
      `A.to_numpy().nonzero()`, which fail because both `F` and `A`
      are `numpy.ndarray` (no `.to_numpy()` method). Use the v0.2.x
      patch (`F.nonzero()`) or call the helpers with DataFrames.
    - The link sizing crashes on an all-zero `F` because `log_flows`
      is empty; guard against this in your wrapper.

## Integration with SFV

For interactive rendering, hand the annotated graph to
[SFV (Seamless Flow Visualization)](https://github.com/dwgoon/sfv).
`sfa.vis.sfv.sfv.visualize_signal_flow` writes the styles directly
into an `sfv.graphics.Network` using Qt-based color and header
classes.

```python
from sfa.vis.sfv.sfv import visualize_signal_flow

visualize_signal_flow(
    net,            # pre-constructed sfv.graphics.Network
    F=F, act=act, A=data.A, n2i=data.n2i,
    color_up=None, color_dn=None,
    show_label=True, show_act=True,
    pct_act=50, pct_link=90,
)
```

This subpackage requires `qtpy` and `sfv` (both optional in v0.1.0).

## Building a `Data` object from an SFV network

```python
from sfa.vis.sfv.sfv import create_from_graphics

data = create_from_graphics(net, abbr="MyNet", inputs={'EGF': 1.0})
```

The returned object only carries topology; experimental DataFrames
are left as `None`.

## Plot helpers

`sfa.plot` collects a few matplotlib-based plotters:

| Symbol                          | Description                                            |
|---------------------------------|--------------------------------------------------------|
| `sfa.plot.Heatmap`              | Heatmap of an $n \times m$ activity/accuracy matrix.   |
| `sfa.plot.BatchResultTable`     | Tabular view of `Algorithm.result.df_sim`.             |
| `sfa.plot.ConditionTable`       | Tabular view of `Data.df_conds`.                       |
| `sfa.plot.HierarchicalClusteringTable` | Hierarchically clustered table view.            |
| `sfa.plot.siplot`               | SPLO-Influence bar grid (see [Control](control.md)).   |
