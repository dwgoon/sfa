# Visualization

`sfa.vis` provides a small set of helpers for turning simulation outputs
into visual representations of signal flow.

## Computing graphics from a simulation

`sfa.vis.compute_graphics` annotates a NetworkX `DiGraph` with per-node
fill colors and per-edge widths/signs/colors derived from an activity
change and a signal-flow matrix. The result is a graph you can render
with any NetworkX-aware drawing layer.

```python
import sfa
from sfa.vis import compute_graphics
from sfa.analysis import analyze_perturb

# Run two conditions to get an activity-change vector `act` and
# signal-flow matrix `F`.
act, F = analyze_perturb(alg, data, targets=['MEK'])

dg = compute_graphics(
    F=F,
    act=act,
    A=data.A,
    n2i=data.n2i,
    lw_min=1.0,
    lw_max=10.0,
    pct_link=90,   # Percentile used to scale link widths.
    pct_act=50,    # Percentile used to scale node fills.
)
```

`dg` is a `NetworkX.DiGraph`. Each node has `FILL_COLOR`, `BORDER_WIDTH`,
and `BORDER_COLOR` attributes; each edge has `SIGN`, `FILL_COLOR`, and
`WIDTH` attributes. The fill color blends from white toward red for
up-regulated nodes and toward blue for down-regulated nodes, scaled by
the percentile of activity change you pass in. Edge widths are mapped
from $\log_{10}|F|$ clipped at the `pct_link` percentile.

The pre-existing nodes of a graph object can be reused by passing a
`dg=` argument; otherwise a fresh `DiGraph` is created and populated
from `n2i`.

## Integration with SFV

For interactive rendering, SFA can hand its annotated graph off to
[SFV (Seamless Flow Visualization)](https://github.com/dwgoon/sfv).
`sfa.vis.sfv.sfv.visualize_signal_flow` does the same coloring/widening
work as `compute_graphics`, but writes the styles directly into an
`sfv.graphics.Network` object using Qt-based color objects and header
shapes.

```python
from sfa.vis.sfv.sfv import visualize_signal_flow

visualize_signal_flow(
    net,            # sfv.graphics.Network
    F=F, act=act, A=data.A, n2i=data.n2i,
    color_up=None,  # Defaults to red.
    color_dn=None,  # Defaults to blue.
    show_label=True,
    show_act=True,
    pct_act=50,
    pct_link=90,
)
```

This subpackage requires `qtpy` and `sfv` to be available; both are
optional dependencies of `sfa` itself.

## Building a `Data` object from an SFV network

If you start from an SFV network instead of a SIF file, the helper
`sfa.vis.sfv.sfv.create_from_graphics` constructs a one-off `sfa.Data`
subclass whose topology is taken from the graphics object.

```python
from sfa.vis.sfv.sfv import create_from_graphics

data = create_from_graphics(net, abbr="MyNet", inputs={'EGF': 1.0})
```

The returned object only carries network topology; experimental
dataframes (`df_conds`, `df_exp`, `df_ptb`) are left as `None`, so it is
ready to use with `sfa.AlgorithmSet`-created algorithms but not with
random batch simulators that compare against experimental data.

## Plot helpers

`sfa.plot` collects a few `matplotlib`-based plots that are useful for
inspecting simulation results:

| Symbol                      | Description                                            |
|-----------------------------|--------------------------------------------------------|
| `sfa.plot.Heatmap`          | Heatmap of an `n × m` activity or accuracy matrix.     |
| `sfa.plot.BatchResultTable` | Tabular view of `Algorithm.result.df_sim`.             |
| `sfa.plot.ConditionTable`   | Tabular view of `Data.df_conds`.                       |
| `sfa.plot.HierarchicalClusteringTable` | Hierarchically clustered table view.        |
| `sfa.plot.siplot`           | SPLO–Influence bar grid (see [Control](control.md)).   |
