# Signal Flow Analysis (SFA)

SFA is a simulation framework, which provides useful data structures and
functions for efficiently analyzing signal flow in complex networks.

## Features

- Topology-only signal flow / influence / control-target estimation.
- CPU backend with a LAPACK closed-form fast path
  (`scipy.linalg.solve`) and an iterative fallback.
- Optional native CUDA backend (cuBLAS + hand-written fused kernels +
  CUDA Graph) with dtype support (`float32`, `float64`, `float16`) and
  a trajectory mode for `SignalPropagation`.
- Convenient data structures for analyzing multiple datasets with
  multiple algorithms.
- Support for visualizing simulation results and signal flow.
- Parallel simulations using multiprocessing.
- User-defined algorithms or datasets.

## Documentation

- [Install](install.md)
- [Tutorials](tutorials.md)
    - [Signal flow analysis](tutorial_sfa.md)
    - [Discovery of control targets](tutorial_dc.md)
- [Data](data.md)
- [Algorithm](algorithm.md)
- [Control](control.md)
- [CUDA backend](cuda.md)
- [Visualization](visualization.md)
- [Simulation](simulation.md)
- [Development](develop.md)
- [API Reference](api.md)

## References

- [Lee & Cho, 2018](https://www.nature.com/articles/s41598-018-23643-5)
- [Lee & Cho, 2019](https://www.nature.com/articles/s41598-019-50790-0)
