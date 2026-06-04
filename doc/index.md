# Signal Flow Analysis (SFA) — v0.1.0

!!! warning "Version note"
    This is the documentation for the **v0.1.0** SFA codebase
    (`origin/0.1.0`, tagged as `v0.1.0`).
    It describes the pre-modernization implementation that supports
    Python 2 and Python 3, uses NumPy aliases such as `np.float` and
    `np.int`, and ships with the original Sphinx documentation in
    `_doc/`. For the Python 3.7+ port with current NumPy/pandas
    compatibility, see the v0.2.x documentation on the `main` branch.

SFA is a simulation framework that provides data structures and
functions for analyzing signal flow in complex networks based purely
on network topology.

## Features

- Convenient data structures for handling multiple datasets with
  multiple algorithms.
- Visualization of simulation results and signal flow.
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
- [Visualization](visualization.md)
- [Simulation](simulation.md)
- [Development](develop.md)
- [API Reference](api.md)

## References

- [Lee & Cho, 2018](https://www.nature.com/articles/s41598-018-23643-5)
- [Lee & Cho, 2019](https://www.nature.com/articles/s41598-019-50790-0)
