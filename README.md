<img src="assets/logo02.png" alt="Drawing" width="200px"/>

Signal Flow Analysis (SFA)
==========================
[![Documentation Status](https://readthedocs.org/projects/sfa/badge/?version=latest)](https://sfa.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/dwgoon/sfa/actions/workflows/tests.yml/badge.svg)](https://github.com/dwgoon/sfa/actions/workflows/tests.yml)

Signal Flow Analysis (SFA) is a computational framework for analyzing
signal propagation in complex directed networks. Using only the
topology of a signed network (no kinetic constants or dynamic data
required), SFA estimates how perturbations to individual nodes
propagate to chosen output nodes, quantifies the influence of every
source on every target, and prioritizes intervention candidates that
steer those outputs in a desired direction.

## Features

- Topological estimation of steady-state signal flow in directed signed
  networks, requiring only the adjacency structure and no kinetic
  parameters.
- Recording of activity trajectories along the iterative solution path,
  enabling inspection of transient dynamics in addition to the
  steady-state estimate.
- Batched simulation across multiple datasets, multiple algorithms, and
  multiple perturbation conditions.
- Quantification of pairwise node-to-node influence (the influence
  matrix) and identification of control-target candidates that steer
  chosen output nodes in a prescribed direction.
- Stratification of control-target candidates by their shortest-path
  distance to the output, following the SPLO-based prioritization of
  [Lee and Cho (2019)](https://www.nature.com/articles/s41598-019-50790-0).
- An extensible model in which user-defined propagation algorithms and
  benchmark datasets integrate with the core without modification.
- Optional GPU acceleration for large-scale problems on NVIDIA hardware.

## Install

SFA supports Python 3.10 and newer on Linux, macOS, and Windows.
Distributions are published in two families:

| Package       | CUDA   | Min. NVIDIA driver  | Platforms              |
|---------------|--------|---------------------|------------------------|
| `sfa`         | none   | -                   | Linux, macOS, Windows  |
| `sfa-cu128`   | 12.8.x | 570 (Linux / Win)   | Linux, Windows         |
| `sfa-cu132`   | 13.2.x | 580                 | Linux, Windows         |
| `sfa-cu133`   | 13.3.x | 580                 | Linux, Windows         |

Each CUDA wheel ships ahead-of-time compiled SASS for NVIDIA SM 7.0
through SM 12.0 (Volta, Turing, Ampere, Ada, Hopper, Blackwell) plus a
PTX fallback for newer GPUs. The cuBLAS and cudart runtime libraries
arrive as pinned `nvidia-*` PyPI dependencies, so no separate CUDA
toolkit install is required.

### CPU only

```bash
pip install sfa
```

### NVIDIA CUDA (Linux or Windows)

Pick **one** variant from the wheel matrix above that matches your
NVIDIA driver. If unsure, run `nvidia-smi` and check the "CUDA
Version" column - that is the maximum CUDA version your driver
supports.

Example (install the newest variant):

```bash
pip install sfa-cu133
```

> [!IMPORTANT]
> Install **only one** CUDA variant per environment. `sfa` and every
> `sfa-cuXYZ` share the `sfa` Python namespace and will conflict if
> stacked.

### Build from source

For a new CUDA major version, a custom GPU architecture, or
development against the source tree. Two paths; pick whichever fits
your environment.

**Conda-based** (recommended; bundles the CUDA toolchain into a
self-contained env):

```bash
git clone https://github.com/dwgoon/sfa.git && cd sfa
conda env create -f environment-cuda.yml
conda activate sfa-cu132
pip install -e .
```

`environment-cuda.yml` pulls the CUDA 13.2 toolkit (`nvcc`, `cudart`,
`cuBLAS`, ...) from the `nvidia` channel into the env, so no
system-wide CUDA install is required.

**Conda-free** (uses a system CUDA install instead):

```bash
git clone https://github.com/dwgoon/sfa.git && cd sfa
python -m venv .venv && source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows
pip install -e .                                    # picks up nvcc from PATH
```

This path needs (a) the NVIDIA CUDA Toolkit installed system-wide
with `nvcc` on `PATH`, and (b) a host C++ compiler (MSVC on Windows
in a "x64 Native Tools" prompt, GCC on Linux). The conda-based path
also needs the host C++ compiler.

The CUDA extension is built automatically when `nvcc` is discoverable.
To force a pure-Python build even with `nvcc` installed, set
`SFA_BUILD_CUDA=0` before `pip install`.

See [INSTALL.md](INSTALL.md#development-install-build-from-source)
for prerequisites, environment variables, and platform-specific
notes on both paths.

See [doc/install.md](doc/install.md) or
[sfa.readthedocs.io](https://sfa.readthedocs.io) for the full install
guide, including BLAS backend selection and CI-built wheel matrix.

## Verify the install

Two checks are available, in increasing order of coverage:

- `python tests/verification.py` - a portable post-install
  verification script. Runs without `pytest`, exits 0 on success,
  exercises the CPU LAPACK path and the `SignalPropagation`
  trajectory, and opportunistically runs a CUDA influence check when
  a GPU is visible.
- `python -m pytest tests/` - the full test suite. CUDA tests
  auto-skip on machines without an NVIDIA GPU.

```bash
python -m pip install pytest
python tests/verification.py
python -m pytest tests/
```

## Quick start

The examples below use the bundled `BORISOV_2009` dataset, an EGF +
insulin signaling network with simulated activity data derived from
the ODE model of
[Borisov et al. (2009)](https://www.embopress.org/doi/full/10.1038/msb.2009.19).

### Loading the network and the algorithm

Two objects are prepared before any computation:

- A *data* object holds:
    - a signed adjacency matrix - the wiring diagram of the signaling
      network, encoding who activates or inhibits whom;
    - the experimental condition - which ligands are present and at
      what dose.
- An *algorithm* object wraps a particular propagation rule. Here we
  use `SP` (signal propagation) from
  [Lee and Cho (2018)](https://www.nature.com/articles/s41598-018-23643-5).
  Calling `alg.initialize()` builds the working weight matrix `alg.W`
  and the basal activity vector `alg.b` from the data.

```python
import numpy as np
import sfa

# Pick one experimental condition from BORISOV_2009: activity AUC over
# 120 min under EGF = 1 nM and insulin = 1 nM stimulation.
mdata = sfa.DataSet().create('BORISOV_2009')
data = mdata['120m_AUC_EGF=1+I=1']

alg = sfa.AlgorithmSet().create('SP')
alg.params.apply_weight_norm = True
alg.data = data
alg.initialize()  # builds the weight matrix alg.W and basal activity alg.b
```

### Simulating signal propagation

- **Biological question.** Given a network of activating / inhibiting
  interactions and a perturbation at one or more nodes (a ligand
  stimulation, a genetic knockdown, a small-molecule inhibitor), how
  does the activity of every other biomolecule in the network change
  as the signal propagates outward from the perturbed nodes?
- **What the method does.** `SignalPropagation.propagate_iterative`
  runs the discrete-time update

  $$
  x(t+1) = \alpha\, W\, x(t) + (1 - \alpha)\, b
  $$

  until activities settle into a steady state.
- **Reading the parameters.**
    - `b` - basal input vector. Encodes the perturbation (e.g. EGF
      and insulin stimulation in the example below).
    - `a` (i.e. `alpha`) - balance between network-driven signal and
      basal drive. Larger `a` lets the network have more say.
    - `trajectory` - the per-iteration activity snapshot returned
      when `get_trj=True`. Reading it like a simulated time-course
      shows each node's activity rising or falling before it
      stabilises.
- **`device`** chooses where the computation runs:
    - `'cpu'` - the host CPU.
    - `'cuda:<device ID>'` (e.g. `'cuda:0'`, `'cuda:1'`) - a specific NVIDIA
      GPU.

```python
from sfa.algorithms.sp import SignalPropagation

sp = SignalPropagation('SP')
xi = np.zeros_like(alg.b)   # start every node at rest

x, trajectory = sp.propagate_iterative(
    alg.W,
    xi,
    alg.b,
    a=0.9,
    lim_iter=2000,
    tol=1e-7,
    get_trj=True,
    device='cuda:0',
    dtype=np.float32,
)
```

### Computing the influence matrix

- **Biological question.** The question shifts from "what does
  *this* perturbation do?" to "*which* perturbation should I apply?".
  Running one simulation per candidate becomes wasteful when there
  are many candidates.
- **What the influence matrix is.** A single `N x N` matrix `S` that
  answers all candidates at once:
    - `S[i, j]` is the steady-state change in node `i` when a unit
      perturbation is applied to node `j`, with the rest of the
      network held at baseline.
    - **Column `j`** = downstream signature of perturbing node `j`.
      Useful for predicting the in-network effect of a knockdown or
      a drug.
    - **Row `i`** = rank of upstream nodes that move readout `i` the
      most. Useful for selecting drug targets that steer a
      disease-relevant output in a chosen direction.
- **What the method does.** `compute_influence` builds `S` in closed
  form as `S = beta * (I - alpha * W)^-1`. This is equivalent to
  summing the `propagate_iterative` responses to every single-node
  perturbation, obtained in a single matrix solve. The
  [Lee and Cho (2019)](https://www.nature.com/articles/s41598-019-50790-0)
  control framework uses exactly this matrix to rank intervention
  candidates.
- **`device`** - same options as `propagate_iterative`: `'cpu'` or
  `'cuda:<device ID>'`.
- **`rtype`** - controls how `S` is returned:
    - `'array'` - raw NumPy `ndarray`, indexed by integer position.
    - `'df'` - pandas `DataFrame`, indexed by node names. Useful when
      you only care about a handful of readouts and want to rank
      source nodes by their effect. Requires two extra kwargs:
      `outputs=` (list of readout node names) and `n2i=` (the
      name-to-index dict, available as `alg.data.n2i`).

```python
from sfa.control import compute_influence

# Influence matrix on the CPU (LAPACK closed-form).
S_cpu = compute_influence(
    alg.W,
    alpha=0.9,
    beta=0.1,
    rtype='array',
    device='cpu',
)

# Same computation on a CUDA GPU, in float32 with TF32 Tensor Cores.
S_gpu = compute_influence(
    alg.W,
    alpha=0.9,
    beta=0.1,
    rtype='array',
    device='cuda:0',
    dtype=np.float32,
)
```

## Benchmarks

### Hardware setup

| Processor | Model                       | Spec                                                              |
|-----------|-----------------------------|-------------------------------------------------------------------|
| CPU       | Intel Core i9-12900KS       | Alder Lake, 16 cores (8P + 8E) / 24 threads, P-core up to 5.5 GHz |
| RAM       | Samsung M323R4GA3BB0        | DDR5, 4 x 32 GB = 128 GB, 4000 MT/s                                |
| GPU       | NVIDIA GeForce RTX 4090     | Ada Lovelace (sm_89), 24 GB GDDR6X, 16,384 CUDA cores             |

### Experimental setup

- `N` is the **number of nodes** in the network. The weight matrix
  `W` is a dense `N x N` synthetic matrix with the diagonal zeroed
  out, so every off-diagonal entry is a signed edge and the network
  has `N * (N - 1)` directed edges. `N` is swept across the values
  shown in each table below.
- Each time cell reports mean ± stddev over 5 independent runs after
  one warm-up call, to surface variance in addition to central
  tendency.
- The two benchmarks below answer two different questions:
    - **Small networks, fp64 unified.** Apples-to-apples comparison
      against the `sfa` v0.1.0 CPU iterative solver. Every column is
      computed in fp64 so that the speed-up reflects the algorithm
      and the hardware, not a precision trade-off.
    - **Large networks, GPU only.** Beyond ~5k nodes the v0.1.0 CPU
      iterative baseline becomes impractical, so we compare only the
      v0.2 GPU paths against each other across the precisions that
      a 4090 actually supports well. A CPU LAPACK fp64 column is
      kept as the accuracy reference.
- Driver scripts: `benchmarks/bench_v010_vs_v020.py` for the small
  networks table, `benchmarks/bench_gpu_largeN.py` for the large
  networks table. Speed-ups in parentheses are versus the leftmost
  column of the same table.

### Small networks

| # Nodes | # Edges  | CPU iter (fp64) ms | CPU LAPACK (fp64) ms | CUDA (fp64) ms        |
|---------|----------|--------------------|----------------------|-----------------------|
|    32   | 992      | 0.1 ± 0.0          | 0.2 ± 0.0 (0.4x)     | 1.3 ± 0.2 (0.06x)     |
|    64   |  ~4.0 K  | 0.2 ± 0.0          | 0.2 ± 0.0 (0.8x)     | 1.4 ± 0.1 (0.13x)     |
|   128   | ~16.3 K  | 2.5 ± 0.0          | 0.4 ± 0.0 (**7.2x**) | 1.9 ± 0.1 (1.3x)      |
|   256   | ~65.3 K  | 6.9 ± 0.2          | 2.4 ± 0.1 (**2.8x**) | 3.1 ± 0.8 (2.2x)      |
|   512   |  ~262 K  | 38.8 ± 1.7         | 190 ± 46 (0.2x)      | 6.4 ± 0.2 (**6.0x**)  |
|  1024   | ~1.05 M  | 180 ± 8            | 486 ± 89 (0.4x)      | 47 ± 10 (**3.8x**)    |
|  2048   | ~4.19 M  | 2140 ± 320         | 3880 ± 2990 (0.6x)   | 245 ± 2 (**8.7x**)    |
|  4096   | ~16.8 M  | 12520 ± 2380       | 5690 ± 1390 (2.2x)   | 4320 ± 580 (**2.9x**) |

### Large networks

| # Nodes | # Edges | CPU LAPACK (fp64) s | CUDA TF32 (fp32) s   | CUDA FP32 (no TF32) s | CUDA FP16 s              |
|---------|---------|---------------------|----------------------|-----------------------|--------------------------|
|  5000   |  ~25 M  |  5.10 ± 2.24             | 0.366 ± 0.027 (14x)  | 0.356 ± 0.034 (14x)   | 0.349 ± 0.037 (**15x**)  |
| 10000   | ~100 M  | 17.60 ± 0.57             | 1.55 ± 0.05 (11x)    | 4.07 ± 0.06 (4.3x)    | 1.13 ± 0.16 (**16x**)    |
| 20000   | ~400 M  | 70.88 ± 0.79             | 9.13 ± 0.10 (7.8x)   | 16.30 ± 0.28 (4.3x)   | 4.28 ± 0.02 (**17x**)    |

- CPU paths show noticeably higher variance than GPU paths (CPU
  LAPACK fp64 stddev reaches ~25-77% of the mean at small `N`),
  reflecting interference from the host OS and the 8P + 8E
  heterogeneous scheduler. GPU paths sit at ~1-10% stddev.
- The CUDA fp64 column beats the CPU LAPACK fp64 column across the
  entire small-network sweep, but the margin is modest because
  consumer Ada GPUs (RTX 4090 included) throttle fp64 to roughly
  1/64 of fp32. Strict fp64 work that does not fit on a workstation
  GPU should still consider a server-class CUDA card with full-rate
  fp64.
- For the lower-precision GPU paths, FP16 wins from `N` >= 5k upward,
  with TF32 close behind once `N` becomes large enough to be
  matmul-bound. Max abs error versus the CPU fp64 reference stays
  around `10^-6` for TF32 and `10^-4` for FP16 across the sweep,
  which is well within the accuracy budget for most SFA analyses.
- `SignalPropagation.propagate_iterative` is GEMV-bound rather than
  matmul-bound, so it scales differently from `compute_influence`;
  the CUDA backend only starts to win around `N` >= 16k, reaching
  roughly 3-4x at `N` = 32k.
- The CPU LAPACK path is sensitive to the BLAS choice and the thread
  count. At `N` = 4096 in our environment, Intel MKL with 8 threads
  is about 1.4x faster than the default scipy-OpenBLAS configuration.
  See `benchmarks/bench_threads_and_backend.py` for the sweep.

## Documentation

<https://sfa.readthedocs.io>

## Citation

If you use SFA in academic work, please cite the original papers that
introduced the framework:

* Daewon Lee & Kwang-Hyun Cho </br>
  "Topological estimation of signal flow in complex signaling networks" </br>
  [*Scientific Reports* (2018) **8**:5262](https://www.nature.com/articles/s41598-018-23643-5)

* Daewon Lee & Kwang-Hyun Cho </br>
  "Signal flow control of complex signaling networks" </br>
  [*Scientific Reports* (2019) **9**:14289](https://www.nature.com/articles/s41598-019-50790-0)
