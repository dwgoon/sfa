# Install

SFA ships pre-built wheels across one CPU and three CUDA optimized
versions. All
of them share the same `sfa` Python namespace, so install **at most
one** into a given environment.

| Distribution  | CUDA   | Min. NVIDIA driver  | OS                          |
|---------------|--------|---------------------|-----------------------------|
| `sfa`         | none   | -                   | Linux, macOS, Windows       |
| `sfa-cu128`   | 12.8.x | 570 (Linux / Win)   | Linux, Windows              |
| `sfa-cu132`   | 13.2.x | 580                 | Linux, Windows              |
| `sfa-cu133`   | 13.3.x | 580                 | Linux, Windows (newest)     |

## Requirements

- Python 3.10 or newer.
- For `sfa`: `numpy`, `scipy`, `pandas`, `networkx`, `threadpoolctl`
  (pulled in automatically).
- For each `sfa-cu1XX`: a sufficiently new NVIDIA driver (see
  table) plus the matching `nvidia-cublas-cu1X` and
  `nvidia-cuda-runtime-cu1X` runtime packages. These are declared as
  pip dependencies of the wheel and resolve automatically; you do not
  need a system-wide CUDA toolkit install.
- AOT-compiled SASS covers Volta (SM 7.0) through Blackwell
  (SM 12.0); newer GPUs use the embedded PTX via the driver JIT.

## Picking the right `sfa-cuXYZ`

Run `nvidia-smi` and look at the "CUDA Version" column. That is the
**maximum** CUDA the installed driver supports. Pick the highest
`sfa-cu1XX` whose CUDA major.minor is less than or equal to that
number:

```text
nvidia-smi -> "CUDA Version: 13.3"  -> any of sfa-cu128 / cu132 / cu133
nvidia-smi -> "CUDA Version: 13.0"  -> sfa-cu128
nvidia-smi -> "CUDA Version: 12.8"  -> sfa-cu128
nvidia-smi -> "CUDA Version: 12.6"  -> upgrade your driver or use `sfa` (CPU)
```

When in doubt, start with `sfa-cu128` for the widest driver coverage
and move to a newer one only when you need the absolute latest
cuBLAS.

## CPU install

```bash
pip install sfa
```

This works on every supported OS, including macOS (both Intel x86_64
and Apple Silicon arm64).

## CUDA install

```bash
pip install sfa-cu130
```

Linux and Windows x86_64 only. macOS is excluded because Apple ended
NVIDIA driver development in 2019.

If the wheel does not match your platform (uncommon architectures,
Alpine `musl`, Windows ARM), pip falls back to the source distribution
and tries to build locally - the source build requires `nvcc` if
`SFA_BUILD_CUDA` is not set to `0`.

## Optional extras

The Matplotlib-based helpers in `sfa.plot` require `matplotlib` and
`seaborn`:

```bash
pip install "sfa[plot]"
```

## Build from source

Clone the repository and use the development install:

```bash
git clone https://github.com/dwgoon/sfa.git
cd sfa
```

### CPU-only build

The CUDA extension is skipped automatically when `nvcc` is unavailable.
To force the CPU-only build even when `nvcc` is on `PATH`, set
`SFA_BUILD_CUDA=0`:

```bash
SFA_BUILD_CUDA=0 pip install -e .
```

### CUDA build (NVIDIA dev environments)

The fastest path is the shipped conda environment, which installs the
CUDA 13.2 toolchain (nvcc, cudart, cuBLAS dev, CCCL) plus `pybind11`
and the runtime Python deps:

```bash
conda env create -f environment-cuda.yml
conda activate sfa-cu132
pip install -e .
```

On Windows you also need a Visual Studio C++ host compiler (Build Tools
for Visual Studio 2022 is enough). On Linux, `gcc` 9+ is fine.

`setup.py` recognizes three environment variables:

| Variable             | Purpose                                                                |
|----------------------|------------------------------------------------------------------------|
| `SFA_BUILD_CUDA`     | `0` to force a pure-Python install. Default: build if `nvcc` is found. |
| `SFA_CUDA_ARCH`      | Semicolon-separated list of CUDA compute capabilities, e.g. `sm_89` for a single-GPU dev build, or `sm_70;sm_75;sm_80;sm_86;sm_89;sm_90` for the wheel-wide AOT matrix. Default: the full matrix. |
| `SFA_PACKAGE_NAME`   | Override the PyPI name. CI uses `SFA_PACKAGE_NAME=sfa-cu1XX` to publish a CUDA-optimized wheel from the same source tree as `sfa`. |

The build emits AOT SASS for every requested arch and a PTX fallback for
the highest one, so the resulting binary works on any current NVIDIA GPU
without a runtime JIT round-trip on first use.

## CPU BLAS backend

The CPU closed-form path in `compute_influence` calls LAPACK `?gesv`
through scipy. The thread count and (optionally) the BLAS library
itself are controllable per-call:

```python
sfa.control.compute_influence(
    W, alpha=0.9, beta=0.1,
    device="cpu",
    num_threads=8,        # cap BLAS threads for this call
    backend="mkl",        # ctypes-load MKL even if scipy is OpenBLAS
)
```

`num_threads` works for whichever BLAS is currently loaded
(`threadpoolctl` handles the dispatch). `backend` is more aggressive:
it bypasses scipy and calls LAPACKE on a library you have installed
separately. This is useful when:

- scipy was installed against OpenBLAS (the conda-forge / pip default)
  but you want MKL's faster threaded `getrf`/`getrs`.
- You need to A/B benchmark BLAS implementations without rebuilding
  scipy.

### Installing MKL alongside scipy-OpenBLAS

```bash
pip install mkl          # PyPI redist of libmkl_rt
# or
conda install -c conda-forge mkl
```

After install, `python -c "from sfa._blas_ctypes import available_backends; print(available_backends())"` should list `['mkl']`. The
ctypes loader probes:

1. The PyPI `mkl` package directory (if importable).
2. The active env's `<prefix>/Library/bin` (Windows) or `<prefix>/lib`
   (Linux/macOS).
3. `ctypes.util.find_library('mkl_rt')` as a fallback.

### Installing a stock OpenBLAS for the ctypes path

scipy 1.17+ vendors an *ILP64* OpenBLAS under `libscipy_openblas64_`
that does not expose the standard `LAPACKE_dgesv` symbol the ctypes
dispatcher uses. To get the runtime swap working with OpenBLAS, install
a normal LP64 build:

```bash
conda install -c conda-forge libblas=*=*openblas openblas
```

### Discovering what's available at runtime

```python
import sfa
sfa.blas.info()              # threadpoolctl record for the loaded BLAS
sfa.blas.loaded_backends()   # ['openblas']  (or ['mkl'], etc.)
sfa.blas.num_threads()       # current limit, if any

from sfa._blas_ctypes import available_backends
available_backends()         # ['mkl']  (lists ctypes-loadable extras)
```

### Notes

- `num_threads` and `backend` only apply to the **CPU** paths. The
  CUDA backend ignores them - GPU thread management is done by cuBLAS
  and our CUDA Graph.
- On the iterative CPU path of `compute_influence` (when an initial
  `S` is supplied or `get_iter=True`), `backend` is a no-op because the
  loop runs through `numpy.dot`; only `num_threads` takes effect there.

## Verify

The repository ships a single-file verification script that does not
require `pytest`:

```bash
python tests/verification.py
```

Successful output ends with `ALL OK`. It exercises:

1. `import sfa` and `import sfa._cuda`.
2. The CPU LAPACK closed-form path (`S = beta * (I - alpha*W)^-1`).
3. `SignalPropagation.propagate_iterative` with a trajectory.
4. (Opportunistically) a CUDA influence computation when a GPU is
   visible.

For the full pytest suite:

```bash
pip install pytest
pytest tests/
```

CUDA-marked tests skip cleanly on machines without an NVIDIA GPU
(`tests/_skip_helpers.py` checks both extension availability and
`device_count() > 0`).

## Updating

If you cloned and used `pip install -e .`, pull and the package updates
in place:

```bash
git pull origin main
```

For wheel installs:

```bash
pip install --upgrade sfa            # CPU
pip install --upgrade sfa-cu130      # CUDA
```

## Uninstall

```bash
pip uninstall sfa                    # or `sfa-cu130`
```

## Troubleshooting

- **`ImportError: cannot import name '_native'`** on a CUDA install:
  the wheel did not match your platform, or the runtime cuBLAS is
  missing. Reinstall with `--force-reinstall` and verify the wheel tag
  matches your Python/OS/arch.
- **`RuntimeError: cuda runtime error 100 - no CUDA-capable device`**:
  the NVIDIA driver is not visible to the process. Confirm with
  `nvidia-smi`; in containers, mount the GPU and use the NVIDIA
  Container Toolkit.
- **Slow first call on the CUDA path**: this should not happen with
  the shipped fat binary; the AOT SASS covers SM 7.0 through SM 9.0.
  If you see significant first-call delay you may be on a brand-new
  arch using the PTX JIT fallback; rebuild from source with your arch
  added to `SFA_CUDA_ARCH`.
