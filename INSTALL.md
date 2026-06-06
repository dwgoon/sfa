# Install

SFA is distributed as one CPU-only wheel plus a family of CUDA wheels,
one per supported CUDA major/minor pair. Each CUDA wheel bundles its
matching cuBLAS at build time, so no separate CUDA toolkit install is
required at runtime - only an NVIDIA driver new enough for that CUDA
version.

| Package      | CUDA    | Min. NVIDIA driver | Platforms              |
|--------------|---------|--------------------|------------------------|
| `sfa`        | none    | -                  | Linux, macOS, Windows  |
| `sfa-cu128`  | 12.8.x  | 570 (Linux / Win)  | Linux, Windows         |
| `sfa-cu132`  | 13.2.x  | 580                | Linux, Windows         |
| `sfa-cu133`  | 13.3.x  | 580                | Linux, Windows         |

All CUDA wheels share the same AOT-compiled SASS matrix (SM 7.0
through SM 12.0: Volta, Turing, Ampere, Ada, Hopper, Blackwell), plus
a PTX fallback that the driver JIT-compiles for newer GPUs. The only
practical difference between `sfa-cuXYZ` variants is which CUDA
runtime they link against.

This file is a quick reference. See
[doc/install.md](doc/install.md) for the full guide.

## CPU (any OS)

```bash
pip install sfa
```

Requires Python 3.10+. macOS Apple Silicon and Intel are both supported.

## CUDA (Linux or Windows + NVIDIA GPU)

Pick **one** variant that matches your NVIDIA driver and install only
that one. Run `nvidia-smi` and read the "CUDA Version" column - that
is the maximum CUDA version your driver supports.

| Variant     | CUDA bundled | Minimum NVIDIA driver | When to pick                                              |
|-------------|--------------|------------------------|-----------------------------------------------------------|
| `sfa-cu133` | 13.3.x       | 580                    | Newest hardware / drivers; default for fresh installs.    |
| `sfa-cu132` | 13.2.x       | 580                    | Matches the `sfa-cu132` conda env used for development.   |
| `sfa-cu128` | 12.8.x       | 570                    | Older driver (CUDA 12 line); broadest backwards compat.   |

Example (install the newest variant):

```bash
pip install sfa-cu133
```

Requires Python 3.10+. macOS is not supported because Apple ended
NVIDIA driver development in 2019.

> [!IMPORTANT]
> `sfa` and `sfa-cuXYZ` install into the same `sfa` Python namespace.
> Install **only one** variant per environment; mixing them causes
> import conflicts.

## Optional extras

Matplotlib-based plotting helpers in `sfa.plot`:

```bash
pip install "sfa[plot]"
```

## Development install (build from source)

Two supported paths: the conda-based one (recommended for first-time
setup because it bundles the entire CUDA toolchain) and a conda-free
one that relies on a system CUDA install. Pick whichever fits your
environment.

In both cases, a host C++ compiler is required for the pybind11
extension - MSVC (Visual Studio Build Tools or Community) on Windows,
GCC/Clang on Linux, Clang on macOS. The CUDA toolkit does not include
the host compiler, and `conda` will not install it for you.

### Option A: conda-based build (recommended)

```bash
git clone https://github.com/dwgoon/sfa.git && cd sfa

conda env create -f environment-cuda.yml
conda activate sfa-cu132
pip install -e .                 # builds the CUDA extension via the env's nvcc

# CPU-only variant (skip CUDA even if nvcc is on PATH):
SFA_BUILD_CUDA=0 pip install -e .
```

This is also how the project maintainers build on Windows: the
`sfa-cu132` env provides `nvcc` and cuBLAS, while system MSVC handles
`bindings.cpp`. The resulting extension is e.g.
`sfa/_cuda/_native.cp312-win_amd64.pyd`.

The shipped `environment-cuda.yml` pins CUDA 13.2 simply because that
is what the maintainers test against. The same workflow works for any
CUDA major / minor that has a `cuda-toolkit` build on the `nvidia`
channel: edit the two `cuda-version` / `cuda-toolkit` pins in lockstep
(see [What `environment-cuda.yml` provides](#what-environment-cudayml-provides)
below) and rename the env on the first line of the file. CUDA 12.8 and
13.3 environments have been tested in CI.

### Option B: conda-free build (system CUDA + system C++ compiler)

Conda is purely a convenience for sourcing the CUDA toolchain - the
build itself only needs `nvcc` and cuBLAS reachable from `setup.py`,
plus a host C++ compiler. If those are already installed system-wide
(e.g. from NVIDIA's CUDA installer), you can skip conda entirely.

Prerequisites:

- Python 3.10+ (system Python, `pyenv`, `uv`, or any other manager).
- NVIDIA CUDA Toolkit installed system-wide, with `nvcc` on `PATH`.
  Verify with `nvcc --version`. Any reasonably recent CUDA version
  works: 11.x, 12.x, and 13.x have all been used successfully. The
  resulting build links against whichever cuBLAS ships with that
  toolkit.
- A host C++ compiler installed system-wide:
    - **Windows**: Visual Studio Build Tools or Visual Studio Community
      with the "Desktop development with C++" workload. Run subsequent
      `pip` commands from a "x64 Native Tools Command Prompt for VS"
      shell, or activate the MSVC environment via `vcvarsall.bat`.
      Match the MSVC version against your CUDA's compatibility table
      (`nvcc` will refuse newer MSVC than it has been validated with).
    - **Linux**: `gcc` and `g++` (whatever the distro provides). CUDA
      versions cap the maximum supported GCC: CUDA 12 supports up to
      GCC 13, CUDA 13 up to GCC 14.
    - **macOS**: Apple's Clang via Xcode Command Line Tools. CUDA is
      unsupported on macOS, so this path is CPU-only there.
- `pip`-installable build deps will be pulled in automatically by
  `pyproject.toml`'s `build-system.requires`
  (`setuptools>=68 wheel pybind11>=2.13 numpy>=2.0`).

Build steps (Linux / macOS):

```bash
git clone https://github.com/dwgoon/sfa.git && cd sfa

python -m venv .venv
source .venv/bin/activate

# Optional but explicit: tell setup.py where nvcc is. Otherwise it
# falls back to `shutil.which("nvcc")`.
export NVCC=$(which nvcc)

pip install -e .                 # CUDA extension built if nvcc is found
# or:
SFA_BUILD_CUDA=0 pip install -e .   # pure-Python install
```

Build steps (Windows, x64 Native Tools Command Prompt):

```bat
git clone https://github.com/dwgoon/sfa.git
cd sfa

python -m venv .venv
.\.venv\Scripts\activate

REM Optional: pin nvcc explicitly. Replace <version> with your
REM installed CUDA, e.g. v13.0, v13.2, v12.8.
set NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\<version>\bin\nvcc.exe

pip install -e .
REM or:
set SFA_BUILD_CUDA=0
pip install -e .
```

The build is CUDA-version-agnostic: whichever toolkit your `NVCC`
(or `PATH`) points at is what `setup.py` will use, and the resulting
extension will link against that toolkit's cuBLAS.

If `nvcc` is not on `PATH`, `setup.py` checks the `NVCC` and
`CUDA_NVCC` environment variables, then `CONDA_PREFIX`, then gives up
and falls through to a CPU-only build (printing
`[sfa] nvcc not found; skipping CUDA extension build.`).

### What `environment-cuda.yml` provides

The shipped conda environment file creates a self-contained build
environment named `sfa-cu132` that does **not** require any
system-wide CUDA install. Everything the build needs - the CUDA
compiler, the CUDA runtime, cuBLAS headers and import libs, plus the
Python build and runtime dependencies - is pulled in from the
`nvidia` and `conda-forge` channels.

Concretely, the file pins:

| Group                | Packages (channel)                                                                                  |
|----------------------|-----------------------------------------------------------------------------------------------------|
| CUDA toolchain       | `cuda-version=13.2`, `cuda-toolkit=13.2` (nvidia)                                                   |
| cuBLAS for linking   | `libcublas-dev` (nvidia)                                                                            |
| Python build deps    | `pybind11>=2.13`, `setuptools>=68`, `wheel`, `ninja` (conda-forge)                                  |
| Runtime deps         | `numpy>=2.0`, `scipy`, `pandas`, `networkx`, `threadpoolctl` (conda-forge)                          |
| Test / bench         | `pytest` (conda-forge)                                                                              |
| Python               | `python=3.12` (conda-forge transitive)                                                              |

The `cuda-toolkit` meta-package pulls in `nvcc`, `cudart`, `nvrtc`,
`cccl`, `cupti`, the profiler API, and the rest of the CUDA dev
toolchain. After `conda activate sfa-cu132`, `nvcc` is on `PATH` and
`setup.py`'s CUDA-extension build picks it up automatically.

Notes for adjusting the file:

- To target a different CUDA major version, change the two `nvidia::`
  pins (`cuda-version` and `cuda-toolkit`) in lockstep. The env name
  on the first line (`sfa-cu132`) is just a label; rename it freely.
- A host C++ compiler is still required (MSVC on Windows, GCC on
  Linux). The toolchain itself is not bundled by `cuda-toolkit`;
  conda will not install it for you.
- To target a single GPU architecture during dev iteration, set
  `SFA_CUDA_ARCH=sm_89` (or your card's SM) before `pip install -e .`
  to skip the full AOT matrix.

### `setup.py` environment variables

| Variable             | Purpose                                                                |
|----------------------|------------------------------------------------------------------------|
| `SFA_BUILD_CUDA`     | `0` to force a pure-Python install. Default: build if `nvcc` is found. |
| `SFA_CUDA_ARCH`      | Semicolon-separated SM list, e.g. `sm_89` (dev) or `sm_70;sm_80;sm_89`. Default: the full wheel-wide AOT matrix. |
| `SFA_PACKAGE_NAME`   | Override the PyPI name (used by CI to produce e.g. `sfa-cu132` or `sfa-cu133` from the same source tree). |

## Verify the install

```bash
python tests/verification.py
```

prints `ALL OK` on success. It exercises the CPU LAPACK fast path, the
SignalPropagation trajectory, and (when an NVIDIA GPU is visible) a
CUDA influence computation.

For the full test suite:

```bash
python -m pip install pytest
python -m pytest tests/
```

CUDA tests auto-skip on machines without an NVIDIA GPU.
