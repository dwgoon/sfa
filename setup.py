# -*- coding: utf-8 -*-
"""
SFA build script.

The pure-Python package builds with ``pip install .`` even on systems without
CUDA. If ``nvcc`` is discoverable and ``SFA_BUILD_CUDA`` is not "0", the
native extension ``sfa._cuda._native`` is also built against cuBLAS for
RTX 4090 (sm_89 by default; override with ``SFA_CUDA_ARCH``).

Toolchain expected from the ``sfa-cu132`` conda env (see environment-cuda.yml):
    cuda-nvcc 13.2, cuda-cudart-dev 13.2, libcublas-dev 13.x, pybind11.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()
CUDA_SRC = ROOT / "sfa" / "_cuda" / "src"


# --------------------------------------------------------------------------- #
# CUDA detection
# --------------------------------------------------------------------------- #
def _find_nvcc() -> Optional[Path]:
    if os.environ.get("SFA_BUILD_CUDA", "1") == "0":
        return None
    env_nvcc = os.environ.get("NVCC") or os.environ.get("CUDA_NVCC")
    if env_nvcc and Path(env_nvcc).exists():
        return Path(env_nvcc)
    found = shutil.which("nvcc")
    if found:
        return Path(found)
    for prefix_env in ("CONDA_PREFIX", "CONDA_PREFIX_1"):
        prefix = os.environ.get(prefix_env)
        if not prefix:
            continue
        cand = Path(prefix) / ("Library/bin/nvcc.exe" if os.name == "nt"
                               else "bin/nvcc")
        if cand.exists():
            return cand
    return None


def _cuda_home(nvcc: Path) -> Path:
    """Return the directory that contains 'include' and 'lib' for CUDA.

    On a conda Windows env (e.g. ``<env>/Library/bin/nvcc.exe``) this is
    ``<env>/Library``. On a standard system install or POSIX it is the
    parent of the ``bin`` directory holding nvcc.
    """
    p = nvcc.parent
    if p.name.lower() == "bin":
        return p.parent
    return p.parent


NVCC = _find_nvcc()
CUDA_AVAILABLE = NVCC is not None
CUDA_HOME = _cuda_home(NVCC) if NVCC else None


# --------------------------------------------------------------------------- #
# Custom build_ext that compiles .cu through nvcc
# --------------------------------------------------------------------------- #
class CudaBuildExt(build_ext):
    user_options = build_ext.user_options + [
        ("cuda-arch=", None,
         "Semicolon-separated CUDA compute capabilities (e.g. "
         "'sm_75;sm_80;sm_86;sm_89;sm_90'). Default is the wheel-wide AOT "
         "matrix; set SFA_CUDA_ARCH=sm_89 for a single-target dev build."),
    ]

    # Default AOT matrix: covers consumer + datacenter NVIDIA GPUs from
    # Volta (sm_70) through Hopper (sm_90). The highest arch is *also*
    # emitted as PTX so unknown future GPUs can JIT from PTX as a fallback.
    DEFAULT_ARCH_LIST = "sm_70;sm_75;sm_80;sm_86;sm_89;sm_90"

    def initialize_options(self):
        super().initialize_options()
        self.cuda_arch = os.environ.get(
            "SFA_CUDA_ARCH", CudaBuildExt.DEFAULT_ARCH_LIST)

    def _gencode_flags(self):
        """Translate the cuda_arch list into -gencode flags for nvcc.

        Each arch like ``sm_89`` becomes
        ``-gencode arch=compute_89,code=sm_89`` for AOT SASS.
        For the highest arch we additionally emit
        ``-gencode arch=compute_XX,code=compute_XX`` so the resulting fat
        binary carries PTX for forward-compat JIT on newer GPUs.
        """
        archs = [a.strip() for a in self.cuda_arch.replace(",", ";").split(";")
                 if a.strip()]
        flags = []
        for a in archs:
            if not a.startswith("sm_"):
                raise ValueError(
                    f"SFA_CUDA_ARCH entry {a!r} must be of form 'sm_XX'")
            num = a[3:]
            flags += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if archs:
            highest = archs[-1][3:]
            flags += ["-gencode",
                      f"arch=compute_{highest},code=compute_{highest}"]
        return flags

    def build_extensions(self):
        if not CUDA_AVAILABLE:
            print("[sfa] nvcc not found; skipping CUDA extension build.",
                  file=sys.stderr)
            self.extensions = [e for e in self.extensions
                               if not getattr(e, "is_cuda", False)]
            super().build_extensions()
            return

        # Register .cu so distutils' object_filenames() recognises it.
        if ".cu" not in self.compiler.src_extensions:
            self.compiler.src_extensions.append(".cu")

        nvcc_path = str(NVCC)
        gencode_flags = self._gencode_flags()
        host_flag = "/O2 /MD /EHsc" if os.name == "nt" else "-fPIC"
        print(f"[sfa] nvcc AOT targets: {self.cuda_arch}", file=sys.stderr)

        def _nvcc_compile(src: str, obj: str):
            cmd = [
                nvcc_path,
                "-c", src,
                "-o", obj,
            ] + gencode_flags + [
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-Xcompiler", host_flag,
            ]
            for inc in self.compiler.include_dirs:
                cmd += ["-I", inc]
            cmd += [
                f"-I{CUDA_HOME / 'include'}",
                f"-I{CUDA_HOME / 'include' / 'targets' / 'x64'}",
                f"-I{CUDA_HOME / 'include' / 'targets' / 'x64' / 'cccl'}",
            ]
            print("[nvcc]", " ".join(cmd))
            subprocess.check_call(cmd)

        original_compile = self.compiler.compile

        def compile_wrapper(sources, output_dir=None, macros=None,
                            include_dirs=None, debug=0, extra_preargs=None,
                            extra_postargs=None, depends=None):
            cu_sources = [s for s in sources if s.endswith(".cu")]
            cxx_sources = [s for s in sources if not s.endswith(".cu")]

            objs: List[str] = []
            for src in cu_sources:
                obj_path = self.compiler.object_filenames(
                    [src], output_dir=output_dir or "")[0]
                Path(obj_path).parent.mkdir(parents=True, exist_ok=True)
                _nvcc_compile(src, obj_path)
                objs.append(obj_path)

            if cxx_sources:
                objs += original_compile(
                    cxx_sources, output_dir=output_dir, macros=macros,
                    include_dirs=include_dirs, debug=debug,
                    extra_preargs=extra_preargs, extra_postargs=extra_postargs,
                    depends=depends)
            return objs

        self.compiler.compile = compile_wrapper
        try:
            super().build_extensions()
        finally:
            self.compiler.compile = original_compile


def _cuda_extension() -> Optional[Extension]:
    if not CUDA_AVAILABLE:
        return None
    try:
        import pybind11
    except ImportError:
        print("[sfa] pybind11 not importable; skipping CUDA extension.",
              file=sys.stderr)
        return None

    # setuptools requires source paths to be relative POSIX paths.
    sources = [
        "sfa/_cuda/src/bindings.cpp",
        "sfa/_cuda/src/influence_iter.cu",
        "sfa/_cuda/src/signal_prop_iter.cu",
    ]
    inc_dirs = [
        str(CUDA_SRC),
        pybind11.get_include(),
        str(CUDA_HOME / "include"),
        # CCCL ('nv/target' etc.) ships under targets/x64 in the conda layout.
        str(CUDA_HOME / "include" / "targets" / "x64"),
        str(CUDA_HOME / "include" / "targets" / "x64" / "cccl"),
    ]
    inc_dirs = [p for p in inc_dirs if Path(p).exists()]
    if os.name == "nt":
        lib_dirs = [
            str(CUDA_HOME / "lib" / "x64"),
            str(CUDA_HOME / "lib"),
        ]
    else:
        lib_dirs = [str(CUDA_HOME / "lib64"), str(CUDA_HOME / "lib")]
    lib_dirs = [p for p in lib_dirs if Path(p).exists()]

    # Linux: embed an $ORIGIN-relative rpath so the shipped wheel can
    # locate libcublas/libcudart that arrive via the nvidia-* PyPI
    # packages declared in install_requires. The .so lives at
    # ``site-packages/sfa/_cuda/_native.so``; the nvidia libs land at
    # ``site-packages/nvidia/{cublas,cuda_runtime}/lib`` so the relative
    # path is ``$ORIGIN/../../nvidia/...``.
    extra_link_args = []
    if os.name != "nt":
        extra_link_args += [
            "-Wl,-rpath,$ORIGIN/../../nvidia/cublas/lib",
            "-Wl,-rpath,$ORIGIN/../../nvidia/cuda_runtime/lib",
            "-Wl,-rpath,$ORIGIN/../../nvidia/cuda_nvrtc/lib",
        ]

    ext = Extension(
        "sfa._cuda._native",
        sources=sources,
        include_dirs=inc_dirs,
        library_dirs=lib_dirs,
        libraries=["cudart", "cublas"],
        language="c++",
        extra_compile_args=(["/std:c++17", "/O2", "/EHsc"]
                            if os.name == "nt"
                            else ["-std=c++17", "-O3", "-fPIC"]),
        extra_link_args=extra_link_args,
    )
    ext.is_cuda = True
    return ext


ext_modules = []
if (cuda_ext := _cuda_extension()) is not None:
    ext_modules.append(cuda_ext)


# Per-build extra install_requires fed in by the CI matrix. Used to pin
# the NVIDIA runtime PyPI packages (nvidia-cublas-cuXX,
# nvidia-cuda-runtime-cuXX) that match the CUDA major version we are
# building against. The value is a whitespace-separated list of pip
# requirement specifiers, e.g.
#   SFA_CUDA_RUNTIME_REQUIRES="nvidia-cublas-cu13>=13.2,<13.3 \
#                              nvidia-cuda-runtime-cu13>=13.2,<13.3"
#
# The wheel's static metadata (name, version, description, author, etc.)
# lives in pyproject.toml's [project] table; setup.py only supplies what
# isn't there: the CUDA build extension, the dynamic install_requires
# composed from the base list plus _extra_runtime, and the
# build-extension command class.
_extra_runtime = [
    spec for spec in os.environ.get("SFA_CUDA_RUNTIME_REQUIRES", "").split()
    if spec.strip()
]


setup(
    install_requires=(["numpy", "scipy", "pandas", "networkx",
                       "threadpoolctl"]
                      + _extra_runtime),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CudaBuildExt} if ext_modules else {},
    zip_safe=False,
)
