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
    p = nvcc.parent
    if p.name.lower() == "bin":
        if p.parent.name.lower() == "library":
            return p.parent.parent
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
        ("cuda-arch=", None, "CUDA compute capability, default sm_89 (RTX 4090)"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.cuda_arch = os.environ.get("SFA_CUDA_ARCH", "sm_89")

    def build_extensions(self):
        if not CUDA_AVAILABLE:
            print("[sfa] nvcc not found; skipping CUDA extension build.",
                  file=sys.stderr)
            self.extensions = [e for e in self.extensions
                               if not getattr(e, "is_cuda", False)]
            super().build_extensions()
            return

        nvcc_path = str(NVCC)
        cuda_arch = self.cuda_arch
        host_flag = "/O2 /MD /EHsc" if os.name == "nt" else "-fPIC"

        def _nvcc_compile(src: str, obj: str):
            cmd = [
                nvcc_path,
                "-c", src,
                "-o", obj,
                f"-arch={cuda_arch}",
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-Xcompiler", host_flag,
            ]
            for inc in self.compiler.include_dirs:
                cmd += ["-I", inc]
            cmd += [f"-I{CUDA_HOME / 'include'}"]
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

    sources = [
        str(CUDA_SRC / "bindings.cpp"),
        str(CUDA_SRC / "influence_iter.cu"),
    ]
    inc_dirs = [
        str(CUDA_SRC),
        pybind11.get_include(),
        str(CUDA_HOME / "include"),
    ]
    if os.name == "nt":
        lib_dirs = [
            str(CUDA_HOME / "lib" / "x64"),
            str(CUDA_HOME / "Library" / "lib" / "x64"),
            str(CUDA_HOME / "lib"),
        ]
    else:
        lib_dirs = [str(CUDA_HOME / "lib64"), str(CUDA_HOME / "lib")]
    lib_dirs = [p for p in lib_dirs if Path(p).exists()]

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
    )
    ext.is_cuda = True
    return ext


ext_modules = []
if (cuda_ext := _cuda_extension()) is not None:
    ext_modules.append(cuda_ext)


setup(
    name="sfa",
    version="0.2.0.dev0",
    description="Signal flow analysis",
    url="http://github.com/dwgoon/sfa",
    author="Daewon Lee",
    author_email="daewon4you@gmail.com",
    license="MIT",
    packages=find_packages(),
    package_data={"": ["*.tsv", "*.sif", "*.json"]},
    python_requires=">=3.10",
    install_requires=["numpy", "scipy", "pandas", "networkx"],
    extras_require={
        "plot": ["matplotlib", "seaborn"],
        "cuda": [],  # toolchain provided by conda env (environment-cuda.yml)
        "test": ["pytest"],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": CudaBuildExt} if ext_modules else {},
    zip_safe=False,
)
