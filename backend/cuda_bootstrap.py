"""Make pip-installed CUDA runtime libraries discoverable to llama-cpp-python.

The prebuilt CUDA wheels of llama-cpp-python (cu124) link against libcudart.so.12
and libcublas.so.12 but do not bundle them. We get those binaries from the
`nvidia-cuda-runtime-cu12` and `nvidia-cublas-cu12` pip packages and surface them
to the dynamic loader before `llama_cpp` is imported.

Cross-platform:
- Linux:   preload the .so files via ctypes (bypasses LD_LIBRARY_PATH timing).
- Windows: register the bin/ dirs with os.add_dll_directory().

Silent no-op if the nvidia-* packages are missing (CPU-only fallback).
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _venv_site_packages() -> Path | None:
    for p in sys.path:
        if p.endswith("site-packages") and Path(p).is_dir():
            return Path(p)
    return None


def bootstrap() -> bool:
    site = _venv_site_packages()
    if site is None:
        return False

    nvidia_root = site / "nvidia"
    if not nvidia_root.is_dir():
        return False

    if sys.platform.startswith("win"):
        # Windows ships DLLs under nvidia/<pkg>/bin/
        added = False
        for sub in ("cuda_runtime", "cublas"):
            bin_dir = nvidia_root / sub / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))  # type: ignore[attr-defined]
                added = True
        return added

    # Linux: preload the shared objects so libllama.so resolves them
    loaded = False
    for sub, soname in (
        ("cublas", "libcublas.so.12"),       # depends on cudart, but ctypes RTLD_GLOBAL handles ordering
        ("cuda_runtime", "libcudart.so.12"),
    ):
        lib_path = nvidia_root / sub / "lib" / soname
        if lib_path.is_file():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                loaded = True
            except OSError:
                pass
    return loaded


bootstrap()
