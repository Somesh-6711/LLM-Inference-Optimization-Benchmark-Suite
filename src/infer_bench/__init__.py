"""infer_bench: reproducible LLM inference benchmarking."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("infer-bench")
except PackageNotFoundError:  # running from an uninstalled source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
