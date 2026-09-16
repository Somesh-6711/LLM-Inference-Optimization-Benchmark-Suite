"""Command-line entry point for infer-bench."""

from __future__ import annotations

import argparse
import importlib.util
import platform
from collections.abc import Sequence

from infer_bench import __version__


def collect_env() -> dict[str, str]:
    """Return environment facts that will be attached to every benchmark result."""
    return {
        "infer_bench": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": "installed" if importlib.util.find_spec("torch") else "not installed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer-bench",
        description="Benchmark Qwen2.5-1.5B-Instruct across seven inference configurations.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("env", help="Print environment info recorded in result metadata.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "env":
        for key, value in collect_env().items():
            print(f"{key}: {value}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
