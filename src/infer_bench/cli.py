"""Command-line entry point for infer-bench."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from infer_bench import __version__
from infer_bench.config import check_consistency, load_engine_configs, load_suite
from infer_bench.prompts import verify_prompt_set

DEFAULT_ENGINES_DIR = Path("configs/engines")
DEFAULT_SUITE = Path("configs/suite.yaml")


def collect_env() -> dict[str, str]:
    """Return environment facts that will be attached to every benchmark result."""
    return {
        "infer_bench": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": "installed" if importlib.util.find_spec("torch") else "not installed",
    }


def cmd_validate(engines_dir: Path, suite_path: Path) -> int:
    try:
        engines = load_engine_configs(engines_dir)
        suite = load_suite(suite_path)
        check_consistency(suite, engines)
        prompt_sets = verify_prompt_set(suite)
    except (ValueError, OSError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"engines ({len(engines)}):")
    for config in engines.values():
        pin = "pinned" if config.model.is_pinned else "UNPINNED"
        print(f"  {config.name:<16} {config.engine.kind:<13} {config.model.quantization:<5} {pin}")

    print(f"prompt set {suite.prompt_set.version} ({suite.prompt_set.directory}):")
    for name, records in prompt_sets.items():
        spec = suite.workloads[name]
        caching = "on" if spec.prefix_caching else "off"
        print(
            f"  {name:<12} {len(records):>3} prompts  input_len={spec.input_len:<5} "
            f"shared_prefix={spec.shared_prefix_len:<4} prefix_caching={caching}"
        )

    unpinned = sorted(c.name for c in engines.values() if not c.model.is_pinned)
    if unpinned:
        print(f"warning: unpinned model revisions: {', '.join(unpinned)}", file=sys.stderr)
    print("OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer-bench",
        description="Benchmark Qwen2.5-1.5B-Instruct across seven inference configurations.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("env", help="Print environment info recorded in result metadata.")

    validate = subparsers.add_parser(
        "validate", help="Validate engine configs, suite, and the committed prompt set."
    )
    validate.add_argument("--engines", type=Path, default=DEFAULT_ENGINES_DIR)
    validate.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "env":
        for key, value in collect_env().items():
            print(f"{key}: {value}")
        return 0
    if args.command == "validate":
        return cmd_validate(args.engines, args.suite)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
