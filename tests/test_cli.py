"""Smoke tests for the CLI entry point and package metadata."""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

from infer_bench import __version__
from infer_bench.cli import collect_env, main


def test_version_is_resolved_from_installed_metadata() -> None:
    # Fails if the package is not installed, i.e. packaging is broken.
    assert __version__ != "0.0.0+unknown"


def test_version_flag_prints_version_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_env_command_reports_python_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env"]) == 0
    assert f"python: {platform.python_version()}" in capsys.readouterr().out


def test_collect_env_has_required_keys() -> None:
    env = collect_env()
    assert {"infer_bench", "python", "platform", "machine", "torch"} <= env.keys()


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_command_passes_on_repo_configs(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "validate",
            "--engines",
            str(REPO_ROOT / "configs" / "engines"),
            "--suite",
            str(REPO_ROOT / "configs" / "suite.yaml"),
        ]
    )
    assert code == 0
    assert "OK" in capsys.readouterr().out


def test_validate_command_fails_on_missing_configs(tmp_path: Path) -> None:
    assert main(["validate", "--engines", str(tmp_path), "--suite", "nope.yaml"]) == 1
