"""The committed prompt set must match configs/suite.yaml exactly."""

from __future__ import annotations

from pathlib import Path

from infer_bench.config import load_suite
from infer_bench.prompts import verify_prompt_set

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_prompt_set_matches_suite() -> None:
    suite = load_suite(REPO_ROOT / "configs" / "suite.yaml")
    loaded = verify_prompt_set(suite)
    assert set(loaded) == set(suite.workloads)
    for name, records in loaded.items():
        assert len(records) == suite.prompt_set.count_per_workload, name
