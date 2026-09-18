"""Schema and cross-file consistency tests for configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from infer_bench.config import (
    BASE_MODEL_ID,
    EXPECTED_ENGINE_NAMES,
    ConfigError,
    EngineConfig,
    ScenarioSpec,
    SuiteConfig,
    VllmEngine,
    WorkloadSpec,
    check_consistency,
    is_commit_sha,
    load_engine_config,
    load_engine_configs,
    load_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINES_DIR = REPO_ROOT / "configs" / "engines"
SUITE_PATH = REPO_ROOT / "configs" / "suite.yaml"


def _raw_suite() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    return data


def _engine_dict(model: dict[str, Any] | None = None, **engine: Any) -> dict[str, Any]:
    return {
        "name": "x",
        "description": "test",
        "model": model or {"repo_id": BASE_MODEL_ID, "revision": "main"},
        "engine": {"kind": "transformers", **engine},
    }


# ------------------------------------------------------------- committed configs


def test_repo_engine_configs_cover_all_seven() -> None:
    engines = load_engine_configs(ENGINES_DIR)
    assert set(engines) == EXPECTED_ENGINE_NAMES
    assert isinstance(engines["vllm"].engine, VllmEngine)
    assert engines["vllm"].is_server and not engines["pytorch_fp16"].is_server


def test_repo_suite_is_consistent_with_engines() -> None:
    check_consistency(load_suite(SUITE_PATH), load_engine_configs(ENGINES_DIR))


def test_base_model_revisions_are_pinned() -> None:
    # Quantized configs stay on 'main' until their repos exist (Days 9-10).
    assert is_commit_sha(load_suite(SUITE_PATH).tokenizer_revision)
    for config in load_engine_configs(ENGINES_DIR).values():
        if config.model.quantization == "none":
            assert config.model.is_pinned, config.name


# ------------------------------------------------------------------ engine schema


def test_unknown_field_is_rejected() -> None:
    raw = _engine_dict()
    raw["engine"] = {"kind": "vllm", "gpu_memory_utilisation": 0.9}  # British spelling typo
    with pytest.raises(ValidationError, match="gpu_memory_utilisation"):
        EngineConfig.model_validate(raw)


def test_bfloat16_is_rejected() -> None:
    raw = _engine_dict(model={"repo_id": BASE_MODEL_ID, "revision": "main", "dtype": "bfloat16"})
    with pytest.raises(ValidationError, match="T4"):
        EngineConfig.model_validate(raw)


def test_revision_must_be_sha_or_main() -> None:
    raw = _engine_dict(model={"repo_id": BASE_MODEL_ID, "revision": "v1.0"})
    with pytest.raises(ValidationError, match="commit SHA"):
        EngineConfig.model_validate(raw)


def test_quantization_requires_transformers_engine() -> None:
    raw = _engine_dict(model={"repo_id": "me/q", "revision": "main", "quantization": "awq"})
    raw["engine"] = {"kind": "vllm"}
    with pytest.raises(ValidationError, match="transformers engine only"):
        EngineConfig.model_validate(raw)


def test_quantized_config_cannot_point_at_base_model() -> None:
    raw = _engine_dict(model={"repo_id": BASE_MODEL_ID, "revision": "main", "quantization": "gptq"})
    with pytest.raises(ValidationError, match="quantized repo"):
        EngineConfig.model_validate(raw)


def test_compile_mode_requires_compile() -> None:
    with pytest.raises(ValidationError, match="torch_compile is false"):
        EngineConfig.model_validate(_engine_dict(compile_mode="default"))


def test_filename_must_match_name(tmp_path: Path) -> None:
    path = tmp_path / "01_something_else.yaml"
    path.write_text(yaml.safe_dump(_engine_dict()), encoding="utf-8")
    with pytest.raises(ConfigError, match="must match filename"):
        load_engine_config(path)


# ------------------------------------------------------------------- suite schema


def test_prefix_caching_is_derived_from_shared_prefix() -> None:
    assert WorkloadSpec(input_len=1024).prefix_caching is False
    assert WorkloadSpec(input_len=1024, shared_prefix_len=896).prefix_caching is True
    with pytest.raises(ValidationError):
        WorkloadSpec(input_len=128, shared_prefix_len=128)


def test_prompt_source_cannot_be_test_split() -> None:
    raw = _raw_suite()
    raw["prompt_set"]["source_split"] = "test"
    with pytest.raises(ValidationError):
        SuiteConfig.model_validate(raw)


def test_single_stream_engines_cannot_sweep_concurrency() -> None:
    with pytest.raises(ValidationError, match="concurrency"):
        ScenarioSpec(engines="all", workloads=["a"], output_lens=[8], concurrency=[1, 4])


def test_server_scenario_requires_num_requests() -> None:
    with pytest.raises(ValidationError, match="num_requests"):
        ScenarioSpec(engines="servers", workloads=["a"], output_lens=[8], concurrency=[4])


def test_scenario_must_reference_known_workload() -> None:
    raw = _raw_suite()
    raw["scenarios"]["single_stream"]["workloads"] = ["does_not_exist"]
    with pytest.raises(ValidationError, match="unknown workloads"):
        SuiteConfig.model_validate(raw)


# -------------------------------------------------------------- cross-file checks


def test_missing_engine_is_detected() -> None:
    engines = load_engine_configs(ENGINES_DIR)
    del engines["sglang"]
    with pytest.raises(ConfigError, match="missing"):
        check_consistency(load_suite(SUITE_PATH), engines)


def test_revision_mismatch_is_detected() -> None:
    suite = load_suite(SUITE_PATH).model_copy(update={"tokenizer_revision": "b" * 40})
    with pytest.raises(ConfigError, match="tokenizer_revision"):
        check_consistency(suite, load_engine_configs(ENGINES_DIR))


def test_context_budget_is_checked_against_server_engines() -> None:
    raw = _raw_suite()
    raw["scenarios"]["single_stream"]["output_lens"] = [4096]
    suite = SuiteConfig.model_validate(raw)
    with pytest.raises(ConfigError, match="context limit"):
        check_consistency(suite, load_engine_configs(ENGINES_DIR))
