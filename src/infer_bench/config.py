"""Typed, validated configuration for engines, workloads, and the benchmark suite.

YAML files under configs/ are the single source of truth. Everything is validated
on load so a typo fails immediately instead of hours into a Kaggle session.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

EXPECTED_ENGINE_NAMES = frozenset(
    {
        "pytorch_fp16",
        "pytorch_compile",
        "onnxruntime",
        "gptq_int4",
        "awq_int4",
        "vllm",
        "sglang",
    }
)

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class ConfigError(ValueError):
    """Raised when a config file is invalid or configs are inconsistent with each other."""


def is_commit_sha(revision: str) -> bool:
    return _COMMIT_SHA.fullmatch(revision) is not None


def _check_revision(value: str) -> str:
    if value != "main" and not is_commit_sha(value):
        raise ValueError(
            "revision must be a full 40-character commit SHA('main' is allowed only before pinning)"
        )
    return value


class StrictModel(BaseModel):
    """Base for all config models: unknown keys are errors, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- model


class ModelSpec(StrictModel):
    repo_id: str
    revision: str
    dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    quantization: Literal["none", "gptq", "awq"] = "none"

    @field_validator("revision")
    @classmethod
    def _revision_format(cls, value: str) -> str:
        return _check_revision(value)

    @field_validator("dtype")
    @classmethod
    def _no_bfloat16(cls, value: str) -> str:
        if value == "bfloat16":
            raise ValueError(
                "bfloat16 is not natively supported on the T4 (compute capability 7.5); use float16"
            )
        return value

    @property
    def is_pinned(self) -> bool:
        return is_commit_sha(self.revision)


# ------------------------------------------------------------------------- engines


class TransformersEngine(StrictModel):
    kind: Literal["transformers"]
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"
    torch_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] | None = None

    @model_validator(mode="after")
    def _compile_mode_requires_compile(self) -> Self:
        if self.compile_mode is not None and not self.torch_compile:
            raise ValueError("compile_mode is set but torch_compile is false")
        return self


class OnnxRuntimeEngine(StrictModel):
    kind: Literal["onnxruntime"]
    provider: Literal["CUDAExecutionProvider", "CPUExecutionProvider"] = "CUDAExecutionProvider"
    use_io_binding: bool = True


class VllmEngine(StrictModel):
    kind: Literal["vllm"]
    gpu_memory_utilization: float = Field(default=0.85, gt=0.0, le=0.95)
    max_model_len: PositiveInt = 2048
    max_num_seqs: PositiveInt = 64
    enforce_eager: bool = False


class SglangEngine(StrictModel):
    kind: Literal["sglang"]
    mem_fraction_static: float = Field(default=0.80, gt=0.0, le=0.95)
    context_length: PositiveInt = 2048
    max_running_requests: PositiveInt = 64
    attention_backend: str | None = None  # decided on Day 12 (T4 compatibility check)


EngineSpec = Annotated[
    TransformersEngine | OnnxRuntimeEngine | VllmEngine | SglangEngine,
    Field(discriminator="kind"),
]

SERVER_ENGINE_KINDS = frozenset({"vllm", "sglang"})


class EngineConfig(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    model: ModelSpec
    engine: EngineSpec

    @property
    def is_server(self) -> bool:
        return self.engine.kind in SERVER_ENGINE_KINDS

    @model_validator(mode="after")
    def _check_combination(self) -> Self:
        quant = self.model.quantization
        if quant != "none":
            if self.engine.kind != "transformers":
                raise ValueError(
                    f"{quant} runs under the transformers engine only: "
                    "quantization is the variable, the engine is held constant"
                )
            if self.model.repo_id == BASE_MODEL_ID:
                raise ValueError(
                    f"{quant} config must point at a quantized repo, not the base model"
                )
            if isinstance(self.engine, TransformersEngine) and self.engine.torch_compile:
                raise ValueError("torch.compile is benchmarked on the fp16 baseline only")
        elif self.model.repo_id != BASE_MODEL_ID:
            raise ValueError(f"unquantized configs must use {BASE_MODEL_ID}")
        return self


# ----------------------------------------------------------------- suite / workloads


class WorkloadSpec(StrictModel):
    input_len: PositiveInt
    shared_prefix_len: int = Field(default=0, ge=0)

    @property
    def prefix_caching(self) -> bool:
        """Derived, never configured: caching is on only for prefix-sharing workloads."""
        return self.shared_prefix_len > 0

    @model_validator(mode="after")
    def _prefix_fits(self) -> Self:
        if self.shared_prefix_len >= self.input_len:
            raise ValueError("shared_prefix_len must be smaller than input_len")
        return self


class PromptSetSpec(StrictModel):
    version: str
    directory: Path  # relative paths are resolved against the suite file's directory
    seed: int
    count_per_workload: PositiveInt
    source_dataset: str
    source_config: str
    source_split: Literal["train"]  # the test split is reserved for perplexity
    system_prompt: str
    user_instruction: str
    prefix_system_instruction: str
    prefix_user_instruction: str


class ScenarioSpec(StrictModel):
    engines: Literal["all", "servers"]
    workloads: list[str] = Field(min_length=1)
    output_lens: list[PositiveInt] = Field(min_length=1)
    concurrency: list[PositiveInt] = Field(default=[1], min_length=1)
    num_requests: PositiveInt | None = None  # per concurrency level (server scenarios)

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.engines == "all" and self.concurrency != [1]:
            raise ValueError("scenarios that include non-server engines must use concurrency [1]")
        if self.engines == "servers" and self.num_requests is None:
            raise ValueError("server scenarios must set num_requests")
        return self


class RunPolicy(StrictModel):
    warmup_requests: PositiveInt = 3
    trials: int = Field(default=5, ge=3)  # fewer than 3 trials cannot show variance
    temperature: float = Field(default=0.0, ge=0.0)
    ignore_eos: Literal[True] = (
        True  # output length is part of the workload, not the model's choice
    )


class SuiteConfig(StrictModel):
    model_repo: str
    tokenizer_revision: str
    expected_gpu: str
    run: RunPolicy
    prompt_set: PromptSetSpec
    workloads: dict[str, WorkloadSpec] = Field(min_length=1)
    scenarios: dict[str, ScenarioSpec] = Field(min_length=1)

    @field_validator("tokenizer_revision")
    @classmethod
    def _revision_format(cls, value: str) -> str:
        return _check_revision(value)

    @field_validator("model_repo")
    @classmethod
    def _model_is_locked(cls, value: str) -> str:
        if value != BASE_MODEL_ID:
            raise ValueError(f"model_repo is locked to {BASE_MODEL_ID}")
        return value

    @model_validator(mode="after")
    def _scenarios_reference_known_workloads(self) -> Self:
        for scenario_name, scenario in self.scenarios.items():
            unknown = sorted(set(scenario.workloads) - self.workloads.keys())
            if unknown:
                raise ValueError(
                    f"scenario '{scenario_name}' references unknown workloads {unknown}"
                )
        return self


# ------------------------------------------------------------------------- loading


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def load_engine_config(path: Path) -> EngineConfig:
    try:
        config = EngineConfig.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"{path}:\n{exc}") from exc
    expected_name = path.stem.split("_", 1)[-1]  # "01_pytorch_fp16" -> "pytorch_fp16"
    if config.name != expected_name:
        raise ConfigError(f"{path}: name '{config.name}' must match filename '{expected_name}'")
    return config


def load_engine_configs(directory: Path) -> dict[str, EngineConfig]:
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise ConfigError(f"no engine configs found in {directory}")
    configs: dict[str, EngineConfig] = {}
    for path in paths:
        config = load_engine_config(path)
        if config.name in configs:
            raise ConfigError(f"duplicate engine name '{config.name}' in {path}")
        configs[config.name] = config
    return configs


def load_suite(path: Path) -> SuiteConfig:
    try:
        suite = SuiteConfig.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"{path}:\n{exc}") from exc
    directory = suite.prompt_set.directory
    if not directory.is_absolute():
        directory = (path.parent / directory).resolve()
    prompt_set = suite.prompt_set.model_copy(update={"directory": directory})
    return suite.model_copy(update={"prompt_set": prompt_set})


def check_consistency(suite: SuiteConfig, engines: Mapping[str, EngineConfig]) -> None:
    """Cross-file checks that no single schema can express."""
    missing = sorted(EXPECTED_ENGINE_NAMES - engines.keys())
    unexpected = sorted(engines.keys() - EXPECTED_ENGINE_NAMES)
    if missing or unexpected:
        raise ConfigError(f"engine set mismatch: missing={missing} unexpected={unexpected}")

    for config in engines.values():
        if (
            config.model.quantization == "none"
            and config.model.revision != suite.tokenizer_revision
        ):
            raise ConfigError(
                f"{config.name}: model revision '{config.model.revision}' != suite "
                f"tokenizer_revision '{suite.tokenizer_revision}'; prompts must be tokenized "
                "with the benchmarked model's tokenizer"
            )

    max_input = max(w.input_len for w in suite.workloads.values())
    max_output = max(max(s.output_lens) for s in suite.scenarios.values())
    for config in engines.values():
        engine = config.engine
        if isinstance(engine, VllmEngine):
            limit: int | None = engine.max_model_len
        elif isinstance(engine, SglangEngine):
            limit = engine.context_length
        else:
            limit = None
        if limit is not None and max_input + max_output > limit:
            raise ConfigError(
                f"{config.name}: context limit {limit} < longest request "
                f"{max_input} + {max_output} tokens"
            )
