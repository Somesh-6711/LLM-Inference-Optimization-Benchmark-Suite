"""Prompt builder tests using a character-level fake tokenizer (no network)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from infer_bench.config import SuiteConfig, WorkloadSpec
from infer_bench.prompts import (
    SYSTEM_SENTINEL,
    USER_SENTINEL,
    ChatTemplate,
    PromptRecord,
    PromptSetError,
    build_workload,
    read_jsonl,
    verify_prompt_set,
    write_jsonl,
    write_prompt_set,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class CharTokenizer:
    """One token per character: lengths are easy to reason about in tests."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in ids)


TOKENIZER = CharTokenizer()
# head "<s>" = 3 tokens, mid "|" = 1, tail "</s>" = 4
TEMPLATE = ChatTemplate.from_rendered(f"<s>{SYSTEM_SENTINEL}|{USER_SENTINEL}</s>", TOKENIZER)
STREAM = list(range(0x4E00, 0x4E00 + 5000))  # 5000 distinct printable CJK characters


@pytest.fixture
def suite(tmp_path: Path) -> SuiteConfig:
    raw: dict[str, Any] = yaml.safe_load(
        (REPO_ROOT / "configs" / "suite.yaml").read_text(encoding="utf-8")
    )
    raw["tokenizer_revision"] = "main"
    raw["prompt_set"].update(
        {
            "directory": str(tmp_path / "prompts"),
            "count_per_workload": 4,
            "system_prompt": "sys",
            "user_instruction": "U:",
            "prefix_system_instruction": "DOC:",
            "prefix_user_instruction": "Q:",
        }
    )
    raw["workloads"] = {
        "plain_64": {"input_len": 64},
        "prefix_96": {"input_len": 96, "shared_prefix_len": 80},
    }
    raw["scenarios"] = {
        "single": {"engines": "all", "workloads": ["plain_64"], "output_lens": [8]},
        "prefix": {
            "engines": "servers",
            "workloads": ["prefix_96"],
            "output_lens": [8],
            "concurrency": [1, 2],
            "num_requests": 4,
        },
    }
    return SuiteConfig.model_validate(raw)


def _build(suite: SuiteConfig, name: str, spec: WorkloadSpec | None = None) -> list[PromptRecord]:
    return build_workload(
        name,
        spec or suite.workloads[name],
        tokenizer=TOKENIZER,
        template=TEMPLATE,
        stream=STREAM,
        prompt_set=suite.prompt_set,
    )


def _write(suite: SuiteConfig) -> None:
    write_prompt_set(
        suite,
        tokenizer=TOKENIZER,
        template=TEMPLATE,
        stream=STREAM,
        generator_versions={"fake": "1"},
    )


def test_plain_prompts_have_exact_length_and_template(suite: SuiteConfig) -> None:
    records = _build(suite, "plain_64")
    assert len(records) == 4
    for record in records:
        assert len(record.token_ids) == 64
        assert record.text.startswith("<s>sys|U:")
        assert record.text.endswith("</s>")
        assert record.shared_prefix_len == 0


def test_prefix_prompts_share_exact_prefix(suite: SuiteConfig) -> None:
    records = _build(suite, "prefix_96")
    assert len({tuple(r.token_ids[:80]) for r in records}) == 1
    assert len({tuple(r.token_ids[80:]) for r in records}) > 1
    assert records[0].text.startswith("<s>DOC:")
    assert "|Q:" in records[0].text


def test_build_is_deterministic(suite: SuiteConfig) -> None:
    assert _build(suite, "plain_64") == _build(suite, "plain_64")


def test_workloads_use_independent_rng_streams(suite: SuiteConfig) -> None:
    spec = suite.workloads["plain_64"]
    first = [r.token_ids for r in _build(suite, "plain_64", spec)]
    second = [r.token_ids for r in _build(suite, "another_name", spec)]
    assert first != second


def test_input_too_short_for_template_raises(suite: SuiteConfig) -> None:
    with pytest.raises(ValueError, match="too short"):
        _build(suite, "tiny", WorkloadSpec(input_len=10))


def test_record_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError):
        PromptRecord(
            prompt_id="x",
            workload="w",
            input_len=3,
            shared_prefix_len=0,
            token_ids=[1, 2],
            text="ab",
        )


def test_template_requires_each_sentinel_once() -> None:
    with pytest.raises(ValueError, match="sentinel"):
        ChatTemplate.from_rendered("<s>no sentinels here</s>", TOKENIZER)


def test_jsonl_roundtrip_and_stable_hash(tmp_path: Path, suite: SuiteConfig) -> None:
    records = _build(suite, "plain_64")
    path = tmp_path / "p.jsonl"
    digest = write_jsonl(records, path)
    assert read_jsonl(path) == records
    assert write_jsonl(records, path) == digest


def test_written_prompt_set_verifies(suite: SuiteConfig) -> None:
    _write(suite)
    loaded = verify_prompt_set(suite)
    assert set(loaded) == {"plain_64", "prefix_96"}


def test_tampered_file_fails_verification(suite: SuiteConfig) -> None:
    _write(suite)
    path = suite.prompt_set.directory / "plain_64.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("sys", "SYS", 1), encoding="utf-8")
    with pytest.raises(PromptSetError, match="sha256"):
        verify_prompt_set(suite)


def test_tokenizer_revision_mismatch_fails_verification(suite: SuiteConfig) -> None:
    _write(suite)
    other = suite.model_copy(update={"tokenizer_revision": "a" * 40})
    with pytest.raises(PromptSetError, match="tokenized with"):
        verify_prompt_set(other)


def test_missing_manifest_fails_verification(suite: SuiteConfig) -> None:
    with pytest.raises(PromptSetError, match="make prompts"):
        verify_prompt_set(suite)
