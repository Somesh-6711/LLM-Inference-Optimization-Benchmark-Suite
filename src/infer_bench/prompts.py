"""Deterministic, token-exact prompt construction and verification.

Prompts are stored as token IDs so every engine receives byte-identical input,
independent of how each engine tokenizes text or applies chat templates.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

from pydantic import Field, PositiveInt, model_validator

from infer_bench.config import PromptSetSpec, StrictModel, SuiteConfig, WorkloadSpec

MANIFEST_NAME = "manifest.json"
SYSTEM_SENTINEL = "@@SYSTEM@@"
USER_SENTINEL = "@@USER@@"


class PromptSetError(ValueError):
    """Raised when a committed prompt set does not match the suite definition."""


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class PromptRecord(StrictModel):
    prompt_id: str
    workload: str
    input_len: PositiveInt
    shared_prefix_len: int = Field(ge=0)
    token_ids: list[int]
    text: str  # human-readable decode of token_ids; engines never consume it

    @model_validator(mode="after")
    def _lengths_consistent(self) -> Self:
        if len(self.token_ids) != self.input_len:
            raise ValueError(
                f"{self.prompt_id}: {len(self.token_ids)} tokens, expected {self.input_len}"
            )
        if self.shared_prefix_len >= self.input_len:
            raise ValueError(f"{self.prompt_id}: shared_prefix_len must be < input_len")
        return self


class ManifestEntry(StrictModel):
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    count: PositiveInt
    input_len: PositiveInt
    shared_prefix_len: int = Field(ge=0)


class PromptManifest(StrictModel):
    version: str
    seed: int
    tokenizer_repo: str
    tokenizer_revision: str
    source: str
    generator_versions: dict[str, str]
    workloads: dict[str, ManifestEntry]


@dataclass(frozen=True)
class ChatTemplate:
    """Token IDs of the chat template around the system and user content."""

    head: tuple[int, ...]
    mid: tuple[int, ...]
    tail: tuple[int, ...]

    @classmethod
    def from_rendered(cls, rendered: str, tokenizer: Tokenizer) -> ChatTemplate:
        """Split a template rendered with SYSTEM_SENTINEL / USER_SENTINEL as contents."""
        if rendered.count(SYSTEM_SENTINEL) != 1 or rendered.count(USER_SENTINEL) != 1:
            raise ValueError("rendered template must contain each sentinel exactly once")
        head, rest = rendered.split(SYSTEM_SENTINEL)
        mid, tail = rest.split(USER_SENTINEL)
        return cls(
            head=tuple(tokenizer.encode(head)),
            mid=tuple(tokenizer.encode(mid)),
            tail=tuple(tokenizer.encode(tail)),
        )

    def assemble(self, system_ids: Sequence[int], user_ids: Sequence[int]) -> list[int]:
        return [*self.head, *system_ids, *self.mid, *user_ids, *self.tail]


def _require_room(workload: str, part: str, length: int) -> None:
    if length <= 0:
        raise ValueError(
            f"workload '{workload}': input_len too short for chat template and "
            f"instructions ({part} would be {length} tokens)"
        )


def _slice(stream: Sequence[int], length: int, rng: random.Random) -> list[int]:
    if length > len(stream):
        raise ValueError(f"token stream ({len(stream)}) shorter than requested slice ({length})")
    start = rng.randrange(len(stream) - length + 1)
    return list(stream[start : start + length])


def build_workload(
    name: str,
    spec: WorkloadSpec,
    *,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    stream: Sequence[int],
    prompt_set: PromptSetSpec,
) -> list[PromptRecord]:
    """Build `count_per_workload` prompts of exactly `spec.input_len` tokens."""
    # Independent RNG per workload: adding a workload never reshuffles the others.
    rng = random.Random(f"{prompt_set.seed}:{name}")

    if spec.shared_prefix_len == 0:
        system_ids = tokenizer.encode(prompt_set.system_prompt)
        instruction = tokenizer.encode(prompt_set.user_instruction)
        body_len = (
            spec.input_len
            - len(template.head)
            - len(system_ids)
            - len(template.mid)
            - len(instruction)
            - len(template.tail)
        )
    else:
        system_instruction = tokenizer.encode(prompt_set.prefix_system_instruction)
        doc_len = (
            spec.shared_prefix_len
            - len(template.head)
            - len(template.mid)
            - len(system_instruction)
        )
        _require_room(name, "shared document", doc_len)
        system_ids = [*system_instruction, *_slice(stream, doc_len, rng)]
        instruction = tokenizer.encode(prompt_set.prefix_user_instruction)
        body_len = spec.input_len - spec.shared_prefix_len - len(instruction) - len(template.tail)

    _require_room(name, "user body", body_len)

    records: list[PromptRecord] = []
    for index in range(prompt_set.count_per_workload):
        user_ids = [*instruction, *_slice(stream, body_len, rng)]
        token_ids = template.assemble(system_ids, user_ids)
        records.append(
            PromptRecord(
                prompt_id=f"{name}_{index:03d}",
                workload=name,
                input_len=spec.input_len,
                shared_prefix_len=spec.shared_prefix_len,
                token_ids=token_ids,
                text=tokenizer.decode(token_ids),
            )
        )
    return records


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(records: Sequence[PromptRecord], path: Path) -> str:
    data = "".join(record.model_dump_json() + "\n" for record in records).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[PromptRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PromptRecord.model_validate_json(line) for line in lines if line]


def write_prompt_set(
    suite: SuiteConfig,
    *,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    stream: Sequence[int],
    generator_versions: dict[str, str],
) -> PromptManifest:
    prompt_set = suite.prompt_set
    prompt_set.directory.mkdir(parents=True, exist_ok=True)

    entries: dict[str, ManifestEntry] = {}
    for name in sorted(suite.workloads):
        spec = suite.workloads[name]
        records = build_workload(
            name,
            spec,
            tokenizer=tokenizer,
            template=template,
            stream=stream,
            prompt_set=prompt_set,
        )
        filename = f"{name}.jsonl"
        entries[name] = ManifestEntry(
            file=filename,
            sha256=write_jsonl(records, prompt_set.directory / filename),
            count=len(records),
            input_len=spec.input_len,
            shared_prefix_len=spec.shared_prefix_len,
        )

    manifest = PromptManifest(
        version=prompt_set.version,
        seed=prompt_set.seed,
        tokenizer_repo=suite.model_repo,
        tokenizer_revision=suite.tokenizer_revision,
        source=f"{prompt_set.source_dataset}/{prompt_set.source_config}:{prompt_set.source_split}",
        generator_versions=dict(sorted(generator_versions.items())),
        workloads=entries,
    )
    manifest_path = prompt_set.directory / MANIFEST_NAME
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_prompt_set(suite: SuiteConfig) -> dict[str, list[PromptRecord]]:
    """Check the committed prompt set against the suite; return records per workload."""
    prompt_set = suite.prompt_set
    manifest_path = prompt_set.directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PromptSetError(f"missing {manifest_path}; run `make prompts`")
    try:
        manifest = PromptManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PromptSetError(f"{manifest_path}: {exc}") from exc

    if manifest.version != prompt_set.version or manifest.seed != prompt_set.seed:
        raise PromptSetError("manifest version/seed differ from suite; run `make prompts`")
    if (
        manifest.tokenizer_repo != suite.model_repo
        or manifest.tokenizer_revision != suite.tokenizer_revision
    ):
        raise PromptSetError(
            f"prompt set was tokenized with {manifest.tokenizer_repo}@"
            f"{manifest.tokenizer_revision}, suite expects {suite.model_repo}@"
            f"{suite.tokenizer_revision}; run `make prompts`"
        )

    loaded: dict[str, list[PromptRecord]] = {}
    for name, spec in suite.workloads.items():
        entry = manifest.workloads.get(name)
        if entry is None:
            raise PromptSetError(f"workload '{name}' missing from manifest; run `make prompts`")
        path = prompt_set.directory / entry.file
        if sha256_file(path) != entry.sha256:
            raise PromptSetError(f"{path}: sha256 mismatch (edited by hand or stale manifest)")
        try:
            records = read_jsonl(path)
        except ValueError as exc:
            raise PromptSetError(f"{path}: {exc}") from exc

        if len(records) != entry.count or entry.count != prompt_set.count_per_workload:
            raise PromptSetError(f"{path}: expected {prompt_set.count_per_workload} prompts")
        for record in records:
            if (
                record.workload != name
                or record.input_len != spec.input_len
                or record.shared_prefix_len != spec.shared_prefix_len
            ):
                raise PromptSetError(f"{path}: {record.prompt_id} does not match workload spec")
        if spec.shared_prefix_len:
            prefixes = {tuple(r.token_ids[: spec.shared_prefix_len]) for r in records}
            if len(prefixes) != 1:
                raise PromptSetError(f"{path}: shared prefix differs across prompts")
        loaded[name] = records
    return loaded
