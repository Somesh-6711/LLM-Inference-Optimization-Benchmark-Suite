"""Generate the committed, tokenized prompt set.

Run with: make prompts
Needs network access (Hugging Face Hub). Writes the JSONL files and a sha256
manifest to prompt_set.directory from configs/suite.yaml.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from infer_bench.config import is_commit_sha, load_suite
from infer_bench.prompts import (
    SYSTEM_SENTINEL,
    USER_SENTINEL,
    ChatTemplate,
    verify_prompt_set,
    write_prompt_set,
)

GENERATOR_PACKAGES = ("transformers", "tokenizers", "datasets")


class HFTokenizer:
    """Adapter exposing the minimal Tokenizer protocol used by infer_bench.prompts."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self._tokenizer.encode(text, add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(ids), skip_special_tokens=False))


def _chat(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _reference_ids(tokenizer: Any, system: str, user: str) -> list[int]:
    output = tokenizer.apply_chat_template(
        _chat(system, user), tokenize=True, add_generation_prompt=True
    )
    # Older transformers return list[int]; newer versions may return a BatchEncoding.
    ids = output if isinstance(output, list) else output["input_ids"]
    return [int(i) for i in ids]


def _check_splice(tokenizer: Any, template: ChatTemplate, wrapper: HFTokenizer) -> None:
    """Our token-level splice must equal what the official chat template produces."""
    system = "You are a helpful assistant."
    user = "Summarize this passage about the history of Rome."
    expected = _reference_ids(tokenizer, system, user)
    actual = template.assemble(wrapper.encode(system), wrapper.encode(user))
    if actual != expected:
        raise SystemExit(
            "template splice does not match apply_chat_template\n"
            f"  expected: {expected}\n  actual:   {actual}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/suite.yaml"))
    args = parser.parse_args(argv)

    suite = load_suite(args.suite)
    if not is_commit_sha(suite.tokenizer_revision):
        print("error: pin tokenizer_revision in the suite to a commit SHA first", file=sys.stderr)
        return 1
    prompt_set = suite.prompt_set

    print(f"tokenizer: {suite.model_repo}@{suite.tokenizer_revision[:12]}")
    tokenizer = AutoTokenizer.from_pretrained(suite.model_repo, revision=suite.tokenizer_revision)
    tokenizer.model_max_length = 10**9  # silences the long-input warning; never fed to a model
    wrapper = HFTokenizer(tokenizer)

    rendered = tokenizer.apply_chat_template(
        _chat(SYSTEM_SENTINEL, USER_SENTINEL), tokenize=False, add_generation_prompt=True
    )
    template = ChatTemplate.from_rendered(str(rendered), wrapper)
    _check_splice(tokenizer, template, wrapper)
    print(
        f"template overhead: head={len(template.head)} mid={len(template.mid)} "
        f"tail={len(template.tail)} tokens"
    )

    source = f"{prompt_set.source_dataset}/{prompt_set.source_config}:{prompt_set.source_split}"
    print(f"source: {source}")
    dataset = load_dataset(
        prompt_set.source_dataset, prompt_set.source_config, split=prompt_set.source_split
    )
    stream = wrapper.encode("".join(dataset["text"]))
    print(f"token stream: {len(stream):,} tokens")

    versions = {pkg: importlib.metadata.version(pkg) for pkg in GENERATOR_PACKAGES}
    manifest = write_prompt_set(
        suite, tokenizer=wrapper, template=template, stream=stream, generator_versions=versions
    )
    verify_prompt_set(suite)

    for name, entry in manifest.workloads.items():
        print(f"  {name:<12} {entry.count} prompts  sha256={entry.sha256[:12]}")
    print(f"wrote {prompt_set.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
