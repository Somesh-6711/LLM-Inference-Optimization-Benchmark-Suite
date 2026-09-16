# LLM Inference Optimization Benchmark Suite

![CI](https://github.com/Somesh-6711/LLM-Inference-Optimization-Benchmark-Suite/actions/workflows/ci.yml/badge.svg)

Reproducible benchmark of **Qwen2.5-1.5B-Instruct** across seven inference configurations, measured on a single NVIDIA T4 (Kaggle free tier).

> **Status:** Week 1, in progress. No benchmark results yet. Every number in this repo will be traceable to a committed run.

## Configurations

| # | Configuration | What it tests |
|---|---|---|
| 1 | PyTorch fp16 | Baseline |
| 2 | PyTorch + `torch.compile` | Graph compilation |
| 3 | ONNX Runtime (CUDA EP) | Alternative runtime |
| 4 | GPTQ 4-bit | Weight quantization |
| 5 | AWQ 4-bit | Activation-aware quantization |
| 6 | vLLM | PagedAttention + continuous batching |
| 7 | SGLang | RadixAttention prefix caching |

## Metrics

Latency (P50/P95/P99), time-to-first-token, inter-token latency, throughput (tokens/s), peak GPU memory, and perplexity on a held-out WikiText-2 subset.

## Methodology (summary)

- **Scenario A, single-stream:** batch size 1, all seven configs.
- **Scenario B, concurrency sweep:** 1/4/16/64 concurrent requests, serving engines only.
- **Prefix-heavy workload:** shared system prompt; vLLM runs with prefix caching enabled for a fair comparison with SGLang.
- Fixed input/output lengths, warmup runs, multiple trials, bootstrap confidence intervals.

## Development

Requires [uv](https://docs.astral.sh/uv/) and `make`.

```bash
git clone https://github.com/Somesh-6711/LLM-Inference-Optimization-Benchmark-Suite.git
cd LLM-Inference-Optimization-Benchmark-Suite
make install   # venv + dev tools + git hooks
make check     # lint, type check, tests
```

## License

MIT
