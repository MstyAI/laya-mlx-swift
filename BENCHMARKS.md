# Benchmarks

These numbers measure one request containing three decisions: department routing, refund detection, and urgency scoring.

| Runtime | Question accuracy | Exact cases | P50 | P95 |
|---|---:|---:|---:|---:|
| Laya MLX Swift | 74% | 36% | 32.4 ms | 33.4 ms |
| Laya MLX Python | 74% | 36% | 27.1 ms | 27.9 ms |
| Laya ONNX | 74% | 36% | 397 ms | 420 ms |
| Jev 1.13.0 | 96% | 88% | 202 ms | 312 ms |

The native Swift port is about 19% slower than Python MLX and about 12 times faster than the ONNX port on this workload. All three Laya runtimes produced the same accuracy. Jev uses a different hosted model, so its accuracy and network latency are useful context rather than a direct runtime comparison.

<details>
<summary>Method</summary>

- Machine: Apple M3 Max, 16-core CPU, 40-core GPU, 128 GB memory
- Model: `aac6fef/laya-mlx` at `20aed815fc6acde75733882e7ec0e3f28aeb9717`
- Dataset: 50 synthetic support requests, with three labeled decisions per request
- Timing: five measured passes after one warmup request; model loading excluded
- Swift and Python: release/optimized execution using MLX and FP16 weights
- ONNX: Core ML execution provider with CPU fallback
- Jev: hosted API latency including the network request

The dataset is small and synthetic. It tests this routing workload, not general reasoning quality.

Swift raw samples and predictions are in [`Benchmarks/results/m3-max-support.json`](Benchmarks/results/m3-max-support.json), with the Python comparison in [`Benchmarks/results/m3-max-python-mlx-summary.json`](Benchmarks/results/m3-max-python-mlx-summary.json). The earlier ONNX and Jev evidence is in the [`laya-onnx` benchmark report](https://github.com/MstyAI/laya-onnx/blob/main/evidence/jev-vs-laya-2026-09-20.json).

Run the Swift benchmark with:

```bash
xcodebuild \
  -scheme laya-mlx-swift \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation \
  -only-testing:LayaMLXTests/SupportBenchmarkTests/testSupportBenchmark \
  test
```

The benchmark uses the prepared default model or the standard Hugging Face cache location.

</details>
