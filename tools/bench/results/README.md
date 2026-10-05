# Benchmark results

Keep compact summaries, methodology, launch settings, input hashes, calibrated
STS temperatures, and reusable tuning choices here. Reports under
[`docs/research`](../../../docs/research/) explain measurement boundaries and limitations.
The small target-verification `results.jsonl` retains its paired timing samples.

Historical sweep logs, request outputs, generated tokens, traces, and large prompt
or input files are available in the [immutable evidence archive](https://github.com/jiazhihao/mpk-apple/tree/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/)
at commit `ade38fb5f81ebdf852a2b65a616703b03f4ec424`. This cleanup does not change retained results or runtime settings.
The earlier serving study has its [own archive and restore instructions](qwen8b-serving-decode-20260929/README.md).

| Study | Raw evidence | Report |
|---|---|---|
| Kernel sweeps and early decode baselines | [Top-level JSONL files](https://github.com/jiazhihao/mpk-apple/tree/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/) | [GEMV study](../../../docs/research/gemv-kernel-study.md), [decode kernels](../../../docs/research/decode-kernels.md) |
| Llama and SmolLM2 generation | [Top-level JSONL files](https://github.com/jiazhihao/mpk-apple/tree/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/) | [Model comparison](../../../docs/research/llama-mlx-comparison.md) |
| Normalization fusion and grid searches | [Top-level JSONL files](https://github.com/jiazhihao/mpk-apple/tree/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/) | [Fusion study](../../../docs/research/norm-projection-fusion.md) |
| Long-context decode | [Request records and prompts](https://github.com/jiazhihao/mpk-apple/tree/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/qwen8b-long-context-20260929/) | [Long-context study](../../../docs/research/qwen8b-long-context-tuning.md) |
| Target-only verification | [Exact inputs and samples](https://github.com/jiazhihao/mpk-apple/tree/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/qwen8b-target-verify-20260930/) | [Target batch comparison](../../../docs/research/qwen8b-target-batch-gap.md) |

Restore the evidence into a temporary directory without adding it back to the checkout:

```bash
benchmark_evidence=$(mktemp -d)
git archive ade38fb5f81ebdf852a2b65a616703b03f4ec424 tools/bench/results/ | tar -x -C "$benchmark_evidence"
# The M1 reader expects only its kernel records, not later generation-study rows.
mkdir "$benchmark_evidence/m1"
cp "$benchmark_evidence"/tools/bench/results/apple-m5-pro-20c_{gemv_m1,nvfp4_decode,kernel_knobs,mlx_baseline}.jsonl "$benchmark_evidence/m1/"
python tools/bench/m1_gate_table.py "$benchmark_evidence/m1/apple-m5-pro-20c"
```

Individual files can also be extracted with `git show <commit>:<path>`; the reports
include commands for restoring their exact prompt/input files. The archive links
remain usable from shallow clones that do not contain the evidence commit locally.
