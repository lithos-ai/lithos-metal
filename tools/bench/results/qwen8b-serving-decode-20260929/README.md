# Qwen3 8B serving evidence

This directory keeps `summary.json`, `round-profile-summary.json`, and
`methodology.json`: compact results, measurement settings, versions, and launches.
The [report](../../../../docs/research/qwen8b-serving-decode.md) explains the timing
boundaries, configuration correction, and limitations.

Raw responses, generated text, per-round timestamps, logs, prompts, and cached
tuning choices are preserved at [commit `13f8604`](https://github.com/jiazhihao/mpk-apple/tree/13f8604af3e7e337e5d3a0d3ab5e461036f302d9/tools/bench/results/qwen8b-serving-decode-20260929).
They are omitted from the current tree to keep the benchmark change reviewable;
no measurements or summary values were changed.

Restore the exact reproduction inputs from repository history:

```bash
git show 13f8604af3e7e337e5d3a0d3ab5e461036f302d9:tools/bench/results/qwen8b-serving-decode-20260929/prompts.json > /tmp/qwen8b-serving-prompts.json
git show 13f8604af3e7e337e5d3a0d3ab5e461036f302d9:tools/bench/results/qwen8b-serving-decode-20260929/round-cached-choices.json > /tmp/qwen8b-serving-cached-choices.json
```
