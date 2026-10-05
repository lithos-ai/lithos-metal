# M5 Max experiment artifacts

Selected executable recipes live under
[`monolith/backends/metal/m5_max_40c/recipes`](../../monolith/backends/metal/m5_max_40c/recipes/).
Raw measurements, sweep configurations, logs and generated figures are excluded
from the current source tree and retained at commit
`11a1f02e8632ebc75dcfbff48c89e8ea0900947a`:

- [Target-kernel and backend evidence](https://github.com/jiazhihao/mpk-apple/tree/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/)
- [DSpark evidence](https://github.com/jiazhihao/mpk-apple/tree/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-dspark/)
- [Figures](https://github.com/jiazhihao/mpk-apple/tree/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/docs/research/figures/)
- [Hardware probes](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/probes/results/Apple-M5-Max_40c_macOS26.5.1_20261001-130937.txt)

Restore the two result directories locally before running historical benchmark
commands or plots that read their inputs. The restored paths are ignored by Git:

```sh
git archive 11a1f02e8632ebc75dcfbff48c89e8ea0900947a \
  tools/bench/results/m5max-27b-n7 \
  tools/bench/results/m5max-27b-dspark | tar -x
```

For a shallow clone, fetch that commit first with
`git fetch origin 11a1f02e8632ebc75dcfbff48c89e8ea0900947a`.
