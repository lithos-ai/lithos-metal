# CI

Three lanes. Hosted runners check every push and pull request; bare-metal Macs run what needs a real Apple GPU,
on request for pull requests because a self-hosted machine executes the code under test.

| Lane | Workflow | Runs on | When | Required check |
|---|---|---|---|---|
| 1. Hosted | `ci.yml` | GitHub-hosted Linux and macOS | every push and pull request | `ci-finish` |
| 2. GPU tiers | `mac-gpu.yml` | self-hosted Macs | `/mac-test` on a pull request; every push to `main` | `mac-gpu (<runner>)` |
| 3. Release gate | `release-gate.yml`, `release.yml` | self-hosted Macs, then hosted | `/release-gate`, or dispatched on a release commit; a `v*` tag | `release-gate (<runner>)` |

## Lane 1: hosted

The contract tier on Linux (Python 3.10 and 3.13), the hygiene and extension checks (CONTRIBUTING.md), and a
`macos-26` job that, for Python 3.10, 3.13 and 3.14, builds the Metal module, imports it, collects the GPU tiers and
runs the contract tier on Darwin. Hosted macOS is virtualized: its paravirtual GPU cannot run the kernel tests and its numbers mean nothing.
`ci-finish` passes when every other job passed or was skipped; it is the one check branch protection requires.

## Lane 2: the GPU tiers on bare-metal Macs

`tools/ci/mac_gpu.py` builds the checkout into a new virtual environment under `$LITHOS_CI_ROOT/venv` every run (the
`dev,oracle,serve` extras, mlx-lm, httpx), downloads the checkpoints the oracle tiers read (`tools/ci/models.json`:
public Hub repositories pinned to a commit, every file checked against its SHA-256 on every run, by
`tools/ci/provision.py`) and checks out the DeepSpec reference the drafter oracle imports afresh (a pinned commit): the
tests run as the runner's user and could have rewritten what an earlier run left. It then runs `tests/contract`,
`kernels`, `runtime`, `spec`, `layers` and `models`, one pytest run per tier. A test may skip only for a reason
`ALLOWED_SKIPS` lists: a module that did not build, a missing checkpoint, package or chip profile skips silently on a
developer's machine and fails here. `--shader-validation` (`/mac-test --shader-validation`) runs the Metal tiers under
`MTL_SHADER_VALIDATION=1`, where each validation report fails the test that made it
([test tiers](../CONTRIBUTING.md#test-tiers)); use it for kernel and emitter changes (CLAUDE.md).

A maintainer starts the lane on a pull request by commenting, after reading the change:

```
/mac-test                          every runner in MAC_CI_RUNNERS
/mac-test m5_max_40c m4_pro        named runners
/mac-test --shader-validation
/mac-test <full SHA>               the head you read: required when the author cannot write here
```

`mac-slash.yml` runs from the default branch, checks that the commenter has write access and that the pull request is
open and ready, pins its head commit (refusing a fork's pull request whose head is not the commit named: the author
could push between the comment and the job), marks a pending `mac-gpu (<runner>)` status and dispatches `mac-gpu.yml` on
the default branch with that commit as input. The Mac job checks out the commit without credentials and holds only a
read token; a hosted job sets the final statuses from the uploaded `result.json`, so no token that can write reaches the
Mac. A new `/mac-test` on the same pull request cancels the older run. A dispatch that runs only some tiers reports
`mac-gpu partial (<runner>)`, a status no ruleset requires. No workflow may name a Hugging Face token
(`tools/ci/hygiene.py`): CI reads public checkpoints only.

## Lane 3: the release gate

`release-gate.yml` first builds the bundle a tag of this commit would publish (`tools/release/build_macos.py` on hosted
macOS; `release.yml` publishes these bytes); on each runner `tools/ci/release_gate.py` installs that bundle and the
previous release's published bundle (the newest `v*` tag behind the commit with a published release, checked against its
formula) the way the Homebrew formula does, each in its own environment, then for every case in
`tools/ci/release_cases.json` that fits the machine (the 27B with and without its DSpark drafter, the 8B) serves each
build in turn, ABBA, through `lithos-metal serve` and sends the prompt set greedily, streaming. It fails when an output
differs from the release's (`--expect-output-change`, or `/release-gate --expect-output-change`, allows a deliberate
numerics change) or when decode speed (the tokens streamed after the first text chunk, per second, as both servers count
them per chunk; against a release that predates per-chunk counts decode is not compared, and the status says so) or the
long prompt's time to first token regresses by more than the larger of 2 % and twice the runs' own spread; a spread
above 5 % or a build that is not deterministic is inconclusive. Everything a gate uses lives under
`$LITHOS_CI_ROOT/gate`, which every gate empties first: the checkpoint downloads, and each build's environment and packs
(an earlier gate's candidate ran as the runner's user and could have changed them). Within one gate the candidate could
still alter the baseline's files, so `/release-gate`, like `/mac-test`, is for a maintainer who has read the change.
Each case's numbers and the servers' logs are in the run's artifact. Dispatch it on a release commit (Actions →
release-gate → Run workflow), or comment `/release-gate [RUNNER ...]` on a pull request that claims a speed change.

### Releasing

Bump `pyproject.toml`, dispatch `release-gate` on the release commit (one run for all runners: they gate the bundle it
builds), and tag `vX.Y.Z` within the artifact retention (90 days) once its statuses pass. Each status description names
the baseline and the bundle (`vs vX.Y.Z bundle <digest>:`). `release.yml` runs one release at a time; it refuses a tag
whose commit has no passing `release-gate (<runner>)` status, or one that compared with another release than the newest
behind the commit, takes the gate's bundle from that run (it builds nothing for a tag) and checks its digest and that
every case of `release_cases.json` passed on at least one of that run's runners (the 27B cases need a runner with the
memory for them), smoke-tests it on hosted macOS, waits for a reviewer of the `release` environment, checks the baseline
and the bundle again, publishes the GitHub release, and opens the pull request that updates `lithos-ai/homebrew-tap`
(whose CI installs and tests the formula). Manual runs of `release.yml` build a bundle for review and publish nothing.

## A runner Mac

Any bare-metal Apple silicon Mac on macOS 26 with the Command Line Tools: an office machine or an EC2 Mac
(`mac-m4pro.metal`, `mac-m4max.metal`, `mac-m3ultra.metal`, … on a dedicated host, allocated for at least 24
hours; grow the APFS container to the EBS volume after the first boot). The GPU tiers fit a 24 GB machine; the 27B
release cases need about 32 GB of GPU working set, so a 48 GB machine or larger.

1. `brew install python@3.12`; keep the machine awake (`sudo pmset -a sleep 0 disablesleep 1`); 200 GB free for
   the checkpoints, packs, environments and Hub cache.
2. Settings → Actions → Runners → New self-hosted runner (macOS, ARM64), then in the runner directory
   `./config.sh --url https://github.com/lithos-ai/lithos-metal --token <token> --labels <backend> --unattended`
   and `./svc.sh install && ./svc.sh start`. The label is the chip's backend name under
   `monolith/backends/metal/` (`m5_max_40c`, `m4_pro`, …) or the name a new backend will take.
3. In the runner directory's `.env`:

   ```
   LITHOS_CI_ROOT=/Users/<user>/lithos-ci
   LITHOS_CI_PYTHON=/opt/homebrew/bin/python3.12
   PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
   # optional: a command prefix that serializes GPU work with other users of the machine
   LITHOS_CI_GPU_WRAPPER=/path/to/lock --
   ```

   `LITHOS_CI_ROOT` must be a real directory, not a symlink: the lanes delete and rebuild what is under it.

4. Add the label to the repository variable `MAC_CI_RUNNERS` (a JSON list, e.g. `["m5_max_40c", "m4_pro"]`).

## Repository settings

* Branch ruleset on `main`: pull requests required; required checks `ci-finish`, and `mac-gpu (<runner>)` once the
  runners are stable.
* Actions → General → "Require approval for all external contributors": a fork's workflows run only after a
  maintainer approves them.
* Variable `MAC_CI_RUNNERS`; environment `release` with required reviewers; secret `HOMEBREW_TAP_TOKEN` (a
  fine-grained token with contents and pull requests write on `lithos-ai/homebrew-tap` only).
