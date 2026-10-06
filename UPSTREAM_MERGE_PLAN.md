# SharpAI `mlx-swift`: fork patches and how to sync with upstream

This fork tracks `ml-explore/mlx-swift` and adds out-of-core inference support
(SSD expert streaming, TurboKV compression, wired-memory/compile fixes). This
document records **what the fork carries on top of upstream** and **how to merge
upstream without losing it or, just as importantly, without silently reverting
upstream fixes**.

## Layout: plain directories, not submodules

Upstream pins `Source/Cmlx/mlx` and `Source/Cmlx/mlx-c` as git submodules. This
fork vendors them as **plain tracked directories** so the SharpAI patches live in
this repository and a SwiftPM consumer needs nothing but this package.

| Directory | Contents |
|---|---|
| `Source/Cmlx/mlx` | upstream `ml-explore/mlx` **v0.32.3** (`64ea011`) + the patch series below |
| `Source/Cmlx/mlx-c` | upstream `ml-explore/mlx-c` **`ebc88f1`** + the patch series below |

> mlx-c has not caught up with core 0.32.3 upstream (its `main` is `a341b49`, a version bump on top of `ebc88f1`), so this fork carries one adaptation: `mlx_gather_qmm` passes `std::nullopt` for the core's new `global_scale` argument (mlx #4458). The C API does not expose it yet.
| `Source/Cmlx/mlx-generated` | produced by `tools/update-mlx.sh` (never edit by hand) |

`.gitmodules` is intentionally absent. When a merge reports a conflict on
`Source/Cmlx/mlx~upstream_main` / `mlx-c~upstream_main`, that is upstream's
submodule pointer: delete it (`git rm`) and re-vendor the tree instead.

## Carried patches

### `Source/Cmlx/mlx` (on top of upstream mlx)

- **SSD expert streaming**: `mlx/core/moe_stream_op.{h,cpp}`,
  `mlx/backend/metal/ssd_streamer.{h,mm}`, `mlx/backend/metal/kernels/moe_stream.metal`,
  and the matching hooks in `mlx/fast.{h,cpp}`.
- **TurboKV**: `mlx/fast/turbo_quant.h` and the TurboQuant decompression path in
  `mlx/backend/metal/kernels/sdpa_vector.h`.
- **I/O loaders**: `mlx/io/load.{h,cpp}`. Since 0.32.3 upstream's `ParallelFileReader` has its own shared batch pool; the fork keeps that and adds only the iOS guards (`MLX_IO_THREAD_COUNT`, `MLX_IOS_SEQUENTIAL_IO`: sequential `pread()`, one reader thread).
- **`floor_divide` on unsigned integers** (`mlx/ops.cpp`): upstream #4515 (in 0.32.3) made integer
  `floor_divide` correct for negative operands by adding `remainder`, `less`, `not_equal`,
  `logical_and` and `subtract` after the divide. For unsigned dtypes that correction is always zero, so
  the fork returns the plain quotient. mlx-swift-lm's MoE expert sort (`SwitchLayers.gatherSort`) calls
  `order.floorDivide(topK)` on uint32 indices in every layer; with #4515 a streamed-experts decode step
  hit a Metal GPU timeout (found by bisecting 0.32.2..0.32.3). Test:
  `FloorDivideTests`. Drop this when upstream makes the same change.
- **Metal device**: `mlx/backend/metal/device.cpp` carries only quieter Metal compile logging. The
  old "lazily create a command encoder when a stream is first used on a new thread" patch is **not**
  carried: upstream's Swift `StreamPool` creates every stream with `mlx_stream_new_thread_unsafe`
  (global encoder map), so streams already work across threads, and a second lazily-created
  encoder/queue for the same stream index would silently break ordering between threads.

### `Source/Cmlx/mlx-c` (on top of upstream mlx-c)

- `mlx/c/fast.{h,cpp}`: C wrappers for the above: `mlx_fast_turbo_*`, `mlx_fast_streamed_gather_mm`,
  `mlx_fast_prefault`, `mlx_fast_pread_into`, `mlx_fast_pread_into_offset`,
  `mlx_fast_submit_prefetch`, `mlx_fast_set_prefetch_enabled`, `mlx_ssd_metrics_snapshot`
  (+ the PAPPS async prefetch pool). Every wrapper must run on the stream it is given
  (`mlx_stream_get_(s)`), not just its device: `mlx_fast_streamed_gather_mm` forwards the stream, which
  is what keeps it correct when Swift evaluates the graph on another thread
  (`SSDStreamingTests.testStreamedGatherMMEvaluatesOnAnotherThread`). The prefetch cache's `try_take`
  only hands back an entry whose stored size equals the requested length.

### Swift

- `Source/MLX/MLXFast.swift`: Swift bindings for the C wrappers above (kept alongside upstream's
  `crossEntropy`).
- `Source/MLX/Transforms+Compile.swift`: `compile` honors a task-scoped default device
  (`Device.withDefaultDevice`) when tracing, so a graph traced on the GPU is not reused on a CPU
  stream (`TransformTests.testCompileHonorsScopedDefaultDevice`).
- `Tests/MLXTests/ForkProtectionTests.swift`: fails if a merge erases `moe_stream.metal` or
  `threadpool.h`.
- `Tests/MLXTests/SSDStreamingTests.swift`: exercises `streamedGatherMM`, `preadInto` and
  `preadIntoOffset` against a real safetensors file, including evaluating on a different thread than
  the one that built the graph.

### Build configuration

- `Package.swift` stays on `swift-tools-version: 5.12`. Upstream's manifest uses
  `6.3;(experimentalCGen)` only to wire the Linux CUDA plugin (`CudaBuild`/`encuda`); Xcode 26.3 and
  our CI cannot parse it, and this fork ships no CUDA backend, so the CUDA plugin/targets are
  stubbed to empty lists (`Source/Encuda`, `Plugins/CudaBuild` stay in the tree, unused).
- `cxxLanguageStandard: .gnucxx20`.

## Merging upstream

1. `git fetch upstream` and branch from `main`; **use a real merge**
   (`git merge --no-ff upstream/main`), not cherry-picks and not `git checkout upstream/main -- .`.
2. Resolve conflicts by hand. For every file the fork touched, check the *whole* diff against
   upstream afterwards (`git diff upstream/main -- <path>`): the only differences left should be the
   carried patches listed above. Anything else is an accidental revert. (A past sync commit,
   `877d992`, silently reverted upstream fixes #356, #358, #363, #367, #369 and #471 this way.)
3. Re-vendor `mlx` and `mlx-c` from the upstream tags/commits that upstream's submodules point to,
   then re-apply the carried patches (see above; the patch series is small and each commit message
   says what it is for). Do not push anything to `SharpAI/mlx` / `SharpAI/mlx-c` for this.
4. Regenerate JIT sources: `./tools/update-mlx.sh`. Afterwards `Source/Cmlx/mlx-generated` should equal
   upstream's except for the TurboQuant `sdpa_vector.h`. Keep `moe_stream.metal`.
5. Build and test: `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS'
   -skipPackagePluginValidation -parallel-testing-enabled NO`, then build and test
   `mlx-swift-lm` and `SwiftLM` against it (TurboKV, `--stream-experts`, SSD `preadInto`).
