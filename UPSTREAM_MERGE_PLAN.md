# SharpAI `mlx-swift` Custom Patches & Upstream Merge Guide

This document captures the divergence between the `SharpAI/mlx-swift` custom fork and the official `ml-explore/mlx-swift` upstream repository. **Do not overwrite these commits or configurations** during an upstream merge unless Apple officially merges the underlying features.

These patches enable out-of-core model inference, specifically for SSD Flash Streaming, by relying on custom bindings and our dedicated `SharpAI/mlx` and `SharpAI/mlx-c` forks.

### 1. Speculative Decoding
_Note: Core speculative decoding logic (Drafting, KV Cache routing) primarily exists natively in `SwiftBuddy`/`mlx-swift-lm`. Modifications inside `mlx-swift` strictly pertain to the underlying memory/cache alignment requirements (like bounds limit resolutions) rather than the decoding algorithms themselves._

### 2. SSD Streaming & Out-of-core Patches
Our core optimizations extend the native Engine bindings to allow pulling model matrices from disk to GPU memory directly.
- `6d3a11f` / `9e10176`: **SSD thread-pooling** - Implements background thread-pools in the wrappers with dynamic API toggles to manage off-main-thread I/O operations.
- `14f267b`: **GLM-5.1 compatibility limit fix** - Removes obsolete buffer bounds checks in the `ssd_streamer` that originally crashed multi-expert GLM instances.
- `8540be8` / `877d992`: **C-API extensions** - Restores modifications to the Swift bridge and `fast.h` to explicitly surface C++ streaming kernels (`moe_stream_op`).
- `b45b31c`: **Gemma 4 Apple Silicon array mapping** - Customized Metal kernels optimized for Gemma 4 memory configurations.

### 3. Metal Code Generation & JIT Extensions
Because Swift package compilation automatically generates C++ metal headers, the JIT builder was intercepted to include our custom kernel `.metal` files.
- `61590a5`: Adds the custom `moe_stream.metal` processing logic directly into the codegen workflow.
- `7bb740b`: Tracks the SharpAI custom metal kernels alongside Apple originals so the JIT builder doesn't ignore them.
- `cbbf55c`: Enforces that the `fast.h` method signatures exactly match the upstreamed `fast.cpp` limits after code generation.
- `441ef64`: Adds `steel_conv_3d` C++ string targeting internally for the `Cmlx` build step.

### 4. Build Configuration & Submodule Pointers
The `Package.swift` and internal git components have been locked to point to our ecosystem. Overwriting these with Apple's pointers will break the compiler.
- `2f60e7b`: Re-points Xcode and SPM to `SharpAI/mlx` and `SharpAI/mlx-c` dedicated GitHub forks instead of `ml-explore`.
- `72daf6c` / `d0d852f`: Retains specific submodule inline configurations holding metadata/device streams specifically built to support the SharpAI metal optimizations.
- `6139f94` / `41c8c4f`: Explicit `mlx-c` pointer bumps required to maintain compilation signatures for `fftshift` and `Shape`.
- `a4b0d27`: Manual override bumping the `cxxLanguageStandard` parameter directly to `.gnucxx20` to guarantee C++ compatibility parity.

---
### When merging upstream...
1. Execute `git fetch upstream` (where upstream is `https://github.com/ml-explore/mlx-swift.git`).
2. Do **not** use `--strategy-option=theirs` blindly! 
3. Any conflicts in `Package.swift` (submodule references), `Cmlx/`, or `Metal/` codegen generators MUST favor the **SharpAI local implementation (HEAD)** to prevent kernel erasure.
