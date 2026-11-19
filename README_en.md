# Paddle Lite · Research Fork

> A **research and study** fork of Paddle Lite. Not an official release, with no stability promise.

This repository is forked from [PaddlePaddle/Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite). The only thing happening here is source reading, hypothesis testing, and small changes proven (or disproven) by reproducible experiments. The bar is "does it explain the problem", not "is it faster".

## What is being explored

### 1. Graph optimization (MIR)

- **Convergence-style pass orchestration**: run a pass group until the graph stops changing; legacy passes run only in the first round so they never re-match nodes that were already deleted. Off by default; enabled at compile time with `LITE_WITH_PASS_MANAGER_V2`.
- **Unified activation fusion**: activation checks gathered into one place, separating "can fuse" from "how to fuse".
- **Elimination and folding**: assign elimination, dynamic-shape recovery, `reshape2`+`cast` folding, identity transpose and layer-norm folding, plus gelu / div-mul / scale-clip-mul pattern fusion.
- **Observability**: `opt` gained `--discarded_passes` so a single pass can be turned off for A/B experiments.

### 2. ARM kernels

- Convolution + activation fusion paths, including how activations that cannot be expressed by a bit flag fall back to a general implementation.
- Boundary conditions in the `sgemv` family: pointer advance inside inline assembly, remainder tails, per-thread buffer partitioning.
- Argument validation at hot math entry points, so out-of-bounds becomes a readable error instead of a random crash.

### 3. Build and toolchain

- Usability fixes for Apple Silicon and Android cross builds.
- Unified ccache, explicit cmake minimum, corrected SDK and compiler probing.

## Method

Changes here aim to be reproducible, attributable (same input, same metric, before and after), and revertible (uncertain changes sit behind a switch that defaults to the old behaviour).

## Build

```bash
./lite/tools/build_android.sh --arch=armv8      # Android
./lite/tools/build_macos.sh arm64               # macOS (Apple Silicon)
./lite/tools/build_macos.sh build_optimize_tool # opt tool
./lite/tools/build_macos.sh --with_testing=ON arm64
```

Built with cmake 3.22. The floor in `CMakeLists.txt` is 3.10, but 3.10.3 does not build and 4.x is unverified. Compiler must support C++11. Output lands in `build.lite.<platform>.<arch>[.<variant>]/`.

## Relation to upstream

- Upstream: https://github.com/PaddlePaddle/Paddle-Lite
- License: Apache-2.0, see [LICENSE](LICENSE)
- Experimental changes here are not guaranteed to land upstream, and interfaces are not guaranteed stable.

## Disclaimer

Research code. It may be restructured at any time and may contain unverified conclusions. Do not rely on it where stability matters.
