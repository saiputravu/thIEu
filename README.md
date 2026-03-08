# thIEu

**Th**e **I**nference **E**ngine (**Thiêu** - _Burn_ in Vietnamese)

A GPU-accelerated inference engine for running LLMs on Apple Silicon, written in Rust with native Metal compute shaders.

### [`Project Roadmap & Planning Board`](https://saimyguy.notion.site/ebd/31607adf7ffe80c48eb8f01d1a1d4fe7)

---

## Overview

thIEu is built from the ground up to run transformer model inference directly on Apple Metal GPUs — no Python, no CUDA, no wrappers. It uses Rust's `objc2-metal` bindings to interface with the Metal API and loads models in the [SafeTensors](https://huggingface.co/docs/safetensors) format.

### What's implemented

- **Metal GPU compute pipeline** — device setup, command queues, compute encoders, and kernel dispatch
- **Metal shader compilation** — custom `.metal` kernels compiled to `.metallib` via a Nix-managed build pipeline
- **SafeTensors model loading** — memory-mapped file access for efficient weight loading from Hugging Face models
- **Layer abstraction** — trait-based architecture (`Layer` trait with `new()` and `forward()`) for composable model components
- **Rotary positional embeddings** — `RotaryEmbeddingLayer` for transformer attention

### Models

Includes git submodules for testing with real model weights:

| Model | Parameters |
|-------|-----------|
| [TinyMistral-248M-v3](https://huggingface.co/M4-ai/TinyMistral-248M-v3) | 248M |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B |

## Project Structure

```
src/
├── main.rs          # Metal device setup, kernel dispatch, model loading
├── metal.rs         # Metal GPU device and kernel utilities
├── model.rs         # Layer trait for neural network components
├── embeddings.rs    # Rotary positional embeddings
└── kernels/
    ├── example.metal    # Metal compute shaders
    └── kernels.metallib # Compiled shader library
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Rust (2024 edition) |
| GPU API | Apple Metal via `objc2-metal` |
| Model format | SafeTensors + `memmap2` |
| Async dispatch | Grand Central Dispatch via `dispatch2` |
| Build environment | Nix flake with Metal shader compilation |

## Getting Started

### Prerequisites

- macOS with Metal-capable GPU
- [Nix](https://nixos.org/) package manager (recommended)

### Build & Run

```sh
# Enter the dev environment (includes Rust toolchain + Metal compiler)
nix develop

# Compile Metal shaders and build the project
cargo build

# Run
cargo run
```

### Without Nix

You'll need `xcrun` (from Xcode Command Line Tools) to compile Metal shaders manually:

```sh
xcrun -sdk macosx metal -c src/kernels/example.metal -o src/kernels/example.air
xcrun -sdk macosx metallib src/kernels/example.air -o src/kernels/kernels.metallib
cargo build && cargo run
```

## Resources

- [Apple Metal GPU Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [MLX Array Documentation](https://ml-explore.github.io/mlx/build/html/index.html)

## License

See [LICENSE](LICENSE) for details.
