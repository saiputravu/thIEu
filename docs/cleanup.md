# Cleanup Changes

| # | File(s) | Change | Reason |
|---|---------|--------|--------|
| 1 | `metal.rs` | Removed `use crate::metal;` self-import, inlined `setup_device()` into `new()` | Eliminated circular self-reference; one-liner with no other callers |
| 2 | `metal.rs` | Added `impl std::error::Error for MetalError` with `source()` | Conforms to Rust error conventions, enables `?` with `Box<dyn Error>` |
| 3 | `metal.rs` | Removed unused `CommitError` variant from `MetalError` | Dead code; never constructed anywhere in the codebase |
| 4 | `metal.rs` | `map_or(self.metal4_supported, \|x\| x)` → `unwrap_or(self.metal4_supported)` | Clearer intent for simple Option unwrapping |
| 5 | `metal.rs` | Moved `new_tensor_descriptor()` from free function to `MetalGPU` method; removed bare `return` | Better encapsulation; expression-style returns |
| 6 | `metal.rs` | Converted `//` comments to `///` doc comments on all public items; resolved all "Comment this function" TODOs | Enables `cargo doc` generation; self-documenting API |
| 7 | `metal.rs`, `main.rs` | Renamed `new_metal_gpu()` → `new()` | Idiomatic Rust constructor naming |
| 8 | `metal.rs`, `main.rs` | Changed `&String` → `&str` on all public methods | More general API, avoids unnecessary allocation at call sites |
| 9 | `embeddings.rs` | Consolidated `use model::Layer` + `use crate::model` → `use crate::model::Layer` | Removed redundant import |
| 10 | `model.rs`, `embeddings.rs` | Changed `Layer::new()` to return `-> Self where Self: Sized` | Makes trait constructor actually implementable with correct return type |
