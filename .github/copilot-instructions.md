This file gives concise, codebase-specific guidance so AI coding agents can be immediately productive.

Purpose
- **Goal**: Implement, modify, or debug HTL (Hybrid Ternary Logic) adders and the HTL convolution benchmark.
- Work revolves around two layers: an uncertainty/binary layer (`U`) and a balanced-ternary layer (`V`).

Big-picture architecture & dataflow
- **Entry point**: `main.py` runs `run_htl_convolution_benchmark(...)` which loads images, encodes them, calls `convolution(...)`, and decodes results.
- **Encoding / Decoding**: `HTLConvolution.dataset_loader` + `HTLConvolution.helpers` provide `encode_uint8_to_UV(...)` and `decode_Y_numeric_center(...)` used by `main.py` and tests.
- **Convolution split**: `HTLConvolution.htl_convolution.py` implements `U_pass(...)` (uncertain/binary low bits) and `V_pass(...)` (balanced-ternary high bits) and composes them in `convolution(...)`.
- **Adders as plugins**: The `convolution(...)` call accepts two adder callables: `ripple_add` (used by `U_pass`) and `balanced_ternary_add` (used by `V_pass`). These are found under `Adders/` (e.g. `sobocinski_adder.py`, `balanced_ternary_adder.py`).

Important conventions & patterns (do not change without checking callers)
- **LSB-first vectors**: Many adder routines and `U_pass`/`V_pass` expect/produce bit-trit vectors in LSB-first order. Keep this consistent when writing or refactoring adders.
- **Value domain**: Balanced ternary / trits are represented as integers -1, 0, +1 in most modules. Some helper/adder code may use 0 to represent uncertainty; check the specific `Adders/*` file before changing conventions.
- **Fixed widths K1/K2**: `K1` (U bits) and `K2` (V trits) are passed through `convolution(...)` and used by `encode_uint8_to_UV` and `decode_Y_numeric_center`. Avoid changing defaults globally — prefer explicit args in function calls.
- **Carry policy**: `U_pass` resolves carries using `carry_policy` (e.g. `"center"`). Be careful when altering carry resolution logic — it affects MAE results reported by `main.py`.
- **Adder signature expectations**:
  - `ripple_add(vecs: list[list[int]]) -> list[int]` typically accepts a list of LSB-first lists and returns a single LSB-first vector (possibly longer). Example: `sobocinski_ripple(vecs)` in `Adders/sobocinski_adder.py`.
  - `balanced_ternary_add(a: list[int], b: list[int]) -> list[int]` adds two balanced-ternary vectors and returns a vector (LSB-first semantics used across `V_pass`).

Developer workflows & commands
- Run the HTL convolution benchmark quickly:
  - `python main.py` (default runs the full benchmark and shows matplotlib plots).
  - For quick iteration use small parameters: `python -c "from main import run_htl_convolution_benchmark; run_htl_convolution_benchmark(num_images=2, num_kernels=1, uncertain_adder=__import__('Adders.sobocinski_adder', fromlist=['sobocinski_ripple']).sobocinski_ripple)"`
- Dependencies: `numpy`, `matplotlib`. Use the system/virtualenv Python for running; there is no requirements file in-repo — create `requirements.txt` if you add CI.
- Debugging tips:
  - Use `K1=3,K2=3,num_images=1,num_kernels=1` to reproduce and step through `U_pass`/`V_pass` quickly.
  - Inspect `convolution(..., return_info=True)` to get `U_trunc`, `carry_vals`, `carry_mask`, and `V_pre` for unit debugging.

Key files to inspect (examples)
- `main.py` — benchmark runner and example usage (see how `uncertain_adder` is injected).
- `HTLConvolution/htl_convolution.py` — central algorithm (`U_pass`, `V_pass`, `convolution`).
- `HTLConvolution/helpers.py` — encoding helpers like `fit_width`, `int_to_balanced_ternary`.
- `HTLConvolution/dataset_loader.py` — how MNIST/FashionMNIST images are loaded for benchmarks.
- `Adders/*.py` — available adder implementations; copy patterns from `sobocinski_adder.py` and `balanced_ternary_adder.py` when adding new adder implementations.

Practical editing rules for AI agents
- Preserve public function signatures in `htl_convolution.py` and `helpers.py` unless you update all callers (`main.py` and any experiments/Notebooks).
- When adding an adder, include a small unit test snippet in the same file (commented) and ensure it supports LSB-first vectors and padding behavior used by `U_pass` (`pad_to_ext`).
- Avoid global state: pass `K1`, `K2`, and `carry_policy` through function args when experimenting.

When unsure, inspect these examples before making changes
- How `main.py` swaps adders: `run_htl_convolution_benchmark(uncertain_adder=sobocinski_ripple)`
- How carry values are injected into `V` (see `int_to_balanced_ternary` usage inside `convolution`).

If anything here is unclear or you want a different focus (tests, CI, or more examples), say which area and I'll iterate quickly.
