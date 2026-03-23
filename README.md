# MLSys Track A - C++ Submission (SystemA baseline port)

This folder contains a self-contained C++ baseline scheduler.

## What it does
- Reads an MLSys scheduling benchmark JSON (see PROBLEM.md for schema).
- Writes a valid schedule JSON (subgraphs, granularities, tensors_to_retain, traversal_orders, subgraph_latencies).
- Uses an anytime strategy:
  1) Write a valid baseline schedule immediately.
  2) Greedily try adjacent fusions and keep improvements.
  3) After each improvement, atomically overwrite the output (kill-safe).

## Folder Structure
mlsys_solution/
├── source/
│   ├── main.cc
│   └── scheduler.cc
|   └── header files
├── third_party/
│   └── hnswlib/
|   └── status.h, statusor.h
└── CMakeLists.txt

## SRS + HNSW
- SRS: we store deterministic embeddings of *accepted* fusion decisions.
- ANN retrieval:
  - If `third_party/hnswlib/hnswlib.h` exists, we use real HNSW (InnerProductSpace on normalized vectors).
  - Otherwise we fall back to brute-force cosine over stored winners.

-To setup hnswlib, run:
  ```git clone https://github.com/nmslib/hnswlib.git third_party/hnswlib ```

This mirrors the Python runner, which tries to import `hnswlib` and falls back when unavailable.

## Build
```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j
```
This produces `build/mlsys`.

## Run
```bash
./mlsys path/to/input.json path/to/output.json
```

## Notes on timeouts
The contest harness kills the process after a benchmark-specific timeout. The timeout is not passed as an argument.
This implementation attempts to infer an upper bound via `RLIMIT_CPU` if set; otherwise it uses a small local budget.
Baseline output is written immediately.

## Adding HNSWlib
To enable HNSW:
- Vendor the C++ hnswlib header into `source/third_party/hnswlib/hnswlib.h`.
- Rebuild.

## Statically linked binary
The contest asks for a statically linked `mlsys` binary on Ubuntu 22.04.
Fully static linking with glibc can be difficult; many teams use a musl toolchain.
Try:
```bash
cmake -DCMAKE_EXE_LINKER_FLAGS='-static -static-libstdc++ -static-libgcc' ..
```

## Files
- `source/scheduler.cc`: algorithm port with extensive comments.
- `source/srs_ann.h`: SRS + optional HNSW wrapper.
- `source/mini_json.h`: tiny JSON parser/writer (no external deps).
- `source/mlsys.h`: structures from starter kit (Apache 2.0).
- `source/third_party/absl/status/*`: minimal stubs to satisfy include.
