# EdgeInfer

Cross-runtime ML inference benchmark. Takes one model (DS-CNN keyword spotter), runs it through four inference backends, and measures where the time actually goes — per operator, per runtime, down to the memory allocation pattern.

The point isn't "which runtime is fastest" as a headline number. It's understanding *why*: which operators fuse, where quantization helps vs hurts, how each runtime plans memory, and what happens when you strip away the runtime entirely and do it in raw C under embedded constraints.

## What this does

```
Speech Commands dataset (35 keywords)
         |
    DS-CNN (PyTorch)
         |
    +---------+-----------+-----------+------------+
    |         |           |           |            |
  ONNX    TFLite    TensorRT     C engine
 Runtime              (GPU)     (from scratch)
    |         |           |           |
    +--- operator-level profiling ---+
    |         |           |           |
    +--- INT8 quantization ----------+
    |                     |           |
    +--- fusion analysis -+--- manual fusion
    |         |           |           |
    +--- memory timeline visualization -+
                          |
                   ARM cross-compile
                   QEMU Cortex-M4 test
                   256KB RAM constraint
```

### The four runtimes

**ONNX Runtime** — CPU, the baseline everyone uses. Graph optimizations on by default, quantization via onnxruntime quantization tools. This is what "just ship it" looks like.

**TFLite** — the mobile/edge standard. Flatbuffer format, built-in INT8 with representative dataset calibration. Interesting because its quantization pipeline is the most mature and the operator coverage is the most restrictive.

**TensorRT** — GPU (runs on the RTX A2000). The only runtime here that does automatic fusion aggressively. FP16 by default, INT8 with calibration cache. This is the ceiling for inference speed on available hardware.

**Custom C engine** — no framework, no runtime, no graph. You write the forward pass operator by operator in C, manage memory yourself, implement quantization by hand. This is the floor for dependencies and the ceiling for understanding what's actually happening.

### The model

DS-CNN (Depthwise Separable CNN) from the ARM ML zoo / keyword spotting literature. Small enough to fit on a Cortex-M4 (target: 256KB RAM, 1MB flash), complex enough to have interesting operator patterns (depthwise conv, pointwise conv, batch norm, ReLU, pooling, dense).

Input: 49x10 MFCC features from 1-second audio clips.

Why this model: it's the standard embedded keyword spotting architecture, it's well-studied so you can validate your numbers, and it has a good mix of operator types for profiling.

## What you measure

### Per-operator profiling

Not just "inference took 12ms." For each runtime:

- Time per operator type (conv2d, depthwise_conv, batch_norm, relu, pool, dense)
- Percentage of total inference time per operator
- Memory allocated per operator
- Where fusion happened (which operators got merged and what the fused kernel timing is vs unfused)

The output is a breakdown table and a stacked bar chart showing where time goes in each runtime.

### INT8 quantization comparison

Each runtime quantizes differently. You measure:

- Accuracy drop from FP32 baseline (on Speech Commands test set)
- Inference speedup
- Model size reduction
- Per-operator error accumulation (run FP32 and INT8 side by side, measure divergence layer by layer)

The C engine does quantization manually: you implement `quantize_tensor()`, `dequantize_tensor()`, and a quantized matmul. This is where you learn what the frameworks hide.

### Operator fusion analysis

TensorRT fuses automatically. The C engine fuses manually. The others don't fuse much. You compare:

- Which operators TensorRT chose to fuse (extract from the engine log)
- Manual Conv+BN+ReLU fusion in C: fold BN weights into conv weights at export time, fuse the ReLU as a clamp on the output
- Speedup from fusion (fused vs unfused timing on the same code path)
- Numerical difference between fused and unfused (should be zero for BN folding, verify it)

### Memory planning

Each runtime allocates intermediate buffers differently. You build:

- A liveness analysis: for each tensor, when it's first written and last read
- A memory timeline visualization showing buffer lifetimes
- Your own greedy allocator in the C engine that reuses dead buffers
- Comparison: peak memory with naive allocation vs your planner vs what each runtime actually uses

The visualization is a Gantt-style chart with tensors on the y-axis and inference steps on the x-axis, colored by which physical buffer backs each tensor.

### Embedded constraint simulation

The C engine compiles for ARM Cortex-M4 via `arm-none-eabi-gcc` and runs under QEMU:

- Does it fit in 256KB RAM? If not, what's the minimum with the memory planner?
- Does the binary fit in 1MB flash?
- Cycle count on QEMU (not real hardware timing, but relative operator cost)
- Fixed-point MFCC preprocessing in C (no floating point on the target)

## Preprocessing: fixed-point MFCC in C

The embedded target has no FPU, so preprocessing can't use `librosa`. You implement:

- 16-bit PCM input handling
- Fixed-point FFT (Q15 format, radix-2 Cooley-Tukey)
- Mel filterbank application in fixed-point
- Log compression using a lookup table
- DCT for the final MFCC coefficients

Validated against librosa output on the same audio clips: max absolute error per coefficient, plotted as a heatmap across time frames and mel bins.

## Project structure

```
edgeinfer/
├── README.md
├── train/
│   ├── ds_cnn.py              # model definition
│   ├── train.py               # training loop, speech commands
│   └── export.py              # onnx, tflite, tensorrt export
├── runtime/
│   ├── onnx_runner.py         # onnx runtime inference + profiling
│   ├── tflite_runner.py       # tflite inference + profiling
│   ├── tensorrt_runner.py     # tensorrt inference + profiling hooks
│   └── profile.py            # unified profiling interface
├── c_engine/
│   ├── include/
│   │   ├── tensor.h           # tensor struct, memory layout
│   │   ├── ops.h              # operator declarations
│   │   ├── quantize.h         # int8 quantization utils
│   │   └── memory.h           # buffer allocator
│   ├── src/
│   │   ├── ops/
│   │   │   ├── conv2d.c       # direct convolution, no im2col
│   │   │   ├── depthwise.c    # depthwise separable conv
│   │   │   ├── dense.c        # fully connected
│   │   │   ├── pool.c         # average pooling
│   │   │   └── fused.c        # conv+bn+relu fused kernel
│   │   ├── quantize.c         # manual int8 quant/dequant
│   │   ├── memory.c           # liveness analysis + greedy allocator
│   │   ├── mfcc.c             # fixed-point mfcc extraction
│   │   └── forward.c          # the full forward pass, wired together
│   ├── weights/               # exported numpy arrays -> C headers
│   ├── test/
│   │   └── test_ops.c         # operator-level unit tests
│   ├── Makefile               # native + cross-compile targets
│   └── link.ld               # linker script for memory constraints
├── analysis/
│   ├── compare_runtimes.py    # generates the comparison tables
│   ├── fusion_analysis.py     # extracts and compares fusion decisions
│   ├── memory_timeline.py     # liveness + allocation visualization
│   └── quant_error.py         # layer-by-layer quantization error
├── embedded/
│   ├── startup.s              # minimal cortex-m4 startup
│   ├── qemu_run.sh            # qemu invocation
│   └── constraints.py         # checks ram/flash fit
├── data/
│   └── speech_commands.py     # dataset download + preprocessing
├── results/
│   ├── profiling/             # raw timing data, json
│   ├── plots/                 # generated figures
│   └── tables/                # markdown comparison tables
├── tests/
│   ├── test_export.py
│   ├── test_mfcc.py           # librosa vs c implementation
│   └── test_quant.py          # fp32 vs int8 numerical checks
├── Makefile
├── Dockerfile
├── .github/workflows/ci.yml
└── requirements.txt
```

## Development plan

### Phase 1 — model and baselines

Train DS-CNN on Speech Commands to ~95% accuracy. Export to ONNX. Verify inference matches between PyTorch and ONNX Runtime. Set up the profiling harness so every operator is timed individually.

**What you learn**: ONNX export mechanics, operator naming conventions, how PyTorch ops map to ONNX ops (some don't map 1:1 and you'll hit that).

### Phase 2 — runtimes

TFLite conversion from ONNX (via tf). TensorRT engine build from ONNX. Get all three Python runtimes producing identical predictions on the same input.

**What you learn**: the conversion quirks of each runtime (TFLite's quantization-aware restrictions, TensorRT's layer fusion log, operator support gaps).

### Phase 3 — C engine

Write the forward pass in C. Start with FP32, operator by operator. Test each operator against PyTorch output. Wire them into a full forward pass. Verify end-to-end predictions match.

**What you learn**: what a convolution actually does when you can't call `torch.nn.Conv2d`. How depthwise separable convs work at the memory layout level. What batch normalization is when you fold it into weights.

### Phase 4 — quantization

INT8 quantization for all four paths. ONNX Runtime's quantization API, TFLite's representative dataset calibration, TensorRT's INT8 calibration cache, and your own manual quantization in C.

**What you learn**: per-channel vs per-tensor quantization, calibration strategies, where quantization error accumulates and why some layers are more sensitive.

### Phase 5 — fusion and memory

Implement Conv+BN+ReLU fusion in the C engine. Extract TensorRT's fusion decisions. Build the memory liveness analyzer and greedy allocator. Generate the memory timeline visualizations.

**What you learn**: why BN folding is mathematically exact (not an approximation), how liveness analysis works on a dataflow graph, how runtime memory planners trade peak RAM for allocation complexity.

### Phase 6 — embedded

Cross-compile for ARM Cortex-M4. Write the fixed-point MFCC. Run under QEMU. Validate against the Python pipeline.

**What you learn**: cross-compilation toolchain, fixed-point arithmetic (Q15 format, overflow handling), what "no FPU" means in practice, linker scripts and memory regions.

### Phase 7 — analysis and writeup

Generate all comparison tables, plots, and the runtime comparison report. The README opens with results, not architecture.

## Hardware requirements

- **GPU**: any CUDA-capable GPU for TensorRT (developed on RTX A2000 4GB). Not needed for the other three runtimes.
- **RAM**: 8GB+ (32GB available, but the model is tiny — RAM is not the bottleneck)
- **Disk**: ~5GB (Speech Commands dataset + all runtime artifacts)
- **Cross-compiler**: `arm-none-eabi-gcc` (installable via apt)
- **QEMU**: `qemu-system-arm` for Cortex-M4 emulation

No external hardware needed. No dev board. Everything runs on a standard Linux workstation or in Docker.

## Running it

```bash
pip install -r requirements.txt          # python 3.12 or 3.13, linux x86_64
sudo apt install gcc-arm-none-eabi libnewlib-arm-none-eabi qemu-system-arm
make help                                # every target, one line each
```

Full run, in order:

```bash
make data        # download speech commands v2 (2.4 GB) and cache the audio
make train       # train ds-cnn, writes artifacts/ds_cnn.pt
make export      # onnx, tflite, tensorrt engines, c headers, eval + calibration sets
make c-engine    # build the c cli, run the operator tests
make bench       # every runtime this machine can run -> results/profiling/*.json
make qemu        # cortex-m4 build, flash/ram check, instruction counts
make analysis    # tables in results/tables, figures in results/plots
make test        # python tests
```

Nothing after `make export` needs the dataset again: every runner reads the same
evaluation features from `artifacts/`, so all seven rows of the comparison table
are scored on identical inputs.

Without a dataset or a GPU you can still exercise the whole pipeline:

```bash
make smoke-export c-engine test qemu analysis
```

That path is what CI runs, using an untrained model and synthetic features.

### What runs where

| Piece | Needs |
|-------|-------|
| training, export, onnx runtime, c engine | cpu only |
| tflite export and runner | tensorflow (cpu is fine) |
| tensorrt engines and runner | nvidia gpu, driver with cuda 13 support |
| `make qemu` | `arm-none-eabi-gcc`, `qemu-system-arm` 5.2+ (needs the mps2-an386 machine) |
| tflite per-operator timings | `make tools`, which fetches tflite's `benchmark_model` |

## Notes on the implementation

Decisions that are worth knowing before reading the code, mostly places where
reality did not match the plan.

**TensorRT 11 removed implicit quantization.** `IInt8Calibrator` and the whole
calibration-cache workflow are gone, and so are `BuilderFlag.FP16` and
`BuilderFlag.INT8`: every network is strongly typed now, so precision comes from
the tensor types in the ONNX file. FP16 is therefore a cast copy of the graph,
and INT8 is a Q/DQ graph calibrated by ONNX Runtime's `quantize_static`. One
calibration run, symmetric int8 with float biases, feeds TensorRT; the asymmetric
uint8 variant feeds ONNX Runtime's x86 kernels, which prefer it.

**The ONNX export deliberately uses the TorchScript exporter.** The dynamo
exporter's optimizer folds batchnorm into conv before any runtime sees the graph,
which would hide exactly the fusion decisions this project is trying to measure.
`training=PRESERVE` keeps one node per module: conv, batchnorm, relu. Both graphs
are exported, and `fusion_analysis.py` prints the difference, because "your
exporter already fused it" is itself a result.

**Features are normalized outside the model.** MFCC coefficient 0 spans roughly
-630 to +60 dB while the rest sit within a few tens of dB. Quantizing that to one
int8 tensor would spend the whole range on c0 and leave three bits for everything
else, so per-coefficient mean/std normalization happens in preprocessing and the
stats travel in the checkpoint. The fixed-point MFCC applies the same
normalization on the target, as one requantization per coefficient.

**MFCC parameters** are 16 kHz, 512-point FFT, 320 hop, 40 mel bands from 20 Hz
to 4 kHz, 10 coefficients, 49 frames, Slaney mel scale, no `top_db` clipping.
`data/speech_commands.py` is the single source of truth: training, every runtime
and the Q15 tables in `c_engine/weights/mfcc_tables.h` all come from it.

**Peak RAM is not one measurement.** The C engine reports its planner's arena
exactly, TensorRT reports the engine's device memory, TFLite reports
`benchmark_model`'s footprint, and ONNX Runtime has no such API so it reports the
process RSS high-water delta, which includes the runtime itself. The comparison
table prints the source next to the number instead of pretending they match.

**QEMU timing is instruction counts, not cycles.** With `-icount shift=0` every
instruction takes 1 ns of virtual time and the 25 MHz SysTick advances one tick
per 40 instructions, which was calibrated against a loop of known length. That
makes per-operator costs deterministic and reproducible, but they are not
hardware cycles: no wait states, no flash latency, no pipeline effects.

## Key results (format)

The README opens with these tables once results exist:

**Runtime comparison — DS-CNN keyword spotter, Speech Commands v2**

| Runtime | Accuracy (%) | Latency p50 (ms) | Model size (KB) | Peak RAM (KB) | Operators fused |
|---------|-------------|-------------------|-----------------|---------------|-----------------|
| PyTorch (ref) | — | — | — | — | — |
| ONNX Runtime | — | — | — | — | — |
| TFLite | — | — | — | — | — |
| TensorRT FP16 | — | — | — | — | — |
| TensorRT INT8 | — | — | — | — | — |
| C engine FP32 | — | — | — | — | — |
| C engine INT8 | — | — | — | — | — |

**Quantization error accumulation** — per-layer max absolute error, INT8 vs FP32:

(heatmap: layers on x-axis, runtimes on y-axis)

**Memory timeline** — buffer lifetimes during inference:

(gantt chart: one row per intermediate tensor, colored by physical buffer assignment)

**Fusion impact** — Conv+BN+ReLU fused vs unfused:

| Configuration | Latency (ms) | Memory (KB) | Numerical diff |
|--------------|-------------|-------------|----------------|
| Unfused (3 ops) | — | — | — |
| Manual fusion (C) | — | — | — |
| TensorRT auto | — | — | — |

## What you understand after this

- What an inference runtime actually does (graph optimization, memory planning, kernel dispatch) and what it hides from you
- INT8 quantization from both sides: the API and the math
- Operator fusion: why it works, when it's exact, and when it introduces error
- Memory management for inference: liveness, allocation, the peak-RAM tradeoff
- Embedded ML constraints: no heap, no FPU, fixed flash/RAM budgets
- Cross-compilation and hardware emulation
- The real performance/accuracy/size tradeoff triangle that governs every deployment decision

This is the project you explain when someone asks "what happens between `model.export()` and inference on a device."

## Status

Every phase is implemented. The result tables are empty because no training run
has happened yet — they fill in from `make bench` and `make analysis`.

Verified so far on a laptop (WSL, no dataset, untrained 32-channel model, so the
numbers below say "it works", not "it is fast"):

- **C engine** — 24 operator tests pass. INT8 kernels are bit-exact against the
  Python reference in `train/quant.py`; FP32 is within 1.5e-7 of PyTorch
  end to end.
- **Fixed-point MFCC** — within 0.31 dB max, 0.043 dB mean of librosa.
- **ONNX + ONNX Runtime** — outputs match PyTorch to 1.2e-7. ORT fuses 18 nodes
  into 9 kernels, worth 2.9x; BN folding changes logits by 1.2e-7, i.e. float
  rounding only, as predicted.
- **Cortex-M4** — cross-compiles, links inside the budget, runs under QEMU.
  INT8 inference takes 8.3 M instructions against 40.3 M for soft-float FP32.
- **Memory planner** — unfused FP32 needs 236.6 kB with one buffer per tensor
  and 31.2 kB with reuse, in two ping-pong buffers.

Not yet run anywhere: training on the real dataset, the TFLite path (needs
TensorFlow) and TensorRT (needs a Turing or newer GPU; the laptop's Maxwell card
is unsupported by TensorRT 11).
