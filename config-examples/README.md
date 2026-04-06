# Torch Hammer Configuration Examples

This directory contains example configuration files for running Torch Hammer benchmarks.

## Usage

Run benchmarks with a configuration file:
```bash
./torch-hammer.py --config config-examples/quick-test.yaml
```

Preview what a configuration will run:
```bash
./torch-hammer.py --config config-examples/stress-test.yaml --list-profiles
```

Override config settings via CLI:
```bash
./torch-hammer.py --config config-examples/quick-test.yaml --verbose --all-gpus
```

## Available Configurations

| File | Purpose | Duration |
|------|---------|----------|
| `quick-test.yaml` | Fast sanity check | ~2 minutes |
| `stress-test.yaml` | Comprehensive test suite | ~15-30 minutes |
| `platform-stress.yaml` | Standardized GPU platform screening | ~5-6 minutes per GPU |

## Configuration File Format

Config files support two key naming styles:

### Short generic keys (used in quick-test.yaml, stress-test.yaml)
```yaml
benchmarks:
  - name: batched_gemm
    precision: float32         # Short key
    batch_count: 128           # Short key
    inner_loop: 50             # Short key
```

### Full argparse attribute names (used in platform-stress.yaml)
```yaml
benchmarks:
  - name: batched_gemm
    precision_gemm: float32    # Full attribute name
    batch_count_gemm: 128      # Full attribute name
    inner_loop_batched_gemm: 50  # Full attribute name
```

Both styles work in all config files. Short keys are checked first for backwards compatibility.

### Global settings
```yaml
profile: "Descriptive Name"
description: "What this config does"

global:
  warmup: 10              # Warmup iterations
  verbose: true           # Enable verbose output
  all_gpus: true          # Run on all GPUs
  cpu_affinity: true      # NUMA-aware CPU binding

runtime:
  duration: 60            # Run each benchmark for 60s
  temp_warn_C: 85.0       # Temperature warning threshold
  temp_critical_C: 92.0   # Temperature critical threshold
```

## Creating Custom Configurations

1. Copy an existing config as a starting point
2. Adjust parameters for your use case
3. Use `--stress-test` flag to auto-scale sizes based on GPU memory

## Available Benchmarks

| Name | Description | Key Parameters |
|------|-------------|----------------|
| `batched_gemm` | Matrix multiplication | m, n, k, batch_count |
| `convolution` | 2D convolution | height, width, channels, kernel_size |
| `fft` | 3D FFT | nx, ny, nz |
| `einsum` | Attention pattern | heads, seq_len, d_model |
| `memory_traffic` | Memory bandwidth | memory_size, memory_pattern |
| `heat_equation` | Stencil computation | heat_grid_size, heat_time_steps |
| `schrodinger` | Quantum simulation | schrodinger_grid_size |
| `atomic_contention` | L2 cache stress | atomic_target_size |
| `sparse_mm` | Sparse matrix multiply | sparse_m, sparse_n, sparse_density |

## Precision Options

- `float16` - Half precision
- `bfloat16` - Brain floating point
- `float32` - Single precision (default)
- `float64` - Double precision
- `complex64` - Complex single
- `complex128` - Complex double
