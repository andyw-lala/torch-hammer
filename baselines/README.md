# Hardware Baseline Files

This directory contains example hardware baseline files for performance validation.

## What Are Baselines?

Baselines define the expected peak performance of your hardware. Torch Hammer compares measured results against these values to calculate efficiency percentages and flag potential issues.

## Files

| File | Format | Description |
|------|--------|-------------|
| `example.yaml` | YAML | Template with documentation |
| `example.json` | JSON | Alternative format example |

## Usage

```bash
# Run with baseline validation
./torch-hammer.py --batched-gemm --baseline-file baselines/example.yaml

# Disable validation (measure only)
./torch-hammer.py --batched-gemm --no-validation
```

## Creating Your Own Baselines

### Step 1: Find Your GPU Model Name

```bash
# NVIDIA
nvidia-smi --query-gpu=name --format=csv,noheader

# AMD
rocm-smi --showproductname

# Or run torch-hammer and check the output
./torch-hammer.py --batched-gemm --verbose 2>&1 | grep "model"
```

### Step 2: Look Up Specifications

Find your GPU's theoretical peak performance from vendor documentation:
- **fp32_tflops**: Single-precision peak TFLOPS
- **fp64_tflops**: Double-precision peak TFLOPS
- **tf32_tflops**: TensorFloat-32 peak (NVIDIA Ampere+)
- **memory_bandwidth_gbps**: Memory bandwidth in GB/s
- **tdp_watts**: Thermal Design Power

### Step 3: Create Baseline File

**YAML format:**
```yaml
# my_baselines.yaml
"NVIDIA A100-SXM4-80GB":
  fp32_tflops: 19.5
  fp64_tflops: 9.7
  tf32_tflops: 156.0
  memory_bandwidth_gbps: 2039.0
  tdp_watts: 400

"AMD Instinct MI250X":
  fp32_tflops: 47.9
  fp64_tflops: 47.9
  memory_bandwidth_gbps: 3276.8
  tdp_watts: 500
```

**JSON format:**
```json
{
  "NVIDIA A100-SXM4-80GB": {
    "fp32_tflops": 19.5,
    "fp64_tflops": 9.7,
    "tf32_tflops": 156.0,
    "memory_bandwidth_gbps": 2039.0,
    "tdp_watts": 400
  }
}
```

### Step 4: Use Your Baselines

```bash
./torch-hammer.py --batched-gemm --baseline-file my_baselines.yaml
```

## Validation Output

When baselines are loaded, Torch Hammer reports:
- **Efficiency %**: Measured / Expected × 100
- **Warnings**: If efficiency drops below threshold (default 70%)
- **Status**: PASS/WARN based on performance

## Tips

- GPU model names must match exactly (case-sensitive)
- Use vendor spec sheets for accurate values
- Real-world efficiency varies (70-90% is typical for GEMM)
- Memory-bound tests often show lower efficiency than compute-bound
