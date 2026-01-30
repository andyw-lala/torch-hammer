---
name: Performance Issue
about: Report a performance concern or optimization request
title: '[PERF] '
labels: performance
assignees: ''
---

## Summary

Describe the performance issue or optimization opportunity.

## Environment

- **OS**: [e.g., Ubuntu 22.04]
- **Python version**: [e.g., 3.11.5]
- **PyTorch version**: [e.g., 2.2.0]
- **CUDA/ROCm version**: [e.g., CUDA 12.6]
- **GPU model(s)**: [e.g., NVIDIA H100 80GB]
- **Torch Hammer version/commit**: [e.g., v1.0.0]

## Benchmark Configuration

```bash
# The command you ran
./torch-hammer.py --batched-gemm --m 8192 --n 8192 --k 8192 --verbose
```

## Observed Performance

| Metric | Value |
|--------|-------|
| GFLOP/s | |
| Power (W) | |
| GPU Utilization | |
| Memory BW Utilization | |

## Expected Performance

What performance did you expect based on hardware specs or other tools?

## Comparison (if applicable)

If you have comparison data from other tools (e.g., nvidia-smi, rocprof, other benchmarks):

```
Paste comparison data here
```

## Additional Context

Any other relevant details (throttling observed, unusual telemetry, etc.).
