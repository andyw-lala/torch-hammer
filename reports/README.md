# torch-hammer-reporter

Fleet report generator for `torch-hammer` benchmark output.  Produces a CLI
summary (default) and an optional self-contained HTML report.  No dependencies
beyond the Python standard library.

## Usage

```bash
# CLI summary (default — works over SSH, meaningful exit codes)
python hammer_report.py results/
python hammer_report.py results.csv
python hammer_report.py results.json

# CLI summary + HTML report
python hammer_report.py results/ -o report.html

# Filter to a specific benchmark / dtype
python hammer_report.py results.csv --benchmark "Batched GEMM" --dtype float32

# Shell dump from HPC
python hammer_report.py dump.txt --shell-output

# Quiet mode for CI (exit 0 = pass, exit 1 = outliers detected)
python hammer_report.py results/ --quiet --outlier-threshold 10
```

The script **auto-detects** the input format (compact CSV, summary CSV,
JSON, or shell dump).

## Input Formats

### Compact CSV (`--compact`)
The primary format.  One row per (GPU, benchmark, dtype):

```
hostname,gpu,gpu_model,serial,benchmark,dtype,iterations,runtime_s,min,mean,max,unit,power_avg_w,temp_max_c
nid005193,0,NVIDIA GH200 120GB,165412307401,...
```

Pass a single file, or a directory of per-node CSV files.

### Summary CSV (`--summary-csv`)
One row per (GPU, benchmark).  Columns: `test,dtype,gpu,serial,performance,unit,...`

### JSON (`--json-output`)
Full torch-hammer JSON export with `metadata`, `runtime_args`, and `gpus[]`.

### Shell dump
Raw output of `for file in *; do echo "file: $file"; cat $file; done`.
Use `--shell-output` to force this mode, or let auto-detection handle it.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | — | Write HTML report to this path |
| `-b`, `--benchmark` | — | Filter: only benchmarks matching this substring |
| `--dtype` | — | Filter: only this exact dtype |
| `--shell-output` | off | Force shell-dump parse mode |
| `--outlier-threshold` | 15.0 | Deviation % to flag as outlier |
| `--quiet` | off | Suppress CLI summary; exit code only |
| `--no-color` | off | Disable ANSI color output |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No outliers detected |
| 1 | Outliers detected (below-fleet GPUs) |
| 2 | Input error (missing file, no data, empty after filter) |

## Report Contents

### CLI Summary (stderr)
- Fleet overview: nodes, GPUs, GPU model, benchmark count
- Per-benchmark table: fleet mean, CV%, avg power, max temp
- Per-node health: GPU count, test count, avg power, max temp, PASS/WARN
- Outlier list with deviation % from fleet mean
- Fleet verdict: PASS / WARN / FAIL counts

### HTML Report (`-o`)
- Metric cards: nodes, GPUs, benchmarks, outlier count
- **Scale-adaptive charts**: per-node SVG bar chart for ≤50 nodes; SVG histogram
  distribution for >50 nodes (fleet mean bin highlighted).  All charts are
  server-side SVG — **zero external dependencies, fully offline**.
- **Multi-metric panels**: power (W) and temperature (°C) charts alongside
  performance for diagnostics
- Percentile stats (p5, median, p95) for large fleets
- **Sortable tables**: click any column header to sort ascending/descending
- **"vs fleet" column**: signed % deviation from fleet mean per node
- **Truncated tables** for large fleets: bottom 5, outliers, top 5 with
  "Show all rows" toggle to expand
- Outlier section with per-GPU deviation details
- Dark mode via `prefers-color-scheme`
- Colorblind-safe Okabe-Ito palette

## Tests

```bash
pytest tests/test_report.py -v
```

83 tests covering: compact/summary/JSON/shell parsing, multi-benchmark
grouping, outlier detection, HTML XSS safety, CLI exit codes, edge cases,
scale-adaptive rendering, sortable table markup, vs-fleet column, SVG charts,
multi-metric panels, sort JS correctness, and CLI truncation.
