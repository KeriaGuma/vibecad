# External Vectorizer Benchmark

Run raster/scanned drawing vectorizers on the same normalized input and compare
their SVG/DXF outputs.

```bash
cd backend
.venv/bin/python -m pip install -r requirements-vectorizers.txt
.venv/bin/python tools/vectorizer_benchmark.py \
  "../data/demo_data/扫描文件_20260629_091905.pdf" \
  --output data/vectorizer_benchmarks/scan_20260629_091905
```

The script writes:

```text
source_first_page.png
normalized_binary.png
normalized_binary.pbm
<tool>.svg
<tool>.dxf
<tool>_preview.svg
summary.json
report.md
```

Tool availability:

```bash
brew install potrace autotrace
```

`vtracer` is a Python dependency. `potrace` and `autotrace` are external CLI
tools and are skipped automatically when they are not installed.
