#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Definir el resto de rutas absolutas
BUILD_DIR="$PROJECT_ROOT/build"
BENCHMARK_BIN="$BUILD_DIR/benchmarks/max_search_benchmarks"
OUTPUT_CSV="$BUILD_DIR/benchmarks/benchmarks_results.csv"
PYTHON_SCRIPT="$SCRIPT_DIR/plot_benchmarks.py"
PLOTS_DIR="$PROJECT_ROOT/plots"

if [ ! -f "$BENCHMARK_BIN" ]; then
    echo "Error: executable $BENCHMARK_BIN not found."
    exit 1
fi

echo "==> Executing benchmarks and storing results in: $OUTPUT_CSV..."

"$BENCHMARK_BIN" --benchmark_format=csv --benchmark_out="$OUTPUT_CSV"

echo "==> Generating plots in: $PLOTS_DIR..."

python3 "$PYTHON_SCRIPT" --input "$OUTPUT_CSV" --output-dir "$PLOTS_DIR"

echo "==> Plots have been stored in dir: $PLOTS_DIR"