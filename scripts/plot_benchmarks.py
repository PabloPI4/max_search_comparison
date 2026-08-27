#!/usr/bin/env python3
"""
plot_benchmarks.py
 
Generates a set of plots from a Google Benchmark JSON results file
(as produced by --benchmark_format=json), comparing four max-search
algorithm variants:
 
    - Atomic          (benchmark_max_search_atomic<T>)
    - Reduction        (benchmark_max_search_reduction<T, false>)
    - Reduction_opt     (benchmark_max_search_reduction<T, true>)
    - Thrust           (benchmark_max_search_thrust<T>)
 
across four data types: int, long long int, float, double.
 
Output layout (created under --output-dir):
 
    <output-dir>/time/time_<dtype>.png
    <output-dir>/memory_usage/memory_usage_<dtype>.png
    <output-dir>/speedup/speedup_<dtype>.png
    <output-dir>/bandwidth/bandwidth_<dtype>.png
    <output-dir>/dashboard/dashboard_<dtype>.png
 
Usage:
    python3 plot_benchmarks.py --input benchmarks_results.json --output-dir results/
"""
 
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
 
import matplotlib.pyplot as plt
 
 
# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
 
# Maps the C++ template base name used in the benchmark name to the
# human-readable algorithm label requested by the user.
ALGO_BASE_TO_LABEL = {
    "benchmark_max_search_atomic": "Atomic",
    # "reduction" is split further below depending on the boolean flag
    "benchmark_max_search_thrust": "Thrust",
}
 
# Consistent draw order and colors across every plot.
ALGO_ORDER = ["Atomic", "Reduction", "Reduction_opt", "Thrust"]
ALGO_COLORS = {
    "Atomic": "#d62728",        # red
    "Reduction": "#1f77b4",     # blue
    "Reduction_opt": "#2ca02c", # green
    "Thrust": "#9467bd",        # purple
}
 
# Data types we expect to find, in the order we want plots generated.
# Maps the C++ type spelling (as it appears in the JSON) to a
# filesystem-safe key and a display label.
DTYPES = [
    ("int", "int", "int"),
    ("long long int", "long_long_int", "long long int"),
    ("float", "float", "float"),
    ("double", "double", "double"),
]
 
# Size in bytes of each data type, used for the bandwidth computation.
DTYPE_SIZE_BYTES = {
    "int": 4,
    "long long int": 8,
    "float": 4,
    "double": 8,
}
 
# Conversion factors from Google Benchmark's time_unit to seconds.
TIME_UNIT_TO_SECONDS = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
}
 
# Regex to parse a benchmark name, e.g.:
#   benchmark_max_search_reduction<float, true>/1000000/min_time:0.500/repeats:3_mean
BENCHMARK_NAME_RE = re.compile(
    r"^benchmark_max_search_(?P<algo>\w+)<(?P<template_args>[^>]+)>/"
    r"(?P<size>\d+)/.*_(?P<agg>mean|stddev)$"
)
 
 
# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
 
@dataclass
class AlgoSeries:
    sizes: list = field(default_factory=list)
    mean_time_s: list = field(default_factory=list)
    stddev_time_s: list = field(default_factory=list)
    memory_mb: list = field(default_factory=list)
 
 
# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
 
def classify_benchmark(name):
    """
    Parses a benchmark name and returns (algo_label, dtype, size_elements)
    or None if the entry does not correspond to a mean/stddev aggregate of
    one of the four known algorithms.
    """
    m = BENCHMARK_NAME_RE.match(name)
    if not m:
        return None
 
    algo_base = m.group("algo")
    template_args = [a.strip() for a in m.group("template_args").split(",")]
    size = int(m.group("size"))
    agg = m.group("agg")
 
    dtype = template_args[0]
 
    if algo_base == "atomic":
        label = "Atomic"
    elif algo_base == "thrust":
        label = "Thrust"
    elif algo_base == "reduction":
        # Second template argument is the "optimized" boolean flag.
        if len(template_args) < 2:
            return None
        flag = template_args[1].lower()
        if flag == "true":
            label = "Reduction_opt"
        elif flag == "false":
            label = "Reduction"
        else:
            return None
    else:
        return None
 
    return label, dtype, size, agg
 
 
def load_benchmark_data(input_path):
    """
    Loads the Google Benchmark JSON file and returns a nested dict:
 
        data[dtype_cpp][algo_label] = AlgoSeries(...)
 
    where dtype_cpp is the C++ type spelling ("int", "long long int",
    "float", "double") and algo_label is one of
    "Atomic", "Reduction", "Reduction_opt", "Thrust".
    """
    with open(input_path, "r") as f:
        raw = json.load(f)
 
    benchmarks = raw.get("benchmarks", [])
 
    # Intermediate storage keyed by (dtype, algo, size) -> {"mean": ..., "stddev": ..., "memory": ...}
    points = {}
 
    for b in benchmarks:
        classification = classify_benchmark(b.get("name", ""))
        if classification is None:
            continue
        label, dtype, size, agg = classification
 
        time_unit = b.get("time_unit", "ms")
        unit_factor = TIME_UNIT_TO_SECONDS.get(time_unit, 1e-3)
        real_time_s = b.get("real_time", 0.0) * unit_factor
 
        key = (dtype, label, size)
        entry = points.setdefault(key, {"mean": None, "stddev": None, "memory": None})
 
        if agg == "mean":
            entry["mean"] = real_time_s
            entry["memory"] = b.get("Memory_MB", 0.0)
        elif agg == "stddev":
            entry["stddev"] = real_time_s
 
    # Reshape into data[dtype][algo] = AlgoSeries
    data = {}
    for (dtype, label, size), entry in points.items():
        if entry["mean"] is None:
            # No mean aggregate found for this point; skip incomplete entries.
            continue
        data.setdefault(dtype, {}).setdefault(label, AlgoSeries())
        series = data[dtype][label]
        series.sizes.append(size)
        series.mean_time_s.append(entry["mean"])
        series.stddev_time_s.append(entry["stddev"] if entry["stddev"] is not None else 0.0)
        series.memory_mb.append(entry["memory"] if entry["memory"] is not None else 0.0)
 
    # Sort every series by number of elements ascending.
    for dtype, algos in data.items():
        for label, series in algos.items():
            order = sorted(range(len(series.sizes)), key=lambda i: series.sizes[i])
            series.sizes = [series.sizes[i] for i in order]
            series.mean_time_s = [series.mean_time_s[i] for i in order]
            series.stddev_time_s = [series.stddev_time_s[i] for i in order]
            series.memory_mb = [series.memory_mb[i] for i in order]
 
    return data
 
 
# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
 
def style_axes(ax, xlabel="Number of elements", ylabel="", title=""):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
 
 
def draw_time(ax, algos, dtype_label):
    for algo in ALGO_ORDER:
        series = algos.get(algo)
        if series is None or not series.sizes:
            continue
        x = series.sizes
        y_ms = [t * 1e3 for t in series.mean_time_s]
        color = ALGO_COLORS[algo]
        ax.plot(x, y_ms, label=algo, color=color, marker="o", linewidth=1.8, markersize=4)
 
    ax.set_yscale("log")
    style_axes(ax, ylabel="Real time (ms)", title=f"Execution time — {dtype_label}")
 
 
def draw_memory(ax, algos, dtype_label):
    for algo in ALGO_ORDER:
        series = algos.get(algo)
        if series is None or not series.sizes:
            continue
        color = ALGO_COLORS[algo]
        ax.plot(series.sizes, series.memory_mb, label=algo, color=color, marker="o",
                 linewidth=1.8, markersize=4)
 
    style_axes(ax, ylabel="Memory usage (MB)", title=f"Memory usage — {dtype_label}")
 
 
def draw_speedup(ax, algos, dtype_label):
    atomic = algos.get("Atomic")
    if atomic is None or not atomic.sizes:
        ax.set_title(f"Speedup — {dtype_label} (no Atomic baseline found)")
        return
 
    atomic_by_size = dict(zip(atomic.sizes, atomic.mean_time_s))
 
    # Reference line for Atomic itself (speedup == 1).
    all_sizes = sorted(atomic_by_size.keys())
    ax.plot(all_sizes, [1.0] * len(all_sizes), linestyle="--", color=ALGO_COLORS["Atomic"],
             label="Atomic (baseline)", linewidth=1.5)
 
    for algo in ["Reduction", "Reduction_opt", "Thrust"]:
        series = algos.get(algo)
        if series is None or not series.sizes:
            continue
        color = ALGO_COLORS[algo]
        xs, ys = [], []
        for size, t in zip(series.sizes, series.mean_time_s):
            if size in atomic_by_size and t > 0:
                xs.append(size)
                ys.append(atomic_by_size[size] / t)
        if xs:
            ax.plot(xs, ys, label=algo, color=color, marker="o", linewidth=1.8, markersize=4)
 
    style_axes(ax, ylabel="Speedup vs Atomic (x)", title=f"Speedup — {dtype_label}")
 
 
def draw_bandwidth(ax, algos, dtype_label, dtype_cpp):
    elem_bytes = DTYPE_SIZE_BYTES[dtype_cpp]
    for algo in ALGO_ORDER:
        series = algos.get(algo)
        if series is None or not series.sizes:
            continue
        color = ALGO_COLORS[algo]
        xs, ys = [], []
        for size, t in zip(series.sizes, series.mean_time_s):
            if t > 0:
                bytes_per_sec = (size * elem_bytes) / t
                xs.append(size)
                ys.append(bytes_per_sec / 1e9)  # GB/s
        if xs:
            ax.plot(xs, ys, label=algo, color=color, marker="o", linewidth=1.8, markersize=4)
 
    style_axes(ax, ylabel="Bandwidth (GB/s)", title=f"Bandwidth — {dtype_label}")
 
 
# --------------------------------------------------------------------------
# Figure builders (one figure per plot type / dtype)
# --------------------------------------------------------------------------
 
def save_fig(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
 
 
def make_time_plot(algos, dtype_cpp, dtype_key, dtype_label, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    draw_time(ax, algos, dtype_label)
    save_fig(fig, os.path.join(output_dir, "time"), f"time_{dtype_key}.png")
 
 
def make_memory_plot(algos, dtype_cpp, dtype_key, dtype_label, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    draw_memory(ax, algos, dtype_label)
    save_fig(fig, os.path.join(output_dir, "memory_usage"), f"memory_usage_{dtype_key}.png")
 
 
def make_speedup_plot(algos, dtype_cpp, dtype_key, dtype_label, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    draw_speedup(ax, algos, dtype_label)
    save_fig(fig, os.path.join(output_dir, "speedup"), f"speedup_{dtype_key}.png")
 
 
def make_bandwidth_plot(algos, dtype_cpp, dtype_key, dtype_label, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    draw_bandwidth(ax, algos, dtype_label, dtype_cpp)
    save_fig(fig, os.path.join(output_dir, "bandwidth"), f"bandwidth_{dtype_key}.png")
 
 
def make_dashboard_plot(algos, dtype_cpp, dtype_key, dtype_label, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    draw_time(axes[0, 0], algos, dtype_label)
    draw_memory(axes[0, 1], algos, dtype_label)
    draw_speedup(axes[1, 0], algos, dtype_label)
    draw_bandwidth(axes[1, 1], algos, dtype_label, dtype_cpp)
    fig.suptitle(f"Benchmark dashboard — {dtype_label}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, os.path.join(output_dir, "dashboard"), f"dashboard_{dtype_key}.png")
 
 
# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots (time, memory, speedup, bandwidth, dashboard) "
                     "from a Google Benchmark JSON results file."
    )
    parser.add_argument("--input", required=False, default="./build/benchmarks/benchmarks_results.csv", help="Path to the input JSON file")
    parser.add_argument("--output-dir", required=False, default="./plots", help="Directory where plots will be written")
    args = parser.parse_args()
 
    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
 
    print(f"Loading benchmark data from {args.input} ...")
    data = load_benchmark_data(args.input)
 
    if not data:
        print("Error: no recognizable benchmark entries found in the input file.", file=sys.stderr)
        sys.exit(1)
 
    plt.rcParams.update({"font.size": 10})
 
    for dtype_cpp, dtype_key, dtype_label in DTYPES:
        algos = data.get(dtype_cpp)
        if not algos:
            print(f"Warning: no data found for data type '{dtype_cpp}', skipping.", file=sys.stderr)
            continue
 
        print(f"Generating plots for data type: {dtype_label}")
        make_time_plot(algos, dtype_cpp, dtype_key, dtype_label, args.output_dir)
        make_memory_plot(algos, dtype_cpp, dtype_key, dtype_label, args.output_dir)
        make_speedup_plot(algos, dtype_cpp, dtype_key, dtype_label, args.output_dir)
        make_bandwidth_plot(algos, dtype_cpp, dtype_key, dtype_label, args.output_dir)
        make_dashboard_plot(algos, dtype_cpp, dtype_key, dtype_label, args.output_dir)
 
    print("Done.")
 
 
if __name__ == "__main__":
    main()
