#!/usr/bin/env python3
import argparse
import json
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_benchmark_name(name):
    """
    Parsea los nombres generados por BENCHMARK_TEMPLATE:
    - benchmark_max_search_atomic<type>/size_stat
    - benchmark_max_search_reduction<type, false>/size_stat
    - benchmark_max_search_reduction<type, true>/size_stat
    """
    pattern = r"benchmark_max_search_([a-z]+)(?:<([^,>]+)(?:,\s*(true|false))?>)?/(\d+).*(mean|stddev)$"
    match = re.search(pattern, str(name))
    if match:
        method = match.group(1).strip()     # 'atomic' o 'reduction'
        dtype = match.group(2).strip()      # 'int', 'float', 'double', 'long long int'
        is_opt = match.group(3)             # 'true', 'false' o None
        size = int(match.group(4))
        stat_type = match.group(5)
        
        if method == 'atomic':
            algo = 'Atomic'
        elif method == 'reduction':
            algo = 'Reduction_Opt' if is_opt == 'true' else 'Reduction_Standard'
        else:
            algo = method

        return algo, dtype, size, stat_type

    return None, None, None, None

def apply_dense_grid(ax):
    """
    Configura el eje X logarítmico para mostrar marcas intermedias
    y activa una cuadrícula (grid) densa.
    """
    ax.set_xscale('log')
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    
    ax.grid(True, which="major", linestyle="-", alpha=0.6, color='#888888')
    ax.grid(True, which="minor", linestyle=":", alpha=0.4, color='#aaaaaa')

def plot_benchmarks(input_json, output_dir):
    if not os.path.exists(input_json) or os.path.getsize(input_json) == 0:
        print(f"Error: File {input_json} is empty or does not exist.")
        return

    subdirs = {
        'time': os.path.join(output_dir, 'time'),
        'bandwidth': os.path.join(output_dir, 'bandwidth'),
        'memory_usage': os.path.join(output_dir, 'memory_usage'),
        'speedup': os.path.join(output_dir, 'speedup'),
        'dashboard': os.path.join(output_dir, 'dashboard')
    }

    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "benchmarks" not in data:
        print("Error: Key 'benchmarks' not found in JSON file.")
        return

    df = pd.DataFrame(data["benchmarks"])

    parsed_data = df['name'].apply(parse_benchmark_name)
    df['algo'] = [p[0] for p in parsed_data]
    df['dtype'] = [p[1] for p in parsed_data]
    df['size'] = [p[2] for p in parsed_data]
    df['stat_type'] = [p[3] for p in parsed_data]
    
    df = df.dropna(subset=['algo', 'dtype', 'size'])
    
    if df.empty:
        print("Error: No valid benchmarks found in JSON.")
        return

    df['real_time_ms'] = pd.to_numeric(df['real_time'], errors='coerce')

    # Tamaño de los tipos de datos en Bytes
    type_sizes = {'int': 4, 'float': 4, 'double': 8, 'long long int': 8, 'long long': 8}
    df['bytes_per_elem'] = df['dtype'].map(type_sizes).fillna(4)

    # Ancho de banda calculado estrictamente desde el tiempo medido en ms
    df['gb_per_sec'] = (df['size'] * df['bytes_per_elem']) / (df['real_time_ms'] * 1e6)

    # Lectura del contador de memoria nativo de C++
    if 'Memory_MB' in df.columns:
        df['memory_used_mb'] = pd.to_numeric(df['Memory_MB'], errors='coerce')
    else:
        df['memory_used_mb'] = 0.0

    colors = {
        'Atomic': '#e74c3c',             # Rojo
        'Reduction_Standard': '#3498db', # Azul
        'Reduction_Opt': '#2ecc71'       # Verde
    }

    dtypes = df['dtype'].unique()

    for dtype in dtypes:
        df_mean = df[(df['dtype'] == dtype) & (df['stat_type'] == 'mean')]
        df_std = df[(df['dtype'] == dtype) & (df['stat_type'] == 'stddev')]
        
        if df_mean.empty:
            continue

        safe_dtype_name = dtype.replace(' ', '_')

        # ----------------------------------------------------
        # 1. TIEMPO DE EJECUCIÓN CON SOMBRA DE STDDEV (ms)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 6))
        for algo in sorted(df_mean['algo'].unique()):
            df_algo_mean = df_mean[df_mean['algo'] == algo].sort_values('size')
            df_algo_std = df_std[df_std['algo'] == algo].sort_values('size')

            color = colors.get(algo, '#7f8c8d')
            ax.plot(df_algo_mean['size'], df_algo_mean['real_time_ms'], marker='o', linewidth=2, color=color, label=algo)

            if not df_algo_std.empty:
                merged = pd.merge(df_algo_mean, df_algo_std, on='size', suffixes=('_mean', '_std'))
                y_mean = merged['real_time_ms_mean']
                y_std = merged['real_time_ms_std']

                # Sombra mostrando Media ± Desviación Típica
                ax.fill_between(merged['size'], 
                                (y_mean - y_std).clip(lower=1e-6), 
                                y_mean + y_std, 
                                color=color, alpha=0.2)

        ax.set_yscale('log')
        apply_dense_grid(ax)
        ax.set_xlabel('Number of elements ($N$)', fontsize=12)
        ax.set_ylabel('Execution Time (ms) [Log Scale]', fontsize=12)
        ax.set_title(f'Execution Time (Mean ± StdDev) - {dtype}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(subdirs['time'], f'time_{safe_dtype_name}.png'), dpi=300)
        plt.close()

        # ----------------------------------------------------
        # 2. ANCHO DE BANDA EFECTIVO CON SOMBRA (GB/s)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 6))
        for algo in sorted(df_mean['algo'].unique()):
            df_algo_mean = df_mean[df_mean['algo'] == algo].sort_values('size')
            df_algo_std = df_std[df_std['algo'] == algo].sort_values('size')

            color = colors.get(algo, '#7f8c8d')
            ax.plot(df_algo_mean['size'], df_algo_mean['gb_per_sec'], marker='s', linewidth=2, color=color, label=algo)

            if not df_algo_std.empty:
                merged = pd.merge(df_algo_mean, df_algo_std, on='size', suffixes=('_mean', '_std'))
                t_mean = merged['real_time_ms_mean']
                t_std = merged['real_time_ms_std']
                bytes_total = merged['size'] * merged['bytes_per_elem_mean']

                # Propagación de incertidumbre para el Ancho de Banda
                bw_max = (bytes_total) / ((t_mean - t_std).clip(lower=1e-6) * 1e6)
                bw_min = (bytes_total) / ((t_mean + t_std) * 1e6)

                ax.fill_between(merged['size'], bw_min, bw_max, color=color, alpha=0.2)

        apply_dense_grid(ax)
        ax.set_xlabel('Number of elements ($N$)', fontsize=12)
        ax.set_ylabel('Effective Bandwidth (GB/s)', fontsize=12)
        ax.set_title(f'Bandwidth Efficiency (Mean ± StdDev) - {dtype}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(subdirs['bandwidth'], f'bandwidth_{safe_dtype_name}.png'), dpi=300)
        plt.close()

        # ----------------------------------------------------
        # 3. MEMORIA MEDIDA DESDE C++ (MB)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 6))
        for algo in sorted(df_mean['algo'].unique()):
            df_algo_mean = df_mean[df_mean['algo'] == algo].sort_values('size')
            color = colors.get(algo, '#7f8c8d')
            linestyle = '--' if algo == 'Reduction_Opt' else '-'
            
            ax.plot(df_algo_mean['size'], df_algo_mean['memory_used_mb'], marker='d', 
                    linewidth=2.5, linestyle=linestyle, alpha=0.8, color=color, label=algo)

        ax.set_yscale('log')
        apply_dense_grid(ax)
        ax.set_xlabel('Number of elements ($N$)', fontsize=12)
        ax.set_ylabel('Peak VRAM Memory Usage (MB) [Log Scale]', fontsize=12)
        ax.set_title(f'Measured Memory Usage (NVML Peak VRAM) - {dtype}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(subdirs['memory_usage'], f'memory_{safe_dtype_name}.png'), dpi=300)
        plt.close()

        # ----------------------------------------------------
        # 4. SPEEDUP RESPECTO A ATOMIC
        # ----------------------------------------------------
        df_atomic_mean = df_mean[df_mean['algo'] == 'Atomic'].sort_values('size')
        if not df_atomic_mean.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            atomic_times = df_atomic_mean.set_index('size')['real_time_ms']

            for algo in sorted(df_mean['algo'].unique()):
                if algo == 'Atomic':
                    continue
                df_algo_mean = df_mean[df_mean['algo'] == algo].sort_values('size')
                merged = df_algo_mean.set_index('size')['real_time_ms'].to_frame('algo_time').join(atomic_times.to_frame('atomic_time'))
                speedup = merged['atomic_time'] / merged['algo_time']

                color = colors.get(algo, '#7f8c8d')
                ax.plot(speedup.index, speedup.values, marker='^', linewidth=2, color=color, label=f'Speedup {algo} vs Atomic')

            ax.axhline(y=1.0, color='black', linestyle=':', label='Baseline (Atomic)')
            apply_dense_grid(ax)
            ax.set_xlabel('Number of elements ($N$)', fontsize=12)
            ax.set_ylabel('Speedup Factor ($X$ times faster)', fontsize=12)
            ax.set_title(f'Speedup relative to Atomic - {dtype}', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(subdirs['speedup'], f'speedup_{safe_dtype_name}.png'), dpi=300)
            plt.close()

        # ----------------------------------------------------
        # 5. DASHBOARD GENERAL (2x2) CON SOMBRAS
        # ----------------------------------------------------
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Comprehensive Benchmark Analysis - Data Type: {dtype}', fontsize=16, fontweight='bold')

        # Subplot 1: TIEMPO
        for algo in sorted(df_mean['algo'].unique()):
            df_algo_mean = df_mean[df_mean['algo'] == algo].sort_values('size')
            df_algo_std = df_std[df_std['algo'] == algo].sort_values('size')
            color = colors.get(algo)

            axs[0, 0].plot(df_algo_mean['size'], df_algo_mean['real_time_ms'], marker='o', color=color, label=algo)
            if not df_algo_std.empty:
                merged = pd.merge(df_algo_mean, df_algo_std, on='size', suffixes=('_mean', '_std'))
                axs[0, 0].fill_between(merged['size'], 
                                       (merged['real_time_ms_mean'] - merged['real_time_ms_std']).clip(lower=1e-6), 
                                       merged['real_time_ms_mean'] + merged['real_time_ms_std'], 
                                       color=color, alpha=0.2)
        axs[0, 0].set_yscale('log')
        apply_dense_grid(axs[0, 0])
        axs[0, 0].set_title('Execution Time (ms) ± StdDev [Log-Log]')
        axs[0, 0].set_ylabel('Time (ms)')
        axs[0, 0].legend()

        # Subplot 2: ANCHO DE BANDA
        for algo in sorted(df_mean['algo'].unique()):
            df_algo_mean = df_mean[df_mean['algo'] == algo].sort_values('size')
            df_algo_std = df_std[df_std['algo'] == algo].sort_values('size')
            color = colors.get(algo)

            axs[0, 1].plot(df_algo_mean['size'], df_algo_mean['gb_per_sec'], marker='s', color=color, label=algo)
            if not df_algo_std.empty:
                merged = pd.merge(df_algo_mean, df_algo_std, on='size', suffixes=('_mean', '_std'))
                t_m = merged['real_time_ms_mean']
                t_s = merged['real_time_ms_std']
                bytes_tot = merged['size'] * merged['bytes_per_elem_mean']
                axs[0, 1].fill_between(merged['size'], 
                                       bytes_tot / ((t_m + t_s) * 1e6), 
                                       bytes_tot / ((t_m - t_s).clip(lower=1e-6) * 1e6), 
                                       color=color, alpha=0.2)
        apply_dense_grid(axs[0, 1])
        axs[0, 1].set_title('Effective Bandwidth (GB/s) ± StdDev')
        axs[0, 1].set_ylabel('GB/s')
        axs[0, 1].legend()

        # Subplot 3: SPEEDUP
        if not df_atomic_mean.empty:
            for algo in sorted(df_mean['algo'].unique()):
                if algo == 'Atomic': continue
                df_algo_mean = df_mean[df_mean['algo'] == algo].sort_values('size')
                merged = df_algo_mean.set_index('size')['real_time_ms'].to_frame('algo_time').join(atomic_times.to_frame('atomic_time'))
                speedup = merged['atomic_time'] / merged['algo_time']
                axs[1, 0].plot(speedup.index, speedup.values, marker='^', color=colors.get(algo), label=f'{algo} vs Atomic')
            axs[1, 0].axhline(y=1.0, color='black', linestyle=':')
            apply_dense_grid(axs[1, 0])
            axs[1, 0].set_title('Speedup vs Atomic Baseline')
            axs[1, 0].set_ylabel('Speedup Factor')
            axs[1, 0].legend()

        # Subplot 4: MEMORIA
        for algo in sorted(df_mean['algo'].unique()):
            df_algo_mean = df_mean[df_mean['algo'] == algo].sort_values('size')
            linestyle = '--' if algo == 'Reduction_Opt' else '-'
            axs[1, 1].plot(df_algo_mean['size'], df_algo_mean['memory_used_mb'], marker='d', 
                           linestyle=linestyle, alpha=0.8, color=colors.get(algo), label=algo)
        axs[1, 1].set_yscale('log')
        apply_dense_grid(axs[1, 1])
        axs[1, 1].set_title('Measured Memory Usage (MB) [Log Scale]')
        axs[1, 1].set_ylabel('Memory (MB)')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(subdirs['dashboard'], f'dashboard_{safe_dtype_name}.png'), dpi=300)
        plt.close()

        print(f" -> Generated plots with StdDev bands and recalculated bandwidth for: {dtype}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Google Benchmark JSON results.')
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    parser.add_argument('--output-dir', default='./plots', help='Root directory to store structured PNGs')
    args = parser.parse_args()
    
    plot_benchmarks(args.input, args.output_dir)