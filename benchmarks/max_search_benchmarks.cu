#include <benchmark/benchmark.h>
#include "../include/max_search.hpp"
#include <nvml.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <thrust/execution_policy.h>



//Aux class to measure the device memory usage
class VRAMTracker {
private:
    std::atomic<bool> running{false};
    std::atomic<size_t> max_used_bytes{0};
    std::thread monitor_thread;
    nvmlDevice_t device;
    bool valid{false};

public:
    VRAMTracker(unsigned int device_index = 0) {
        if (nvmlInit() == NVML_SUCCESS) {
            if (nvmlDeviceGetHandleByIndex(device_index, &device) == NVML_SUCCESS) {
                valid = true;
            }
        }
    }

    // Detiene el hilo de monitorización en segundo plano si está activo
    void stop() {
        running = false;
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
    }

    ~VRAMTracker() {
        stop();
        if (valid) {
            nvmlShutdown();
        }
    }

    void start() {
        if (!valid) return;
        max_used_bytes = 0;
        running = true;
        
        monitor_thread = std::thread([this]() {
            while (running) {
                nvmlMemory_t memory;
                if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
                    size_t current_used = memory.used;
                    size_t prev_max = max_used_bytes.load();
                    while (current_used > prev_max && 
                           !max_used_bytes.compare_exchange_weak(prev_max, current_used));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    double stop_and_get_max_mb() {
        if (!valid) return 0.0;
        stop();
        return static_cast<double>(max_used_bytes.load()) / (1024.0 * 1024.0);
    }

    double get_current_mb() {
        if (!valid) return 0.0;
        nvmlMemory_t memory;
        if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
            return static_cast<double>(memory.used) / (1024.0 * 1024.0);
        }
        return 0.0;
    }
};





template <typename T>
void benchmark_max_search_atomic(benchmark::State& state) {
    unsigned long long int size = state.range(0);
    thrust::host_vector<T> array(size);

    for (unsigned long long int i = 0; i < size; i++) {
        array[i] = i;
    }

    VRAMTracker vram_tracker(0);
    double baseline_vram_mb = vram_tracker.get_current_mb();
    vram_tracker.start();

    thrust::device_vector<T> array_gpu = array;

    for (auto _ : state) {
        auto maximum = max_search::atomic_max(array_gpu);

        if (!maximum.has_value()) {
            state.SkipWithError("Error: device memory not allocated");
            break;
        }
    }

    double peak_vram_mb = vram_tracker.stop_and_get_max_mb();

    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(T));
    state.counters["Memory_MB"] = std::max(0.0, peak_vram_mb - baseline_vram_mb);
}



template <typename T, bool useOpt>
void benchmark_max_search_reduction(benchmark::State& state) {
    unsigned long long int size = state.range(0);
    thrust::host_vector<T> array(size);

    for (unsigned long long int i = 0; i < size; i++) {
        array[i] = i;
    }

    VRAMTracker vram_tracker(0);
    double baseline_vram_mb = vram_tracker.get_current_mb();
    vram_tracker.start();

    thrust::device_vector<T> array_gpu = array;

    for (auto _ : state) {
        auto maximum = max_search::reduction_max_modifiable(array_gpu, useOpt);

        if (!maximum.has_value()) {
            state.SkipWithError("Error: device memory not allocated");
            break;
        }
    }

    double peak_vram_mb = vram_tracker.stop_and_get_max_mb();

    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(T));
    state.counters["Memory_MB"] = std::max(0.0, peak_vram_mb - baseline_vram_mb);
}



template <typename T>
void benchmark_max_search_thrust(benchmark::State& state) {
    unsigned long long int size = state.range(0);
    thrust::host_vector<T> array(size);

    for (unsigned long long int i = 0; i < size; i++) {
        array[i] = i;
    }

    VRAMTracker vram_tracker(0);
    double baseline_vram_mb = vram_tracker.get_current_mb();
    vram_tracker.start();

    thrust::device_vector<T> array_gpu = array;

    for (auto _ : state) {
        auto maximum = thrust::reduce(thrust::device, array_gpu.begin(), array_gpu.end(), (T) 0, thrust::maximum<T>());

        benchmark::DoNotOptimize(maximum);
    }

    double peak_vram_mb = vram_tracker.stop_and_get_max_mb();

    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(T));
    state.counters["Memory_MB"] = std::max(0.0, peak_vram_mb - baseline_vram_mb);
}





BENCHMARK_TEMPLATE(benchmark_max_search_atomic, int)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_atomic, long long int)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_atomic, float)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_atomic, double)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);





BENCHMARK_TEMPLATE(benchmark_max_search_reduction, int, false)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_reduction, long long int, false)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_reduction, float, false)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_reduction, double, false)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);





BENCHMARK_TEMPLATE(benchmark_max_search_reduction, int, true)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_reduction, long long int, true)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_reduction, float, true)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_reduction, double, true)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);





BENCHMARK_TEMPLATE(benchmark_max_search_thrust, int)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_thrust, long long int)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_thrust, float)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);



BENCHMARK_TEMPLATE(benchmark_max_search_thrust, double)
    ->RangeMultiplier(10)->Range(1000, 100000000)
    ->MinTime(0.5)
    ->Repetitions(3)
    ->DisplayAggregatesOnly(true)
    ->Unit(benchmark::kMillisecond);