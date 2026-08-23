#include <benchmark/benchmark.h>
#include "../include/max_search.hpp"





template <typename T>
void benchmark_max_search_atomic(benchmark::State& state) {
    unsigned long long int size = state.range(0);
    thrust::host_vector<T> array(size);

    for (unsigned long long int i = 0; i < size; i++) {
        array[i] = i;
    }

    thrust::device_vector<T> array_gpu = array;

    for (auto _ : state) {
        auto maximum = max_search::atomic_max(array_gpu);

        if (!maximum.has_value()) {
            state.SkipWithError("Error: device memory not allocated");
            break;
        }
    }

    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(T));
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