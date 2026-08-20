add_executable(max_search_benchmarks)
target_sources(max_search_benchmarks
    PRIVATE
        max_search_benchmarks.cu
)

find_package(benchmark REQUIRED)

target_link_libraries(max_search_benchmarks
    PRIVATE
        max_search_lib
        benchmark::benchmark
        benchmark::benchmark_main
)

target_compile_options(max_search_benchmarks
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
)