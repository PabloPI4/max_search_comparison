# max_search_comparison

## Description
This project aims to compare two different methods for getting the maximum number of an array using the GPU for acceleration.

## Technologies and minimum versions
- CMake 3.18
- C++ 17
- CUDA 12.0
- GoogleTest
- GoogleBenchmark
- Thrust
- Python
- Matplotlib

## Methods
### atomic_max
This method uses the atomicMax function from CUDA.
Each thread calls this function to update the global memory address where the maximum number is stored with its corresponding number of the array, only if it is greater than the currently stored.

### reduction
This method calls the kernel function until the maximum number is calculated.
The kernel function is called with X thread per block and ⌈(N/X)/2⌉ blocks, being N the number of elements of the array. The number of blocks is divided by 2 because every thread calculates the maximum number between two array elements.
When this local maximum is calculated, it enters a loop where every iteration the number of threads will be divided by 2 in every block until there's only one active thread left that contains the maximum number of its block's array portion. Every iteration, the local maximum number per thread is calculated with 2 local maximum numbers of the last iteration.

### reduction_opt
The same method as reduction but the local maximums are stored in another array (using ping-pong buffers), so in the next kernel call, coalescent reads are performed.

## Build and run
### build
- Go to directory max_search_comparison (base)
- Set the configuration of the project:
    cmake -B build
- Compile:
    cmake --build build

### run tests
- Run the tests:
    ./build/tests/max_search_tests

### run benchmarks
- Run the benchmarks:
    ./build/benchmarks/max_search_benchmarks

### replicate benchmarks of Performance Analysis Document
- Explained in document:
    [Performance Analysis Document](./docs/Performance_Analysis_Document.pdf)

## Results
### Analysis
Explained in document:
[Performance Analysis Document](./docs/Performance_Analysis_Document.pdf)

### Conclusion
The correct selection of an algorithm for calculating the maximum number of an array is reduction, which is also the one used by thrust implementation.
As mentioned, the speedup of this algorithm in comparison with the atomic method is very high, and scales with the number of elements of the array.