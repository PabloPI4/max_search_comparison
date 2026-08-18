# max_search_comparison

## Description
This project aims to compare two different methods for getting the maximum number of an array using the GPU for acceleration.

## Technologies
- CMake
- C++
- CUDA
- GoogleTest
- GoogleBenchmark

## Methods
### atomic_max
This method uses the atomicMax function from CUDA.
Each thread calls this function to update the global memory address where the maximum number is stored with its corresponding number of the array, only if it is greater than the currently stored.

### reduction
This method calls the kernel function until the maximum number is calculated.
The kernel function is called with X thread per block and ⌈(N/X)/2⌉ blocks, being N the number of elements of the array. The number of blocks is divided by 2 because every thread calculates the maximum number between two array elements.
When this local maximum is calculated, it enters a loop where every iteration the number of threads will be divided by 2 in every block until there's only one active thread left that contains the maximum number of it's block's array portion. Every iteration, the local maximum number per thread is calculated with 2 local maximum numbers of the last iteration.