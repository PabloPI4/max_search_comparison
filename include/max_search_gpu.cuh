#pragma once

#define BLOCK_SIZE 512

/*
    It executes the atomic max search calling atomicMax or atomicCAS depending on the data types.
*/
template<typename T>
__host__ void execute_max_search_atomic(const T *array, T *maximum, dim3 grid_size, dim3 block_size);





/*
    It executes the reduction max search calling reduction_max without optimization.
*/
template<typename T>
__host__ void execute_max_search_reduction(T *array, unsigned long long int size);

/*
    It executes the reduction max search calling reduction_max with optimization.
*/
template<typename T>
__host__ void execute_max_search_reduction_opt(T *array, T *array_copy, unsigned long long int size);