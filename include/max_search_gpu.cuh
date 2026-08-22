#pragma once

/*
    It executes the atomic max search calling atomicMax or atomicCAS depending on the data types.
*/
template<typename T>
__host__ void execute_max_search_atomic(const T *array, T *maximum, dim3 grid_size, dim3 block_size);