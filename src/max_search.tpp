#pragma once

#include "../include/max_search_gpu.cuh"

#define BLOCK_SIZE 512





template<typename T>
cuda::std::expected<T, max_search::SearchError> max_search::atomic_max(const thrust::host_vector<T> &array) {
    /*
        Error checking.
    */
    if (array.size() == 0) {
        return cuda::std::unexpected(max_search::SearchError::EmptyArray);
    }

    /*
        It copies the array to GPU global memory.
    */
    const thrust::device_vector<T> array_gpu = array;

    cuda::std::expected<T, max_search::SearchError> result = atomic_max(array_gpu);
    
    return result;
}

template<typename T>
cuda::std::expected<T, max_search::SearchError> max_search::atomic_max(const thrust::device_vector<T> &array) {
    /*
        Error checking.
    */
    static_assert(
        std::is_arithmetic_v<T>,
        "ERROR: Type T must be a numeric type.\n"
    );

    if (array.size() == 0) {
        return cuda::std::unexpected(max_search::SearchError::EmptyArray);
    }

    /*
        Accessing raw vector from thrust device vector.
    */
    const T *raw_vector = thrust::raw_pointer_cast(array.data());

    /*
        Error checking.
    */
    cudaPointerAttributes attributes;
    if (cudaPointerGetAttributes(&attributes, raw_vector) != cudaSuccess) {
        return cuda::std::unexpected(max_search::SearchError::DeviceAllocationFailed);
    }

    /*
        It reserves memory for writing the maximum number in GPU.
    */
    T *maximum;
    cudaError_t err = cudaMallocManaged(reinterpret_cast<void**>(&maximum), sizeof(T));

    /*
        Error checking.
    */
    if (err != cudaSuccess) {
        return cuda::std::unexpected(max_search::SearchError::DeviceAllocationFailed);
    }

    /*
        Calculating the number of threads per block and number of blocks
        required.
    */
    dim3 block_size(512);
    unsigned int n_blocks = array.size()/512;
    if (array.size() % 512 != 0) {
        n_blocks++;
    }
    dim3 grid_size(n_blocks);

    execute_max_search_atomic(raw_vector, maximum, grid_size, block_size);

    /*
        Copying the result before freeing the GPU memory.
    */
    T result = *maximum;
    cudaFree(maximum);

    return result;
}