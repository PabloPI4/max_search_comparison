#pragma once

#include "../include/max_search_gpu.cuh"





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
    dim3 block_size(BLOCK_SIZE);
    unsigned int n_blocks = array.size()/BLOCK_SIZE;
    if (array.size() % BLOCK_SIZE != 0) {
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





template<typename T>
cuda::std::expected<T, max_search::SearchError> max_search::reduction_max(const thrust::host_vector<T> &array, bool use_opt) {
    /*
        Error checking.
    */
    if (array.size() == 0) {
        return cuda::std::unexpected(max_search::SearchError::EmptyArray);
    }

    /*
        It copies the array to GPU global memory.
    */
    thrust::device_vector<T> array_gpu = array;

    cuda::std::expected<T, max_search::SearchError> result = reduction_max_modifiable(array_gpu, use_opt);
    
    return result;
}



template<typename T>
cuda::std::expected<T, max_search::SearchError> max_search::reduction_max(const thrust::device_vector<T> &array, bool use_opt) {
    /*
        Error checking.
    */
    if (array.size() == 0) {
        return cuda::std::unexpected(max_search::SearchError::EmptyArray);
    }

    /*
        It copies the array to another GPU vector.
    */
    thrust::device_vector<T> array_copy = array;

    cuda::std::expected<T, max_search::SearchError> result = reduction_max_modifiable(array_copy, use_opt);
    
    return result;
}


template<typename T>
cuda::std::expected<T, max_search::SearchError> max_search::reduction_max_modifiable(thrust::device_vector<T> &array, bool use_opt) {
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

    T *raw_vector = thrust::raw_pointer_cast(array.data());

    /*
        Error checking.
    */
    cudaPointerAttributes attributes;
    if (cudaPointerGetAttributes(&attributes, raw_vector) != cudaSuccess) {
        return cuda::std::unexpected(max_search::SearchError::DeviceAllocationFailed);
    }

    if (use_opt) {
        thrust::device_vector<T> array_copy(array.size());

        T *raw_vector_copy = thrust::raw_pointer_cast(array.data());

        /*
            Error checking.
        */
        cudaPointerAttributes attributes;
        if (cudaPointerGetAttributes(&attributes, raw_vector) != cudaSuccess) {
            return cuda::std::unexpected(max_search::SearchError::DeviceAllocationFailed);
        }

        execute_max_search_reduction_opt(raw_vector, raw_vector_copy, array.size());
    }
    else {
        execute_max_search_reduction(raw_vector, array.size());
    }

    /*
        Copying the result before freeing the GPU memory.
    */
    T result = array[0];

    return result;
}