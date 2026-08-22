#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda/std/expected>

namespace max_search {
    enum class SearchError {
        EmptyArray,
        DeviceAllocationFailed
    };

    /***********************************************************************//**
     * @brief Maximum number search in array using CUDA atomic max from host 
     *  array.
     * 
     * @tparam T 
     * @param array The array where the maximum number will be searched.
     * @return The maximum number of the array. 
     ***************************************************************************/
    template<typename T>
    __host__ cuda::std::expected<T, SearchError> atomic_max(const thrust::host_vector<T> &array);
    
    /***********************************************************************//**
     * @brief Maximum number search in array using CUDA atomic max from device
     *  array.
     * 
     * @tparam T 
     * @param array The array where the maximum number will be searched.
     * @return The maximum number of the array. 
     ***************************************************************************/
    template<typename T>
    __host__ cuda::std::expected<T, SearchError> atomic_max(const thrust::device_vector<T> &array);


    /***********************************************************************//**
     * @brief Maximum number search in array using reduction method with CUDA
     *  from host array.
     * 
     * @details It will create a copy of the array in GPU memory.
     * 
     * @tparam T 
     * @param array The array where the maximum number will be searched.
     * @param use_more_mem If true, it will apply an optimization to reduce the
     *  execution time but duplicating the gpu memory size used for the search.
     * @return The maximum number of the array.
     ***************************************************************************/
    template<typename T>
    __host__ cuda::std::expected<T, SearchError> reduction_max(const thrust::host_vector<T> &array, bool use_more_mem = false);

    /***********************************************************************//**
     * @brief Maximum number search in array using reduction method with CUDA
     *  from device array.
     *
     * @details It will create a copy of the array in GPU memory.
     * 
     * @tparam T 
     * @param array The array where the maximum number will be searched.
     * @param use_more_mem If true, it will apply an optimization to reduce the
     *  execution time but duplicating the gpu memory size used for the search.
     * @return The maximum number of the array.
     ***************************************************************************/
    template<typename T>
    __host__ cuda::std::expected<T, SearchError> reduction_max(const thrust::device_vector<T> &array, bool use_more_mem = false);

    /***********************************************************************//**
     * @brief Maximum number search in array using reduction method with CUDA
     *  from device array.
     *
     * @details It will modify the array.
     * 
     * @tparam T 
     * @param array The array where the maximum number will be searched.
     * @param use_more_mem If true, it will apply an optimization to reduce the
     *  execution time but duplicating the gpu memory size used for the search.
     * @return The maximum number of the array.
     ***************************************************************************/
    template<typename T>
    __host__ cuda::std::expected<T, SearchError> reduction_max_modifiable(thrust::device_vector<T> &array, bool use_more_mem = false);
}

#include "../src/max_search.tpp"