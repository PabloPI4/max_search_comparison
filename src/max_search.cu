#include "../include/max_search_gpu.cuh"





template<typename T>
__global__ void atomic_max_gpu(const T *array, T *maximum) {
    unsigned int position = threadIdx.x + blockIdx.x * blockDim.x;

    T element = array[position];

    /*
        Stores the maximum number between element and the one stored in maximum
        address atomically.
    */
    atomicMax(maximum, element);
}



__global__ void atomic_cas_gpu_float(const float *array, float *maximum) {
    unsigned int position = threadIdx.x + blockIdx.x * blockDim.x;

    /*
        It gets the element that will be processed by the thread, the value
        stored in maximum address.
    */
    unsigned int *max_address_as_int = (unsigned int *) maximum;
    float element_float = array[position];
    unsigned int old = *maximum;
    unsigned int current_int;

    /*
        While the maximum value after and before executing atomicCAS are not
        the same, it continues executing this loop. It means that another
        thread has modified the value stored in maximum address before this
        thread.
    */
    do {
        current_int = old;

        float current_float = __int_as_float(current_int);

        /*
            It executes the max comparison.
        */
        if (current_float >= element_float) {
            break;
        }

        unsigned int element_int = __float_as_int(element_float);

        /*
            Using atomic max with float types converted to int type.
        */
        old = atomicCAS(max_address_as_int, current_int, element_int);
    } while (old != current_int);
}

__global__ void atomic_cas_gpu_double(const double *array, double *maximum) {
    unsigned int position = threadIdx.x + blockIdx.x * blockDim.x;

    /*
        It gets the element that will be processed by the thread, the value
        stored in maximum address.
    */
    unsigned long long int *max_address_as_int = (unsigned long long int *) maximum;
    double element_float = array[position];
    unsigned long long int old = *maximum;
    unsigned long long int current_int;

    /*
        While the maximum value after and before executing atomicCAS are not
        the same, it continues executing this loop. It means that another
        thread has modified the value stored in maximum address before this
        thread.
    */
    do {
        current_int = old;

        double current_float = __longlong_as_double(current_int);

        /*
            It executes the max comparison.
        */
        if (current_float >= element_float) {
            break;
        }

        unsigned long long int element_int = __double_as_longlong(element_float);

        /*
            Using atomic max with double types converted to long long int type.
        */
        old = atomicCAS(max_address_as_int, current_int, element_int);
    } while (old != current_int);
}





template<typename T>
__host__ void execute_max_search_atomic(const T *array, T *maximum, dim3 grid_size, dim3 block_size) {
    /*
        CUDA does not provide atomicMax for floating point data types.
    */
    if constexpr (std::is_floating_point_v<T>) {
        if constexpr (std::is_same_v<T, float>) {
            atomic_cas_gpu_float<<<grid_size, block_size>>>(array, maximum);
        }
        else {
            atomic_cas_gpu_double<<<grid_size, block_size>>>(array, maximum);
        }
    }
    else {
        /*
            If type is long int or unsigned long int it needs to reinterpret the type.
            In windows, long int will be 32 bits long, and in linux, long int will be
            64 bits long.
        */
        using NativeType = std::conditional_t<
            std::is_signed_v<T>,
            std::conditional_t<sizeof(T) == sizeof(int), int, long long int>,
            std::conditional_t<sizeof(T) == sizeof(unsigned int), unsigned int, unsigned long long int>
        >;

        atomic_max_gpu<NativeType><<<grid_size, block_size>>>(
            reinterpret_cast<const NativeType*>(array),
            reinterpret_cast<NativeType*>(maximum)
        );
    }

    cudaDeviceSynchronize();
}





/*
    Explicit instanciation of template.
*/
template __host__ void execute_max_search_atomic<int>(const int*, int*, dim3, dim3);
template __host__ void execute_max_search_atomic<unsigned int>(const unsigned int*, unsigned int*, dim3, dim3);
template __host__ void execute_max_search_atomic<long int>(const long int*, long int*, dim3, dim3);
template __host__ void execute_max_search_atomic<unsigned long int>(const unsigned long int*, unsigned long int*, dim3, dim3);
template __host__ void execute_max_search_atomic<long long int>(const long long int*, long long int*, dim3, dim3);
template __host__ void execute_max_search_atomic<unsigned long long int>(const unsigned long long int*, unsigned long long int*, dim3, dim3);
template __host__ void execute_max_search_atomic<float>(const float*, float*, dim3, dim3);
template __host__ void execute_max_search_atomic<double>(const double*, double*, dim3, dim3);