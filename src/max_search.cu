#include "../include/max_search_gpu.cuh"
#include <iostream>





template<typename T>
__global__ void atomic_max_gpu(const T *array, T *maximum) {
    unsigned int index_thread = threadIdx.x + blockIdx.x * blockDim.x;

    T element = array[index_thread];

    /*
        Stores the maximum number between element and the one stored in maximum
        address atomically.
    */
    atomicMax(maximum, element);
}



__global__ void atomic_cas_gpu_float(const float *array, float *maximum) {
    unsigned int index_thread = threadIdx.x + blockIdx.x * blockDim.x;

    /*
        It gets the element that will be processed by the thread, the value
        stored in maximum address.
    */
    unsigned int *max_address_as_int = (unsigned int *) maximum;
    float element_float = array[index_thread];
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
    unsigned int index_thread = threadIdx.x + blockIdx.x * blockDim.x;

    /*
        It gets the element that will be processed by the thread, the value
        stored in maximum address.
    */
    unsigned long long int *max_address_as_int = (unsigned long long int *) maximum;
    double element_float = array[index_thread];
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
__global__ void reduction_max_gpu(T *array, unsigned long long int offset,
    unsigned long long int last_thread, unsigned long long int size) {
    
    int index_thread = threadIdx.x + blockIdx.x * blockDim.x;
    int position_access = index_thread * offset;
    T maximum;
    T element;
    __shared__ T temporal_storage[(BLOCK_SIZE >> 5) + 1];

    /*
        Access to the first element of the first max element comparison.
        Each thread will access an element separated offset positions from the
        element accessed by the previous thread.
        Not executed by leftover threads.
    */
    if (index_thread <= last_thread) {
        element = array[position_access];
    }

    maximum = element;

    position_access += (last_thread + 1) * offset;

    /*
        Access to the first element of the first max element comparison on the
        second half of the array.
        Not executed by leftover threads and the last thread if the number of
        elements to be processed is odd.
    */
    if (index_thread < last_thread || (index_thread == last_thread && size % 2 == 0)) {
        element = array[position_access];

        /*
            Max element comparison.
        */
        if (maximum < element) {
            maximum = element;
        }
    }

    /*
        Preparing warps for the warp level reduction.
    */
    int jump = 16;
    unsigned int mask = ~0;
    int last_thread_pos_warp = last_thread % 32;

    /*
        Preparing the last warp for the warp level reduction because it could
        have less than 32 threads.
    */
    if (index_thread >= last_thread - last_thread_pos_warp && last_thread_pos_warp != 31) {
        mask = (1 << (last_thread_pos_warp + 1)) - 1;
        jump = last_thread_pos_warp >> 1;
    }

    /*
        The warp level reduction executes in a loop the following step:
        The first half of the threads of each iteration (32 in the first
        iteraion) gets the local maximum element calculated by another thread
        of the second half and calculates the maximum element between its max
        element and the one obtained. Then, the number of threads is divided by
        2 for the next iteration.
        It is not executed by the leftover threads or the last thread if it is
        the only thread in it's warp.
    */
    if (index_thread > last_thread || (index_thread == last_thread && last_thread_pos_warp == 0)) {
        if (threadIdx.x % 32 == 0) {
            temporal_storage[threadIdx.x>>5] = maximum;
        }
    }
    else {
        for (; jump >= 1; jump >>= 1) {
            element = __shfl_down_sync(mask, maximum, jump);

            if (maximum < element) {
                maximum = element;
            }

            mask >>= jump;
        }

        if (threadIdx.x % 32 == 0) {
            temporal_storage[threadIdx.x>>5] = maximum;
        }
    }

    __syncthreads();

    /*
        Same warp level reduction but only for the first warp of each block
        with the local max elements calculated by each warp.
    */
    int active_threads_in_block = (BLOCK_SIZE >> 5);
    if (BLOCK_SIZE % 32 != 0) {
        active_threads_in_block++;
    }
    jump = BLOCK_SIZE >> 6;
    mask = (1 << active_threads_in_block) - 1;

    /*
        Now the special preparation is executed by the first warp of the first
        block because it could have less than BLOCK_SIZE/32 warps (in this case
        16).
    */
    if (blockIdx.x == gridDim.x - 1) {
        last_thread_pos_warp = (last_thread - blockIdx.x*BLOCK_SIZE + 1) >> 5;

        if ((last_thread - blockIdx.x*BLOCK_SIZE + 1) % 32 != 0) {
            last_thread_pos_warp++;
        }

        if (last_thread_pos_warp != active_threads_in_block) {
            jump = last_thread_pos_warp >> 1;
            mask = (1 << last_thread_pos_warp) - 1;
        }

        active_threads_in_block = last_thread_pos_warp;
    }

    if (threadIdx.x < active_threads_in_block) {
        maximum = temporal_storage[threadIdx.x];
    }

    if (threadIdx.x > 31 || (blockIdx.x == gridDim.x - 1 && active_threads_in_block == 1)) {}
    else {
        for (; jump >= 1; jump >>= 1) {
            element = __shfl_down_sync(mask, maximum, jump);

            if (maximum < element) {
                maximum = element;
            }

            mask >>= jump;
        }
    }

    /*
        Storing the local maximum element calculated by each block.
    */
    if (threadIdx.x == 0) {
        array[blockIdx.x * offset * BLOCK_SIZE*2] = maximum;
    }
}


template<typename T>
__global__ void reduction_max_gpu_opt(T *array, T *array_out, unsigned long long int offset,
    unsigned long long int last_thread, unsigned long long int size) {
    
    int index_thread = threadIdx.x + blockIdx.x * blockDim.x;
    int position_access = index_thread;
    T maximum;
    T element;
    __shared__ T temporal_storage[(BLOCK_SIZE >> 5) + 1];

    /*
        Access to the first element of the first max element comparison.
        In optimised version, accesses are coalesced.
        Not executed by leftover threads.
    */
    if (index_thread <= last_thread) {
        element = array[position_access];
    }

    maximum = element;

    position_access += last_thread + 1;

    /*
        Access to the first element of the first max element comparison on the
        second half of the array.
        Not executed by leftover threads and the last thread if the number of
        elements to be processed is odd.
    */
    if (index_thread < last_thread || (index_thread == last_thread && size % 2 == 0)) {
        element = array[position_access];

        /*
            Max element comparison.
        */
        if (maximum < element) {
            maximum = element;
        }
    }

    /*
        Preparing warps for the warp level reduction.
    */
    int jump = 16;
    unsigned int mask = ~0;
    int last_thread_pos_warp = last_thread % 32;

    /*
        Preparing the last warp for the warp level reduction because it could
        have less than 32 threads.
    */
    if (index_thread >= last_thread - last_thread_pos_warp && last_thread_pos_warp != 31) {
        mask = (1 << (last_thread_pos_warp + 1)) - 1;
        jump = last_thread_pos_warp >> 1;
    }

    /*
        The warp level reduction executes in a loop the following step:
        The first half of the threads of each iteration (32 in the first
        iteraion) gets the local maximum element calculated by another thread
        of the second half and calculates the maximum element between its max
        element and the one obtained. Then, the number of threads is divided by
        2 for the next iteration.
        It is not executed by the leftover threads or the last thread if it is
        the only thread in it's warp.
    */
    if (index_thread > last_thread || (index_thread == last_thread && last_thread_pos_warp == 0)) {
        if (threadIdx.x % 32 == 0) {
            temporal_storage[threadIdx.x>>5] = maximum;
        }
    }
    else {
        for (; jump >= 1; jump >>= 1) {
            element = __shfl_down_sync(mask, maximum, jump);

            if (maximum < element) {
                maximum = element;
            }

            mask >>= jump;
        }

        if (threadIdx.x % 32 == 0) {
            temporal_storage[threadIdx.x>>5] = maximum;
        }
    }

    __syncthreads();

    /*
        Same warp level reduction but only for the first warp of each block
        with the local max elements calculated by each warp.
    */
    int active_threads_in_block = (BLOCK_SIZE >> 5);
    if (BLOCK_SIZE % 32 != 0) {
        active_threads_in_block++;
    }
    jump = BLOCK_SIZE >> 6;
    mask = (1 << active_threads_in_block) - 1;

    /*
        Now the special preparation is executed by the first warp of the first
        block because it could have less than BLOCK_SIZE/32 warps (in this case
        16).
    */
    if (blockIdx.x == gridDim.x - 1) {
        last_thread_pos_warp = (last_thread - blockIdx.x*BLOCK_SIZE + 1) >> 5;

        if ((last_thread - blockIdx.x*BLOCK_SIZE + 1) % 32 != 0) {
            last_thread_pos_warp++;
        }

        if (last_thread_pos_warp != active_threads_in_block) {
            jump = last_thread_pos_warp >> 1;
            mask = (1 << last_thread_pos_warp) - 1;
        }

        active_threads_in_block = last_thread_pos_warp;
    }

    if (threadIdx.x < active_threads_in_block) {
        maximum = temporal_storage[threadIdx.x];
    }

    if (threadIdx.x > 31 || (blockIdx.x == gridDim.x - 1 && active_threads_in_block == 1)) {}
    else {
        for (; jump >= 1; jump >>= 1) {
            element = __shfl_down_sync(mask, maximum, jump);

            if (maximum < element) {
                maximum = element;
            }

            mask >>= jump;
        }
    }

    /*
        Storing the local maximum element calculated by each block in array_out.
    */
    if (threadIdx.x == 0) {
        array_out[blockIdx.x] = maximum;
    }
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





template<typename T>
void execute_max_search_reduction(T *array, unsigned long long int size) {
    unsigned long long int offset = 1;
    unsigned long long int n_elements;
    unsigned long long int n_elements_aux;
    unsigned long long int n_blocks;

    /*
        Calculating the number of threads per block and number of blocks
        required.
    */
    n_elements = size;
    n_elements_aux = n_elements/2;
    if (n_elements % 2 != 0) {
        n_elements_aux++;
    }
    n_blocks = n_elements_aux/BLOCK_SIZE;
    if (n_elements_aux % BLOCK_SIZE != 0) {
        n_blocks++;
    }
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(n_blocks);

    do {
        reduction_max_gpu<<<grid_size, block_size>>>(array, offset, n_elements_aux-1, n_elements);

        /*
            Every block will calculate the maximum number of BLOCK_SIZE*2
            elements of the array.
            The number of threads needed for the next iteration is:
                ⌈n_blocks/2⌉
            The number of blocks needed for the next iteration is:
                ⌈number_threads_next_iter/BLOCK_SIZE⌉
        */
        n_elements = n_blocks;
        n_elements_aux = n_elements/2;
        if (n_elements % 2 != 0) {
            n_elements_aux++;
        }
        n_blocks = n_elements_aux/BLOCK_SIZE;
        if (n_elements_aux % BLOCK_SIZE != 0) {
            n_blocks++;
        }

        grid_size.x = n_blocks;

        offset *= (BLOCK_SIZE*2);
    } while(n_elements > 1);
}



template<typename T>
void execute_max_search_reduction_opt(T *array, T *array_copy, unsigned long long int size) {
    T *array_in = {array};
    T *array_out = {array_copy};
    unsigned long long int offset = 1;
    unsigned long long int n_elements;
    unsigned long long int n_elements_aux;
    unsigned long long int n_blocks;

    /*
        Calculating the number of threads per block and number of blocks
        required.
    */
    n_elements = size;
    n_elements_aux = n_elements/2;
    if (n_elements % 2 != 0) {
        n_elements_aux++;
    }
    n_blocks = n_elements_aux/BLOCK_SIZE;
    if (n_elements_aux % BLOCK_SIZE != 0) {
        n_blocks++;
    }
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(n_blocks);

    do {
        if (n_blocks == 1 && array_in == array) {
            reduction_max_gpu<<<grid_size, block_size>>>(array, offset, n_elements_aux-1, n_elements);
        }
        else {
            reduction_max_gpu_opt<<<grid_size, block_size>>>(array_in, array_out, offset, n_elements_aux-1, n_elements);
        }

        /*
            Every block will calculate the maximum number of BLOCK_SIZE*2
            elements of the array.
            The number of threads needed for the next iteration is:
                ⌈n_blocks/2⌉
            The number of blocks needed for the next iteration is:
                ⌈number_threads_next_iter/BLOCK_SIZE⌉
        */
        n_elements = n_blocks;
        n_elements_aux = n_elements/2;
        if (n_elements % 2 != 0) {
            n_elements_aux++;
        }
        n_blocks = n_elements_aux/BLOCK_SIZE;
        if (n_elements_aux % BLOCK_SIZE != 0) {
            n_blocks++;
        }

        grid_size.x = n_blocks;

        offset *= (BLOCK_SIZE*2);

        std::swap(array_in, array_out);
    } while(n_elements > 1);
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



template __host__ void execute_max_search_reduction<int>(int*, unsigned long long int size);
template __host__ void execute_max_search_reduction<unsigned int>(unsigned int*, unsigned long long int size);
template __host__ void execute_max_search_reduction<long int>(long int*, unsigned long long int size);
template __host__ void execute_max_search_reduction<unsigned long int>(unsigned long int*, unsigned long long int size);
template __host__ void execute_max_search_reduction<long long int>(long long int*, unsigned long long int size);
template __host__ void execute_max_search_reduction<unsigned long long int>(unsigned long long int*, unsigned long long int size);
template __host__ void execute_max_search_reduction<float>(float*, unsigned long long int size);
template __host__ void execute_max_search_reduction<double>(double*, unsigned long long int size);



template __host__ void execute_max_search_reduction_opt<int>(int*, int*, unsigned long long int size);
template __host__ void execute_max_search_reduction_opt<unsigned int>(unsigned int*, unsigned int*, unsigned long long int size);
template __host__ void execute_max_search_reduction_opt<long int>(long int*, long int*, unsigned long long int size);
template __host__ void execute_max_search_reduction_opt<unsigned long int>(unsigned long int*, unsigned long int*, unsigned long long int size);
template __host__ void execute_max_search_reduction_opt<long long int>(long long int*, long long int*, unsigned long long int size);
template __host__ void execute_max_search_reduction_opt<unsigned long long int>(unsigned long long int*, unsigned long long int*, unsigned long long int size);
template __host__ void execute_max_search_reduction_opt<float>(float*, float*, unsigned long long int size);
template __host__ void execute_max_search_reduction_opt<double>(double*, double*, unsigned long long int size);