#define BLOCK_SHARED_SMALL 32

template<typename T>

__device__ void forward_transform_features_batch(const size_t *indexes, const size_t *boundaries,
                                                 const T *units, const T *bias, T *output,
                                                 const size_t output_len,
                                                 const size_t batch_size) {
    extern __shared__ char smem[];

    T *sdata_sum = reinterpret_cast<T*>(&smem[0]);
    T *sdata_c = reinterpret_cast<T*>(&smem[BLOCK_SHARED_SMALL * sizeof(T)]);

    const size_t batch_index = blockIdx.x / output_len;
    size_t start_index = 0;
    size_t end_index = 0;

    if (blockIdx.x < output_len * batch_size) {
        start_index = boundaries[batch_index];
        end_index = boundaries[batch_index + 1];
    }

    if (blockIdx.x < output_len * batch_size) {
        const size_t out_index = blockIdx.x - batch_index * output_len;

        const size_t tid = threadIdx.x;
        const size_t tid_warp = threadIdx.x % 32;

        if (tid < 2) {
            sdata_c[tid] = 0.0;
            sdata_sum[tid] = 0.0;
        }
        __syncthreads();

        T c = 0.0;
        T acc = 0.0;

        if (threadIdx.x < end_index - start_index) {
            acc = units[indexes[start_index + tid] * output_len + out_index];
        }

        /**
         * Kahan summation algorithm
         */
        {
            T dc = 0.0;
            T dacc = 0.0;

            dc = __shfl_down_sync(0xffffffff,c,16);
            dacc = __shfl_down_sync(0xffffffff,acc,16);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,8);
            dacc = __shfl_down_sync(0xffffffff,acc,8);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,4);
            dacc = __shfl_down_sync(0xffffffff,acc,4);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,2);
            dacc = __shfl_down_sync(0xffffffff,acc,2);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,1);
            dacc = __shfl_down_sync(0xffffffff,acc,1);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }
        }

        if (tid_warp == 0) {
            sdata_c[tid / 32] = c;
            sdata_sum[tid / 32] = acc;
        }
        __syncthreads();

        if (tid < 2) {
            c = sdata_c[tid];
            acc = sdata_sum[tid];

            T dc = 0.0;
            T dacc = 0.0;

            dc = __shfl_down_sync(0xffffffff,c,1);
            dacc = __shfl_down_sync(0xffffffff,acc,1);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }
        }

        if (tid == 0) {
            const T y = bias[out_index] - c;
            const T t = acc + y;
            output[blockIdx.x] = t;
        }
    }
}

extern "C" {
    __global__ void forward_transform_features_batch_float(const size_t *indexes, const size_t *boundaries,
                                                     const float *units, const float *bias, float *output,
                                                     const size_t output_len,
                                                     const size_t batch_size) {
        forward_transform_features_batch(indexes,boundaries,units,bias,output,output_len,batch_size);
    }
}