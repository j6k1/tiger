#define BLOCK_SHARED_SMALL 32

template<typename T>

__device__ void forward_transform_features_batch(const size_t *indexes, const size_t *boundaries,
                                                 const T *units, const T *bias, T *output,
                                                 const size_t output_len,
                                                 const size_t batch_size) {
    extern __shared__ char smem[];

    const size_t batch_index = blockIdx.x / output_len;
    size_t start_index = 0;
    size_t end_index = 0;

    if (blockIdx.x < output_len * batch_size) {
        start_index = boundaries[batch_index];
        end_index = boundaries[batch_index + 1];
    }

    if (blockIdx.x < output_len * batch_size) {
        const size_t out_index = blockIdx.x - batch_index * output_len;

        T acc = 0.0;

        const size_t tid = threadIdx.x;

        for (size_t i = tid; i < end_index - start_index; i += 32) {
            acc += units[indexes[start_index + i] * output_len + out_index];
        }

        acc += __shfl_down_sync(0xffffffff,acc,16);
        acc += __shfl_down_sync(0xffffffff,acc,8);
        acc += __shfl_down_sync(0xffffffff,acc,4);
        acc += __shfl_down_sync(0xffffffff,acc,2);
        acc += __shfl_down_sync(0xffffffff,acc,1);

        if (tid == 0) {
            output[blockIdx.x] = acc + bias[out_index];
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