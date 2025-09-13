#define BLOCK_SHARED_SMALL 32

template<typename T>

__device__ void warp_reduce(volatile T *sdata, int tid) {
    sdata[tid] += sdata[tid+32];
    sdata[tid] += sdata[tid+16];
    sdata[tid] += sdata[tid+8];
    sdata[tid] += sdata[tid+4];
    sdata[tid] += sdata[tid+2];
    sdata[tid] += sdata[tid+1];
}

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

template<typename T>

__device__ void transform_features_gradient_batch(const T *loss,
                                                  const size_t *indexes, const size_t *boundaries,
                                                  T *output,
                                                  const size_t input_len,
                                                  const size_t output_len,
                                                  const size_t batch_size) {
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    const size_t tid = threadIdx.x;
    const size_t unit_index = blockIdx.x * blockIdx.y;
    const size_t input_index = blockIdx.x;
    const size_t out_index = blockIdx.y;
    const size_t input_index_index = blockIdx.z;

    const size_t end = (batch_size + blockDim.x - 1) / blockDim.x * blockDim.x;

    T g = (T)0;

    for (size_t batch_index = tid; batch_index < end; batch_index += blockDim.x) {
        sdata[tid] = (T)0;
        __syncthreads();

        size_t start_index = 0;
        size_t end_index = 0;

        if (batch_index < batch_size) {
            start_index = boundaries[batch_index];
            end_index = boundaries[batch_index+1];
        }

        int skip = batch_index >= batch_size ||
                   input_index >= input_len ||
                   out_index >= output_len ||
                   end_index - start_index <= input_index_index ||
                   indexes[start_index + input_index_index] != input_index;

        if (!skip) {
            sdata[tid] += loss[batch_index * output_len + out_index];
        }
        __syncthreads();

        if (!skip && blockDim.x >= 1024 && tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();

        if (!skip && blockDim.x >= 512 && tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();

        if (!skip && blockDim.x >= 256 && tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();

        if (!skip && blockDim.x >= 128 && tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();

        T acc = (T)0;

        if (!skip && tid < 32) {
            acc += __shfl_down_sync(0xffffffff,acc,16);
            acc += __shfl_down_sync(0xffffffff,acc,8);
            acc += __shfl_down_sync(0xffffffff,acc,4);
            acc += __shfl_down_sync(0xffffffff,acc,2);
            acc += __shfl_down_sync(0xffffffff,acc,1);
        }

        g += acc;
    }

    if (tid == 0) {
        output[unit_index] += g;
    }
}

extern "C" {
    __global__ void forward_transform_features_batch_float(const size_t *indexes, const size_t *boundaries,
                                                     const float *units, const float *bias, float *output,
                                                     const size_t output_len,
                                                     const size_t batch_size) {
        forward_transform_features_batch(indexes,boundaries,units,bias,output,output_len,batch_size);
    }

    __global__ void transform_features_gradient_batch_float(const float *loss,
                                                           const size_t *indexes, const size_t *boundaries,
                                                           float *output,
                                                           const size_t input_len,
                                                           const size_t output_len,
                                                           const size_t batch_size) {
        transform_features_gradient_batch(loss,indexes,boundaries,output,input_len,output_len,batch_size);
    }
}