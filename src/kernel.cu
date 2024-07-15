template<typename T>

__device__ void features_batch_combine(const T *self_output, const T *oppoent_output, T *combined_output, const size_t nlen, const size_t batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen * 2) {
        if (index >= nlen) {
            combined_output[batch_index * nlen * 2 + index] = oppoent_output[batch_index * nlen + index - nlen];
        } else {
            combined_output[batch_index * nlen * 2 + index] = oppoent_output[batch_index * nlen + index];
        }
    }
}

template<typename T>

__device__ void loss_input_transform_to_features(T *self_input, T *oppoent_input, const T *combined_input, const size_t nlen, const size_t batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen * 2) {
        if (index >= nlen) {
            oppoent_input[batch_index * nlen + index - nlen] = combined_input[batch_index * nlen * 2 + index];
        } else {
            self_input[batch_index * nlen + index] = combined_input[batch_index * nlen * 2 + index];
        }
    }
}

template<typename T>

__device__ void forward_transform_features_batch(const size_t *indexes, const size_t *boundaries,
                                                 const T *units, const T *bias, T *output,
                                                 const size_t output_len,
                                                 const size_t batch_size) {
    extern __shared__ char smem[];

    T *sdata_sum = reinterpret_cast<T*>(&smem[0]);
    T *sdata_c = reinterpret_cast<T*>(&smem[BLOCK_SHARED * sizeof(T)]);

    if (blockIdx.x < output_len * batch_size) {
        const size_t batch_index = blockIdx.x / output_len;
        const size_t out_index = blockIdx.x - batch_index * output_len;

        const size_t tid = threadIdx.x;

        const size_t start_index = boundaries[batch_index];
        const size_t end_index = boundaries[batch_index + 1];

        if (tid < end_index - start_index) {
            input_len = end_index - start_index;

            sdata_sum[tid] = 0.0;
            sdata_c[tid] = 0.0;

            T c = 0.0;

            sdata_sum[tid] = units[indexes[start_index + tid] * output_len + out_index];

            /**
             * Kahan summation algorithm
             */
            if (tid < 64 && tid + 64 < input_len) {
                const T y = sdata_sum[tid + 64] - c - sdata_c[tid + 64];
                const T t = sdata_sum[tid] + y;
                c = (t - sdata_sum[tid]) - y;
                sdata_c[tid] = c;
                sdata_sum[tid] = t;
            }
            __syncthreads();

            if (tid < 32 && tid + 32 < input_len) {
                const T y = sdata_sum[tid + 32] - c - sdata_c[tid + 32];
                const T t = sdata_sum[tid] + y;
                c = (t - sdata_sum[tid]) - y;
                sdata_c[tid] = c;
                sdata_sum[tid] = t;
            }
            __syncthreads();

            if (tid < 16 && tid + 16 < input_len) {
                const T y = sdata_sum[tid + 16] - c - sdata_c[tid + 16];
                const T t = sdata_sum[tid] + y;
                c = (t - sdata_sum[tid]) - y;
                sdata_c[tid] = c;
                sdata_sum[tid] = t;
            }
            __syncthreads();

            if (tid < 8 && tid + 8 < input_len) {
                const T y = sdata_sum[tid + 8] - c - sdata_c[tid + 8];
                const T t = sdata_sum[tid] + y;
                c = (t - sdata_sum[tid]) - y;
                sdata_c[tid] = c;
                sdata_sum[tid] = t;
            }
            __syncthreads();

            if (tid < 4 && tid + 4 < input_len) {
                const T y = sdata_sum[tid + 4] - c - sdata_c[tid + 4];
                const T t = sdata_sum[tid] + y;
                c = (t - sdata_sum[tid]) - y;
                sdata_c[tid] = c;
                sdata_sum[tid] = t;
            }
            __syncthreads();

            if (tid < 2 && tid + 2 < input_len) {
                const T y = sdata_sum[tid + 2] - c - sdata_c[tid + 2];
                const T t = sdata_sum[tid] + y;
                c = (t - sdata_sum[tid]) - y;
                sdata_c[tid] = c;
                sdata_sum[tid] = t;
            }
            __syncthreads();

            if (tid < 1 && tid + 1 < input_len) {
                const T y = sdata_sum[tid + 1] - c - sdata_c[tid + 1];
                const T t = sdata_sum[tid] + y;
                c = (t - sdata_sum[tid]) - y;
                sdata_sum[tid] = t;
            }

            if (tid == 0) {
                const T y = bias[out_index] - c;
                const T t = sdata_sum[0] + y;
                output[blockIdx.x] = t;
            }
        }
    }
}

template<typename T>

__device__ void transform_features_gradient_batch(const T *loss, const int *input, T *output,
                                                  const size_t input_len, const size_t output_len,
                                                  const size_t units_size, const size_t batch_size) {
    extern __shared__ char smem[];

    T *sdata_sum = reinterpret_cast<T*>(&smem[0]);

    if (blockIdx.x < units_size) {
        const size_t batch_index = blockIdx.x / input_len;
        const size_t out_index = blockIdx.x - batch_index * input_len;

        const size_t tid = threadIdx.x;
        size_t i = blockIdx.x / output_len;
        size_t j = blockIdx.x - output_len * i;
        size_t k = tid;

        i = i + k * input_len;
        j = j + k * output_len;

        size_t distance = blockDim.x;

        sdata[tid] = (T)0;

        while (k < batch_size && input[i]) {
            sdata[tid] += loss[j];
            k += distance;
            i += distance * input_len;
            j += distance * output_len;
        }
        __syncthreads();

        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();

        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();

        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();

        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();

        if (tid < 32) {
            sdata[tid] += sdata[tid + 32];
        }
        __syncthreads();

        if (tid < 16) {
            sdata[tid] += sdata[tid + 16];
        }
        __syncthreads();

        if (tid < 8) {
            sdata[tid] += sdata[tid + 8];
        }
        __syncthreads();

        if (tid < 4) {
            sdata[tid] += sdata[tid + 4];
        }
        __syncthreads();

        if (tid < 2) {
            sdata[tid] += sdata[tid + 2];
        }
        __syncthreads();

        if (tid < 1) {
            sdata[tid] += sdata[tid + 1];
        }

        if (tid == 0) {
            output[blockIdx.x] = sdata[0];
        }
    }
}

extern "C" {
	__global__ void features_batch_combine_float(const float *self_output, const float *oppoent_output, float *combined_output, const size_t nlen, const size_t batch_size) {
        features_batch_combine(self_output,oppoent_output,combined_output,nlen,batch_size);
    }

	__global__ void features_batch_combine_double(const double *self_output, const double *oppoent_output, double *combined_output, const size_t nlen, const size_t batch_size) {
        features_batch_combine(self_output,oppoent_output,combined_output,nlen,batch_size);
    }

    __global__ void loss_input_transform_to_features_float(float *self_input, float *oppoent_input, const float *combined_input, const size_t nlen, const size_t batch_size) {
        loss_input_transform_to_features(self_input,oppoent_input,combined_input,nlen,batch_size);
    }

    __global__ void loss_input_transform_to_features_double(double *self_input, double *oppoent_input, const double *combined_input, const size_t nlen, const size_t batch_size) {
        loss_input_transform_to_features(self_input,oppoent_input,combined_input,nlen,batch_size);
    }
}