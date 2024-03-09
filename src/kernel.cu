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