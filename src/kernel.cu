#include<cmath>
#include<mma.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>

using namespace nvcuda;

static __device__ half _to_half(float x) {
    return __float2half(x);
}

static __device__ half _to_half(double x) {
    return __double2half(x);
}

static __device__ size_t calc_index(size_t row, size_t col, size_t leading_dimension) {
    return row * leading_dimension + col;
}

#define BLOCK_SHARED_SMALL 32
#define TILE_SIZE 16
#define TILE_SIZE_2D 256
#define BSEARCH_PART_LEN 10
#define HALFKP_ACTIVE_INPUTS 40

template<typename T>

__device__ void forward_transform_features_batch(const size_t *indexes, const size_t *boundaries,
                                                 const T *units, const T *bias, T *output,
                                                 const size_t output_len,
                                                 const size_t batch_size) {
    extern __shared__ char smem[];

    const size_t index = blockIdx.x;

    const size_t batch_index = index / output_len;
    size_t start_index = 0;
    size_t end_index = 0;

    if (batch_index < batch_size) {
        start_index = boundaries[batch_index];
        end_index = boundaries[batch_index + 1];

        const size_t out_index = index - batch_index * output_len;

        T acc = 0.0;

        const size_t tid = threadIdx.x;

        /*
        for (size_t i = tid; i < end_index - start_index; i += 32) {
            acc += units[indexes[start_index + i] * output_len + out_index];
        }

        acc += __shfl_down_sync(0xffffffff,acc,16);
        acc += __shfl_down_sync(0xffffffff,acc,8);
        acc += __shfl_down_sync(0xffffffff,acc,4);
        acc += __shfl_down_sync(0xffffffff,acc,2);
        acc += __shfl_down_sync(0xffffffff,acc,1);
        */

        if (tid == 0) {
            T acc = 0.0;

            for (size_t i = 0; i < end_index - start_index; ++i) {
                size_t idx_pos = start_index + i;
                size_t feat    = indexes[idx_pos];
                T w = units[feat * output_len + out_index];

                acc += w;
            }

            T val = acc + bias[out_index];

            output[index] = val;
        }

        /*
        if (tid == 0 && index < batch_size * output_len) {
            output[index] = acc + bias[out_index];
        }
        */
    }
}

template<typename T>

__device__ void transform_features_gradient_batch(const T * __restrict__ loss,
                                                  const uint8_t * __restrict__ input,
                                                  T *output,
                                                  const size_t input_len,
                                                  const size_t output_len,
                                                  const size_t batch_size) {
    extern __shared__ char smem[];

    float *sdata_c = reinterpret_cast<float*>(&smem[0]);
    half *sdata_a = reinterpret_cast<half*>(&smem[TILE_SIZE_2D * sizeof(float)]);
    half *sdata_b = reinterpret_cast<half*>(&smem[TILE_SIZE_2D * sizeof(float) + TILE_SIZE_2D * sizeof(half)]);

    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> c_frag;

    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    size_t bx = blockIdx.x * TILE_SIZE;
    size_t by = blockIdx.y * TILE_SIZE;

    wmma::fill_fragment(c_frag, 0.0f);

    __syncthreads();

    for (int k = 0; k < batch_size; k += TILE_SIZE) {
        size_t chunk_offset = (input_len + 7) / 8 * (k + ty);
        size_t chunk_index = (bx + tx) / 8;
        size_t bit_index = (bx + tx) - chunk_index * 8;

        if (k + ty < batch_size && bx + tx < input_len &&
            (input[chunk_offset + chunk_index] & (1 << bit_index)) != 0) {
            sdata_a[tx * TILE_SIZE + ty] = __float2half(1.0f);
        } else {
            sdata_a[tx * TILE_SIZE + ty] = __float2half(0.0f);
        }

        if (k + ty < batch_size && by + tx < output_len) {
            T g = loss[calc_index(k+ty, by+tx, output_len)];

            half h = _to_half(g);

            sdata_b[ty * TILE_SIZE + tx] = h;
        } else {
            sdata_b[ty * TILE_SIZE + tx] = __float2half(0.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, sdata_a, TILE_SIZE);
        wmma::load_matrix_sync(b_frag, sdata_b, TILE_SIZE);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(sdata_c, c_frag, TILE_SIZE, wmma::mem_row_major);

    __syncthreads();

    if (ty + by < output_len && tx + bx < input_len) {
        output[calc_index(tx+bx,ty+by,output_len)] = (T)sdata_c[tx * TILE_SIZE + ty];
    }
}

template<typename T>

__device__ void bi_mix_accumulator(const T * __restrict__ input,
                                         T *output,
                                   const size_t input_len,
                                   const size_t batch_size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    long offset = (batch_index % 2) == 0 ? input_len : -((long)input_len);

    if (index < input_len && batch_index < batch_size) {
        output[batch_index * input_len + index] = input[batch_index * input_len + index] + input[batch_index * input_len + index + offset];
    }
}

extern "C" {
    __global__ void forward_transform_features_batch_float(const size_t *indexes, const size_t *boundaries,
                                                     const float *units, const float *bias, float *output,
                                                     const size_t output_len,
                                                     const size_t batch_size) {
        forward_transform_features_batch(indexes,boundaries,units,bias,output,output_len,batch_size);
    }

    __global__ void bi_mix_accumulator_float(const float * __restrict__ input,
                                                   float *output,
                                             const size_t input_len,
                                             const size_t batch_size) {
        bi_mix_accumulator(input,output,input_len,batch_size);
    }

    __global__ void forward_transform_features_batch_double(const size_t *indexes, const size_t *boundaries,
                                                     const double *units, const double *bias, double *output,
                                                     const size_t output_len,
                                                     const size_t batch_size) {
        forward_transform_features_batch(indexes,boundaries,units,bias,output,output_len,batch_size);
    }

    __global__ void transform_features_gradient_batch_float(const float * __restrict__ loss,
                                                            const uint8_t * __restrict__ input,
                                                            float *output,
                                                            const size_t input_len,
                                                            const size_t output_len,
                                                            const size_t batch_size) {
        transform_features_gradient_batch(loss,input,output,input_len,output_len,batch_size);
    }

    __global__ void transform_features_gradient_batch_double(const double * __restrict__ loss,
                                                             const uint8_t * __restrict__ input,
                                                             double *output,
                                                             const size_t input_len,
                                                             const size_t output_len,
                                                             const size_t batch_size) {
        transform_features_gradient_batch(loss,input,output,input_len,output_len,batch_size);
    }

    __global__ void bi_mix_accumulator_double(const double * __restrict__ input,
                                                    double *output,
                                              const size_t input_len,
                                              const size_t batch_size) {
        bi_mix_accumulator(input,output,input_len,batch_size);
    }

    __global__ void transform_features_input_to_bits(const size_t * __restrict__ indexes,
                                                     const size_t * __restrict__ boundaries,
                                                     uint8_t * bits,
                                                     const size_t input_len,
                                                     const size_t batch_size) {
        size_t batch_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_index < batch_size) {
            size_t start_index = boundaries[batch_index];
            size_t end_index = boundaries[batch_index + 1];

            for (size_t j = start_index; j < end_index; j++) {
                size_t input_index = indexes[j];

                size_t chunk_offset = (input_len + 7) / 8 * batch_index;
                size_t chunk_index = input_index / 8;
                size_t bit_index = input_index - chunk_index * 8;

                bits[chunk_offset + chunk_index] |= 1 << bit_index;
            }
        }
    }
}