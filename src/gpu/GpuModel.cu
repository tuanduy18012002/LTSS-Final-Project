#include "GpuModel.h"
#include <cuda_fp16.h>

#define TILE_WIDTH 28

void GPU_Info::printGpuInfo()
{
    cudaDeviceProp devProv;
	CHECK(cudaGetDeviceProperties(&devProv, 0));
	printf("**********GPU info**********\n");
	printf("Name: %s\n", devProv.name);
	printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
	printf("Num SMs: %d\n", devProv.multiProcessorCount);
	printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
	printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
	printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
	printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
	printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
	printf("****************************\n");
}

Timer::Timer()
{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

Timer::~Timer()
{
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void Timer::Start()
{
	cudaEventRecord(start, 0);
	cudaEventSynchronize(start);
}

void Timer::Stop()
{
	cudaEventRecord(stop, 0);
}

float Timer::Elapsed()
{
	float elapsed;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	return elapsed;
}

__global__ void kernel_conv_forward_gpu(float* output, const float* input, const float* weight, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel) 
{
    // Calculate indices
    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;

    int batch_idx = blockIdx.x;
    int output_feature_idx = blockIdx.y;
    int row_idx = (blockIdx.z / gridDim.z) * blockDim.y + threadIdx.y;
    int col_idx = (blockIdx.z % gridDim.z) * blockDim.x + threadIdx.x;

    // Initialize accumulator with fixed-point representation
    __half2 accumulator = __float2half2_rn(0.0f);

    extern __shared__ __half2 shared_data[];

    // Một thread sao chép dữ liệu trọng số vào shared memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < channel_out * channel_in * height_kernel * height_kernel; i++) {
            shared_data[i] = __float2half2_rn(weight[i]);
        }
    }

    __syncthreads();

    #pragma unroll 1
    for (int channel_in_idx = 0; channel_in_idx < channel_in; channel_in_idx++) {
        #pragma unroll 1
        for (int kernel_row = 0; kernel_row < height_kernel; kernel_row++) {
            #pragma unroll 1
            for (int kernel_col = 0; kernel_col < height_kernel; kernel_col++) {
                // Use restrict keyword to indicate that input and weight do not alias
                int input_row = row_idx + kernel_row;
                int input_col = col_idx + kernel_col;

                // Load input values directly from global memory
                int input_index = (batch_idx * (channel_in * height_in * width_in)) + (channel_in_idx * (height_in * width_in)) + (input_row * width_in) + input_col;
                __half2 input_value = __float2half2_rn(input[input_index]);

                int weight_index = (output_feature_idx * (channel_in * height_kernel * height_kernel)) + (channel_in_idx * (height_kernel * height_kernel)) + (kernel_row * height_kernel) + kernel_col;
                __half2 weight_value = shared_data[weight_index];

                // Multiplication and addition
                __half2 input_mul_weight = __hmul2(input_value, weight_value);
                accumulator = __hadd2(accumulator, input_mul_weight);
            }
        }
    }

    // Convert fixed-point result back to FP16 for storage in output
    __half result = __low2half(accumulator);

    // Check bounds before writing to output
    if (row_idx < height_out && col_idx < width_out) {
        int output_index = (batch_idx * (channel_out * height_out * width_out)) + (output_feature_idx * (height_out * width_out)) + (row_idx * width_out) + col_idx;

        if (output_index < n_sample * channel_out * height_out * width_out) {

            atomicAdd(&output[output_index], __half2float(result));
        }
    }
}

void GPU_Conv::conv_forward_gpu(float* output, const float* input, const float* weight, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel)
{
    // Calculate output size
    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;

    // Allocate device memory
    float *device_input, *device_output, *device_weight;
    cudaMalloc((void **)&device_input, n_sample * channel_in * height_in * width_in * sizeof(float));
    cudaMalloc((void **)&device_output, n_sample * channel_out * height_out * width_out * sizeof(float));
    cudaMalloc((void **)&device_weight, channel_out * channel_in * height_kernel * height_kernel * sizeof(float));

    // Copy input and weight data to device
    cudaMemcpy(device_input, input, n_sample * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight, channel_out * channel_in * height_kernel * height_kernel * sizeof(float), cudaMemcpyHostToDevice);

        dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 num_blocks_in_grid(n_sample, channel_out, ceil(1.0 * height_out / TILE_WIDTH) * ceil(1.0 * width_out / TILE_WIDTH));

        // Launch kernel
        kernel_conv_forward_gpu<<<num_blocks_in_grid, num_threads_per_block, channel_out*channel_in*height_kernel*height_kernel*sizeof(float)>>>(device_output, device_input, device_weight, n_sample, channel_out, channel_in, height_in, width_in, height_kernel);
        CHECK(cudaGetLastError());
    // Copy the result back to host
    cudaMemcpy(output, device_output, n_sample * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_weight);
}