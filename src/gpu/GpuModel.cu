#include "GpuModel.h"

#define TILE_WIDTH 16

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
    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;
    
    extern __shared__ float shared_data[];
    
    int batch_idx = blockIdx.x;
    int output_feature_idx = blockIdx.y;
    int row_idx = (blockIdx.z / gridDim.z) * blockDim.y + threadIdx.y;
    int col_idx = (blockIdx.z % gridDim.z) * blockDim.x + threadIdx.x;
    
    float accumulator = 0.0f;

    for (int channel_in_idx = 0; channel_in_idx < channel_in; channel_in_idx++)
    {
        for (int kernel_row = 0; kernel_row < height_kernel; kernel_row++)
        {
            for (int kernel_col = 0; kernel_col < height_kernel; kernel_col++)
            {
                int input_row = row_idx + kernel_row;
                int input_col = col_idx + kernel_col;

                // Load input and kernel values into shared memory
                int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
                shared_data[shared_index] = input[(batch_idx * (channel_in * height_in * width_in)) +
                                                (channel_in_idx * (height_in * width_in)) +
                                                (input_row * width_in) +
                                                input_col];

                __syncthreads();

                // Compute convolution with shared memory
                for (int i = 0; i < height_kernel; i++)
                {
                    for (int j = 0; j < height_kernel; j++)
                    {
                        accumulator += shared_data[(threadIdx.y + i) * blockDim.x + (threadIdx.x + j)] *
                                       weight[(output_feature_idx * (channel_in * height_kernel * height_kernel)) +
                                              (channel_in_idx * (height_kernel * height_kernel)) +
                                              (kernel_row * height_kernel) +
                                              kernel_col];
                    }
                }

                __syncthreads();
            }
        }
    }

    if (row_idx < height_out && col_idx < width_out)
    {
        output[(batch_idx * (channel_out * height_out * width_out)) +
               (output_feature_idx * (height_out * width_out)) +
               (row_idx * width_out) +
               col_idx] = accumulator;
    }
}


void GPU_Conv::conv_forward_gpu(float* output, const float* input, const float* weight, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel)
{
    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;

    // Cấp phát bộ nhớ trên thiết bị
    float *device_input, *device_output, *device_weight;
    cudaMalloc((void **)&device_input, n_sample * channel_in * height_in * width_in * sizeof(float));  // Bản đồ đặc trưng đầu vào có kích thước input_channel
    cudaMalloc((void **)&device_output, n_sample * channel_out * height_out * width_out * sizeof(float));  // Bản đồ đặc trưng đầu ra có kích thước channel_out
    cudaMalloc((void **)&device_weight, channel_out * channel_in * height_kernel * height_kernel * sizeof(float));  // Bộ lọc kích thước input_channel * channel_out có kích thước height_kernel * height_kernel

    // Sao chép dữ liệu đầu vào và trọng số từ máy chủ đến thiết bị
    cudaMemcpy(device_input, input, n_sample * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight, channel_out * channel_in * height_kernel * height_kernel * sizeof(float), cudaMemcpyHostToDevice);

    // Đặt kích thước grid và block cho kernel và gọi kernel
    int height_grid = ceil(1.0 * height_out / TILE_WIDTH);
    int width_grid = ceil(1.0 * width_out / TILE_WIDTH);
    int Z = height_grid * width_grid;
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(n_sample, channel_out, Z);

    // Gọi kernel
    kernel_conv_forward_gpu<<<num_blocks_in_grid, num_threads_per_block, TILE_WIDTH * TILE_WIDTH * sizeof(float)>>>(device_output, device_input, device_weight, n_sample, channel_out, channel_in, height_in, width_in, height_kernel);
	CHECK(cudaGetLastError());

    // Sao chép kết quả đầu ra từ thiết bị về máy chủ
    cudaMemcpy(output, device_output, n_sample * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Giải phóng bộ nhớ trên thiết bị
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_weight);
}