#include "GpuModel.h"
#include <cuda_fp16.h>

#define TILE_WIDTH_L1 28
#define TILE_WIDTH_L3 12

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

__global__ void kernel_conv_forward_gpu_v1(float *output, const float *input, const float *weight, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel)
{
    // Calculate indices
    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;

    int batch_idx = blockIdx.x;
    int output_feature_idx = blockIdx.y;
    int row_idx = (blockIdx.z / gridDim.z) * blockDim.y + threadIdx.y;
    int col_idx = (blockIdx.z % gridDim.z) * blockDim.x + threadIdx.x;

    float accumulator = 0.0f;

    // Loop over input channels, kernel rows, and kernel columns
    for (int channel_in_idx = 0; channel_in_idx < channel_in; channel_in_idx++)
    {
        for (int kernel_row = 0; kernel_row < height_kernel; kernel_row++)
        {
            for (int kernel_col = 0; kernel_col < height_kernel; kernel_col++)
            {
                int input_row = row_idx + kernel_row;
                int input_col = col_idx + kernel_col;

                // Load input values directly from global memory
                int input_index = (batch_idx * (channel_in * height_in * width_in)) + (channel_in_idx * (height_in * width_in)) + (input_row * width_in) + input_col;
                float input_value = input[input_index];

                // Compute convolution with shared memory (weight data)
                accumulator += input_value * weight[(output_feature_idx * (channel_in * height_kernel * height_kernel)) + (channel_in_idx * (height_kernel * height_kernel)) + (kernel_row * height_kernel) + kernel_col];
            }
        }
    }

    // Check bounds before writing to output
    if (row_idx < height_out && col_idx < width_out)
    {
        int output_index = (batch_idx * (channel_out * height_out * width_out)) + (output_feature_idx * (height_out * width_out)) + (row_idx * width_out) + col_idx;

        if (output_index < n_sample * channel_out * height_out * width_out)
        {
            atomicAdd(&output[output_index], accumulator);
        }
    }
}

void GPU_Conv::conv_forward_gpu_v1(float *output, const float *input, const float *weight, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel)
{
    int tile_width = (channel_in == 1) ? TILE_WIDTH_L1 : TILE_WIDTH_L3;
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

    // Set grid and block dimensions for kernel and launch it
    dim3 num_threads_per_block(tile_width, tile_width, 1);
    dim3 num_blocks_in_grid(n_sample, channel_out, ceil(1.0 * height_out / tile_width) * ceil(1.0 * width_out / tile_width));

    // Launch kernel
    kernel_conv_forward_gpu_v1<<<num_blocks_in_grid, num_threads_per_block>>>(device_output, device_input, device_weight, n_sample, channel_out, channel_in, height_in, width_in, height_kernel);
    CHECK(cudaGetLastError());

    // Copy the result back to host
    cudaMemcpy(output, device_output, n_sample * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_weight);
}

__global__ void kernel_conv_forward_gpu_v2(float *output, const float *input, const float *weight, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel, const int stream_offset)
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

    // Mỗi thread sao chép dữ liệu trọng số vào shared memory
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        for (int i = 0; i < channel_out * channel_in * height_kernel * height_kernel; i++)
        {
            shared_data[i] = __float2half2_rn(weight[i + stream_offset]);
        }
    }

    __syncthreads();

    // Mỗi stream chỉ xử lý một phần của n_sample
    if (batch_idx >= n_sample)
    {
        return;
    }

#pragma unroll 1
    for (int channel_in_idx = 0; channel_in_idx < channel_in; channel_in_idx++)
    {
#pragma unroll 1
        for (int kernel_row = 0; kernel_row < height_kernel; kernel_row++)
        {
#pragma unroll 1
            for (int kernel_col = 0; kernel_col < height_kernel; kernel_col++)
            {
                // Sử dụng restrict keyword để chỉ định rằng input và weight không trỏ đến nhau
                int input_row = row_idx + kernel_row;
                int input_col = col_idx + kernel_col;

                // Tải giá trị input trực tiếp từ bộ nhớ toàn cục
                int input_index = (batch_idx * (channel_in * height_in * width_in)) + (channel_in_idx * (height_in * width_in)) + (input_row * width_in) + input_col;
                __half2 input_value = __float2half2_rn(input[input_index]);

                int weight_index = (output_feature_idx * (channel_in * height_kernel * height_kernel)) + (channel_in_idx * (height_kernel * height_kernel)) + (kernel_row * height_kernel) + kernel_col;
                __half2 weight_value = shared_data[weight_index];

                // Nhân và cộng
                __half2 input_mul_weight = __hmul2(input_value, weight_value);
                accumulator = __hadd2(accumulator, input_mul_weight);
            }
        }
    }

    // Chuyển đổi kết quả fixed-point trở lại FP16 để lưu trữ trong output
    __half result = __low2half(accumulator);

    // Kiểm tra ranh giới trước khi ghi vào output
    if (row_idx < height_out && col_idx < width_out)
    {
        int output_index = (batch_idx * (channel_out * height_out * width_out)) + (output_feature_idx * (height_out * width_out)) + (row_idx * width_out) + col_idx;

        if (output_index < n_sample * channel_out * height_out * width_out)
        {
            atomicAdd(&output[output_index], __half2float(result));
        }
    }
}

void GPU_Conv::conv_forward_gpu_v2(float *output, const float *input, const float *weight, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel)
{
    int tile_width = (channel_in == 1) ? TILE_WIDTH_L1 : TILE_WIDTH_L3;

    // Tính kích thước output
    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;

    // Lấy số stream tối đa có thể tạo
    int n_streams;
    cudaGetDeviceCount(&n_streams);

    // Tạo và quản lý nhiều stream
    cudaStream_t *streams = new cudaStream_t[n_streams];
    for (int i = 0; i < n_streams; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    // Tính toán samples_per_stream dựa trên n_sample và số stream
    int samples_per_stream = (n_sample + n_streams - 1) / n_streams;

    // Mảng lưu trữ offset của từng stream
    int *stream_offsets = new int[n_streams];
    for (int i = 0; i < n_streams; ++i)
    {
        stream_offsets[i] = i * samples_per_stream * channel_out * height_out * width_out; // Điều chỉnh offset dựa trên cách dữ liệu được tổ chức
    }

    // Duyệt qua từng stream và thực hiện công việc bất đồng bộ
    for (int i = 0; i < n_streams; ++i)
    {
        // Cấp phát bộ nhớ trên thiết bị
        float *device_input, *device_output, *device_weight;
        cudaMalloc((void **)&device_input, samples_per_stream * channel_in * height_in * width_in * sizeof(float));
        cudaMalloc((void **)&device_output, samples_per_stream * channel_out * height_out * width_out * sizeof(float));
        cudaMalloc((void **)&device_weight, channel_out * channel_in * height_kernel * height_kernel * sizeof(float));

        // Sao chép dữ liệu input và weight đến thiết bị bất đồng bộ
        cudaMemcpyAsync(device_input, input + stream_offsets[i], samples_per_stream * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(device_weight, weight, channel_out * channel_in * height_kernel * height_kernel * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        // Đồng bộ để đảm bảo dữ liệu sẵn có trước khi khởi động kernel
        cudaStreamSynchronize(streams[i]);

        dim3 num_threads_per_block(tile_width, tile_width, 1);
        dim3 num_blocks_in_grid(samples_per_stream, channel_out, ceil(1.0 * height_out / tile_width) * ceil(1.0 * width_out / tile_width));

        // Khởi động kernel với stream cụ thể
        kernel_conv_forward_gpu_v2<<<num_blocks_in_grid, num_threads_per_block, channel_out * channel_in * height_kernel * height_kernel * sizeof(float), streams[i]>>>(device_output, device_input, device_weight, samples_per_stream, channel_out, channel_in, height_in, width_in, height_kernel, stream_offsets[i]);
        cudaDeviceSynchronize(); // Đảm bảo kernel đã hoàn thành

        // Sao chép kết quả trở lại máy chủ bất đồng bộ
        cudaMemcpyAsync(output + stream_offsets[i], device_output, samples_per_stream * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);

        // Giải phóng bộ nhớ trên thiết bị
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFree(device_weight);
    }

    // Hủy bỏ các stream và giải phóng bộ nhớ
    for (int i = 0; i < n_streams; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
    delete[] stream_offsets;
}

__constant__ float deviceMaskData[2400];

__global__ void kernel_conv_forward_gpu_v3(float *output, const float *input, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel)
{
    int tile_width = (channel_in == 1) ? TILE_WIDTH_L1 : TILE_WIDTH_L3;

    extern __shared__ __half shared_input[];

    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;

    int W_grid = ceil(1.0 * width_out / tile_width);

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int h = (blockIdx.z / W_grid) * tile_width + ty;
    int w = (blockIdx.z % W_grid) * tile_width + tx;

    int startOfTile_h = (blockIdx.z / W_grid) * tile_width;
    int startOfTile_w = (blockIdx.z % W_grid) * tile_width;

#pragma unroll
    for (int c = 0; c < channel_in; c++)
    {
        for (int i = ty; i < tile_width + height_kernel - 1; i += tile_width)
        {
            for (int j = tx; j < tile_width + height_kernel - 1; j += tile_width)
            {
                if (startOfTile_h + i < height_in && startOfTile_w + j < width_in)
                {
                    int input_index = blockIdx.x * (channel_in * height_in * width_in) + c * (height_in * width_in) + (startOfTile_h + i) * width_in + startOfTile_w + j;
                    shared_input[c * (tile_width + height_kernel - 1) * (tile_width + height_kernel - 1) + i * (tile_width + height_kernel - 1) + j] = __float2half(input[input_index]);
                }
            }
        }
    }
    __syncthreads();

    if ((h < height_out) && (w < width_out))
    {
        __half accumulator = __float2half(0.0f);

        for (int c = 0; c < channel_in; c++)
        {
            for (int i = 0; i < height_kernel; i++)
            {
                for (int j = 0; j < height_kernel; j++)
                {
                    accumulator = __hadd(accumulator, __hmul(shared_input[c * (tile_width + height_kernel - 1) * (tile_width + height_kernel - 1) + (i + ty) * (tile_width + height_kernel - 1) + (j + tx)], deviceMaskData[blockIdx.y * (channel_in * height_kernel * height_kernel) + c * (height_kernel * height_kernel) + i * height_kernel + j]));
                }
            }
        }

        int output_index = blockIdx.x * (channel_out * height_out * width_out) + blockIdx.y * (height_out * width_out) + h * width_out + w;
        atomicAdd(&output[output_index], accumulator);
    }
}

void GPU_Conv::conv_forward_gpu_v3(float *output, const float *input, const float *weight, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel)
{
    int tile_width = (channel_in == 1) ? TILE_WIDTH_L1 : TILE_WIDTH_L3;

    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - height_kernel + 1;

    int inputSize = n_sample * channel_in * height_in * width_in * sizeof(float);
    int outputSize = n_sample * channel_out * height_out * width_out * sizeof(float);

    float *device_input, *device_output;

    cudaMalloc((void **)&device_input, inputSize);
    cudaMalloc((void **)&device_output, outputSize);

    cudaMemcpy(device_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceMaskData, weight, channel_out * channel_in * height_kernel * height_kernel * sizeof(float));

    dim3 numThreadsPerBlock, numBlocksInGrid;

    numThreadsPerBlock = dim3(tile_width, tile_width, 1);
    int shmem_size = channel_in * (tile_width + height_kernel - 1) * (tile_width + height_kernel - 1) * sizeof(float);
    numBlocksInGrid = dim3(n_sample, channel_out, ceil(1.0 * height_out / tile_width) * ceil(1.0 * width_out / tile_width));

    kernel_conv_forward_gpu_v3<<<numBlocksInGrid, numThreadsPerBlock, shmem_size>>>(device_output, device_input, n_sample, channel_out, channel_in, height_in, width_in, height_kernel);

    cudaMemcpy(output, device_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
}
