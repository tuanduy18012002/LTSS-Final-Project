#ifndef SRC_GPU_GPUMODEL_H
#define SRC_GPU_GPUMODEL_H

#pragma once

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                \
	{                                                              \
		const cudaError_t error = call;                            \
		if (error != cudaSuccess)                                  \
		{                                                          \
			fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
			fprintf(stderr, "code: %d, reason: %s\n", error,       \
					cudaGetErrorString(error));                    \
			exit(1);                                               \
		}                                                          \
	}

struct Timer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	Timer();

	~Timer();

	void Start();

	void Stop();

	float Elapsed();
};

class GPU_Info
{
public:
	void printGpuInfo();
};

class GPU_Conv
{
public:
    void conv_forward_gpu_v1(float* output, const float* input, const float* weight, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel);
    void conv_forward_gpu_v2(float* output, const float* input, const float* weight, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel);
	void conv_forward_gpu_v3(float* output, const float* input, const float* weight, const int n_sample, const int channel_out, const int channel_in, const int height_in, const int width_in, const int height_kernel);
};
#endif