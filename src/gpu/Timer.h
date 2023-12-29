#ifndef SRC_GPU_TIMER_H
#define SRC_GPU_TIMER_H
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

#endif