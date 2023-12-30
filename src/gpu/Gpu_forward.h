#ifndef SRC_GPU_FORWARD_H
#define SRC_GPU_FORWARD_H
#pragma once

#include <stdio.h>
#include <stdint.h>
//#include <cuda_runtime.h>

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

class GpuForward
{
public:
  GpuForward(){};
	void Conv_forward(const Matrix &bottom, int Blocksize); 
};

#endif