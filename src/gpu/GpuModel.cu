#include "Timer.h"

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