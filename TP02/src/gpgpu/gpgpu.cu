#include <gpgpu.h>
#include <algorithm>
#include <iostream>

void GetGPGPUInfo() {
	cudaDeviceProp cuda_properties;
	cudaGetDeviceProperties(&cuda_properties, 0);
	std::cout << cuda_properties::cudamaxThreadsPerBlock;
}