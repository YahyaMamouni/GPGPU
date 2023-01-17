#include <gpgpu.h>
#include <algorithm>
#include <iostream>


// structures
struct Circle {
	float u;
	float v;
	float radius;
};

struct Rabbit {
	float u;
	float v;
	float radius;
	float direction_u;
	float direction_v;
	bool is_alive;
	//...
};

struct Fox {
	float u;
	float v;
	float radius;
	//float direction_u;
	//float direction_v;
	//...
};

void GetGPGPUInfo() {
	cudaDeviceProp cuda_propeties;
	cudaGetDeviceProperties(&cuda_propeties, 0);
	std::cout << "maxThreadsPerBlock: " << cuda_propeties.maxThreadsPerBlock << std::endl;
}

__global__ void kernel_uv(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float u = (float)x / width;
	float v = (float)y / height;
	float4 color = make_float4(u, v, cos(time), 1.0f);
	surf2Dwrite(color, surface, x * sizeof(float4), y);
}

__global__ void kernel_copy(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	float4 color = make_float4(1.f, 0.f, 1.f, 1.0f);
	surf2Dread(&color, surface_in, x * sizeof(float4), y);
	surf2Dwrite(color, surface_out, x * sizeof(float4), y);
}

// kernel of draw map
__global__ void kernel_draw_map(cudaSurfaceObject_t surface) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 color = make_float4(0.6f, 0.9f, 0.05f, 1.0f);

	surf2Dwrite(color, surface, x * sizeof(float4), y);

}

void DrawUVs(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	kernel_uv << <blocks, threads >> > (surface, width, height, time);
}

void CopyTo(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out, int32_t width, int32_t height) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	kernel_copy << <blocks, threads >> > (surface_in, surface_out);
}

// Added a draw map 
void DrawMap(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);


	kernel_draw_map << <blocks, threads >> > (surface);
}


/*
void DrawAnimal(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time) {
	// buffer of Foxes & rabbits
	Fox* fox_buffer = new Fox[6];
	Rabbit* rabbit_buffer = new Rabbit[40];
	float4 rabbit_color = make_float4(0f, 0f, 0f, 1f);
	float4 fox_color = make_float4(255f, 255f, 255f, 1f);

	for (int i = 0; i < 6; i++) {
		std::uniform_real_distribution<> x(1, 1024);
		std::uniform_real_distribution<> y(1, 1024);
		fox_buffer[i].u = x;
		fox_buffer[i].v = y;
		
		surf2Dwrite(fox_color, surface, x * sizeof(float4), y);
	}

	for (int j = 0; j < 40; j++) {
		std::uniform_real_distribution<> x(1, 1024);
		std::uniform_real_distribution<> y(1, 1024);
		rabbit_buffer[i].u = x;
		rabbit_buffer[i].v = y;
		surf2Dwrite(rabbit_color, surface, x * sizeof(float4), y);
	}

	DrawMap(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time);
}
*/

