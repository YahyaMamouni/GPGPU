#include <gpgpu.h>
#include <algorithm>
#include <iostream>
#include <include.h>
#include <curand.h>
#include <cuda_runtime.h>


void GetGPGPUInfo() {
	cudaDeviceProp cuda_propeties;
	cudaGetDeviceProperties(&cuda_propeties, 0);
	std::cout << "maxThreadsPerBlock: " << cuda_propeties.maxThreadsPerBlock << std::endl;
}

__device__ float fracf(float x)
{
	return x - floorf(x);
}

__device__ float random(float x, float y) {
	float t = 12.9898f * x + 78.233f * y;
	return abs(fracf(t * sin(t)));
}

__global__ void kernel_draw_map(cudaSurfaceObject_t surface) {

	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 color = WEED_COLOR;
	surf2Dwrite(color, surface, x * sizeof(float4), y);
	
}


__global__ void kernel_draw_fox(cudaSurfaceObject_t surface, Fox* foxs, int32_t width, int32_t height) {

	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = (float)x / width;
	float v = (float)y / height;

	float4 color = FOX_COLOR;
	for (int i = 0; i < 50; i++) {
		if (foxs[i].is_alive == 1) {
			if (hypotf(foxs[i].u - u, foxs[i].v - v) < foxs[i].radius) {
				color = FOX_COLOR;
				surf2Dwrite(color, surface, x * sizeof(float4), y);
			}
		}
	}

}

__global__ void kernel_draw_rabbit(cudaSurfaceObject_t surface, Rabbit* rabbits, int32_t width, int32_t height) {

	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = (float)x / width;
	float v = (float)y / height;

	float4 color = RABIT_COLOR;
	for (int i = 0; i < 500; i++) {
		if (rabbits[i].is_alive == 1) {
			if (hypotf(rabbits[i].u - u, rabbits[i].v - v) < rabbits[i].radius) {
				color = RABIT_COLOR;
				surf2Dwrite(color, surface, x * sizeof(float4), y);
			}
		}
	}
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

void DrawUVs(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);

	/*kernel_uv << <blocks, threads >> > (surface, width, height, time);*/
	kernel_draw_map << <blocks, threads >> > (surface);
	//kernel_draw_fox << <blocks, threads >> > (surface, width , height);
}

void CopyTo(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out, int32_t width, int32_t height) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	kernel_copy << <blocks, threads >> > (surface_in, surface_out);
}


void SendToGPU(void** dst, void* src, size_t size) {
	cudaMalloc(dst, size);
	cudaMemcpy(
		*dst,
		src,
		size,
		cudaMemcpyHostToDevice);
}

void DrawRabbits(cudaSurfaceObject_t surface, Rabbit* rabbits, int32_t width, int32_t height, float time) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);

	kernel_draw_rabbit << <blocks, threads >> > (surface, rabbits, width, height);
}
void DrawFoxs(cudaSurfaceObject_t surface, Fox* foxs, int32_t width, int32_t height, float time) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);

	kernel_draw_fox << <blocks, threads >> >  (surface, foxs, width, height);
}


__global__ void kernel_move_foxs(Fox* foxs, float time__) {
	int32_t x = threadIdx.x;

	float rand = (random(foxs[x].u, foxs[x].v)-0.5) * 3.14/4;
	float temp = foxs[x].direction_u * cos(rand) - foxs[x].direction_v * sin(rand);
	foxs[x].direction_v = foxs[x].direction_u * sin(rand) + foxs[x].direction_v * cos(rand);
	foxs[x].direction_u = temp;

	//printf("here: \f", (*rabbits[0]).u);
	foxs[x].u += foxs[x].direction_u / 700;
	foxs[x].v += foxs[x].direction_v / 700;
	if (foxs[x].u < 0 || foxs[x].u > 1) {
		foxs[x].direction_u *= -1;
	}
	if (foxs[x].v < 0 || foxs[x].v > 1) {
		foxs[x].direction_v *= -1;
	}

}

__global__ void kernel_move_rabbits(Rabbit* rabbits, float time__) {
	int32_t x = threadIdx.x;

	float rand = (random(rabbits[x].u, rabbits[x].v) - 0.5) * 3.14 / 2;
	float temp = rabbits[x].direction_u * cos(rand) - rabbits[x].direction_v * sin(rand);
	rabbits[x].direction_v = rabbits[x].direction_u * sin(rand) + rabbits[x].direction_v * cos(rand);
	rabbits[x].direction_u = temp;

	//printf("here: \f", (*rabbits[0]).u);
	rabbits[x].u += rabbits[x].direction_u / 5000;
	rabbits[x].v += rabbits[x].direction_v / 5000;

	if (rabbits[x].u < 0 || rabbits[x].u > 1) {
		rabbits[x].direction_u *= -1;
	}
	if (rabbits[x].v < 0 || rabbits[x].v > 1) {
		rabbits[x].direction_v *= -1;
	}

}

__global__ void kernel_kill_rabbits(Rabbit* rabbits, Fox* foxs) {
	int32_t x = threadIdx.x;

	if (foxs[x].is_alive == 1) {
		for (int i = 0; i < 500; i++) {
			if (rabbits[i].is_alive == 1) {
				if (hypotf(rabbits[i].u - foxs[x].u, rabbits[i].v - foxs[x].v) < rabbits[i].radius + 0.01) {
					rabbits[i].is_alive = 0;
					foxs[x].life_duration = 500;
				}
			}
		}
	}
}

__global__ void kernel_kill_foxs(Fox* foxs) {
	int32_t x = threadIdx.x;

	if (foxs[x].is_alive == 1) {
		//foxs[x].life_duration -= 50;
		if (--foxs[x].life_duration < 0) {
			foxs[x].is_alive = 0;
		}
		printf("%d \n", foxs[x].life_duration);
	}
}


void move(cudaSurfaceObject_t surface, Fox* foxs, Rabbit* rabbits, int32_t width, int32_t height, float time__) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);


	kernel_move_rabbits << <1, 500 >> > (rabbits, time__);
	kernel_move_foxs << <1, 50 >> > (foxs, time__);

	//kernel_draw_rabbit << <blocks, threads >> > (surface, rabbits, width, height);
	//kernel_draw_fox << <blocks, threads >> > (surface, foxs, width, height);

	kernel_kill_rabbits << <1, 50 >> > (rabbits, foxs);
	kernel_kill_foxs << <1, 50 >> > (foxs);
}