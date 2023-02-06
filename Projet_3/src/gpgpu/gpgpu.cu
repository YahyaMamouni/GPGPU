#include <gpgpu.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <assert.h>

//float4 FOX_COLOR = make_float4(1.0f, 0.5f, 0.0f, 1.0f);



// Override - 
__device__ float2 operator-(float2 a, float2 b) {
	return make_float2(a.x - b.x, a.y - b.y);
};

// random function

__device__ float fracf(float x)
{
	return x - floorf(x);
}

__device__ float random(float x, float y) {
	float t = 12.9898f * x + 78.233f * y;
	return abs(fracf(t * sin(t)));
}

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


// Kernel thats draws the green background
__global__ void kernel_draw_map(cudaSurfaceObject_t surface) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 color = make_float4(0.6f, 0.9f, 0.05f, 1.0f);

	surf2Dwrite(color, surface, x * sizeof(float4), y);
}


// Kernel to draw foxes
__global__ void DrawFoxes(cudaSurfaceObject_t surface, Fox* fox_buffer, float4 fox_color, int width, int height) {
	// calculate the x and y coordinates for the current thread
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	float u = (float)x / width;
	float v = (float)y / height;

	// iterate through the buffer & draw circles that represent foxes
	for (int i = 0; i < MAX_FOX; i++) {
		if (fox_buffer[i].is_alive == true){
			if (hypotf(fox_buffer[i].u - u, fox_buffer[i].v - v) < fox_buffer[i].radius) {
				surf2Dwrite(fox_color, surface, sizeof(float4) * x, y, cudaBoundaryModeTrap);
			}
		}
	}
}

// Kernel to draw rabbits
__global__ void DrawRabbits(cudaSurfaceObject_t surface, Rabbit* rabbit_buffer, float4 rabbit_color, int width, int height) {
	// calculate the x and y coordinates for the current thread
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	float u = (float)x / width;
	float v = (float)y / height;

	// iterate through the buffer & draw circles that represent rabbits
	for (int i = 0; i < MAX_RABBIT; i++) {
		if (rabbit_buffer[i].is_alive == true){
			if ((hypotf(rabbit_buffer[i].u - u, rabbit_buffer[i].v - v) < rabbit_buffer[i].radius)) {
				surf2Dwrite(rabbit_color, surface, sizeof(float4) * x, y, cudaBoundaryModeTrap);
			}
		}
	}
}

// Kernel to move rabbits
__global__ void MoveRabbits(Rabbit* rabbit_buffer, int random_age, float random_u, float random_v) {
	int32_t index = threadIdx.x;
	if (rabbit_buffer[index].is_alive) {
		rabbit_buffer[index].age = rabbit_buffer[index].age + random_age;

		// calculating new dierctions
		float random_val = (random(rabbit_buffer[index].u, rabbit_buffer[index].v) - 0.5) * 3.14 / 4;
		float temp = rabbit_buffer[index].direction_u * cos(random_val) - rabbit_buffer[index].direction_v * sin(random_val);
		rabbit_buffer[index].direction_v = rabbit_buffer[index].direction_u * sin(random_val) + rabbit_buffer[index].direction_v * cos(random_val);
		rabbit_buffer[index].direction_u = temp;

		// In case we hit a border we inverse the direction
		if (rabbit_buffer[index].u >= 1 || rabbit_buffer[index].u <= 0) {
			rabbit_buffer[index].direction_u = 0.0f - rabbit_buffer[index].direction_u;
		}

		if (rabbit_buffer[index].v >= 1 || rabbit_buffer[index].v <= 0) {
			rabbit_buffer[index].direction_v = 0.0f - rabbit_buffer[index].direction_v;
		}
		// Moving rabbits
		rabbit_buffer[index].u = rabbit_buffer[index].u + (rabbit_buffer[index].direction_u / 1000);
		rabbit_buffer[index].v = rabbit_buffer[index].v + (rabbit_buffer[index].direction_v / 1000);
	}
}

// Kernel to move rabbits and spawn new ones
__global__ void SpawnRabbits(Rabbit* rabbit_buffer, int random_age, float random_u, float random_v) {
	int32_t index = threadIdx.x;
	// Add a random offset to the age threshold
	int age_threshold = random_age * 100;
	if (rabbit_buffer[index].age >= age_threshold) {
		for (int i = 0 ; i < MAX_RABBIT; i++) {
			if (rabbit_buffer[i].is_alive != true) {
				// Change previous position to avoid flickering
				rabbit_buffer[i].u = random_u;
				rabbit_buffer[i].v = random_v;
				rabbit_buffer[i].is_alive = true;
				rabbit_buffer[i].age = 0;
				break;
			}
		}
	}

}

// Kernel to move foxes
__global__ void MoveFoxes(Fox* fox_buffer, int random_age) {
	int32_t index = threadIdx.x;
	if (fox_buffer[index].is_alive) {
		fox_buffer[index].age = fox_buffer[index].age + random_age;

		// calculating new dierctions
		float random_val = (random(fox_buffer[index].u, fox_buffer[index].v) - 0.5) * 3.14 / 4;
		float temp = fox_buffer[index].direction_u * cos(random_val) - fox_buffer[index].direction_v * sin(random_val);
		fox_buffer[index].direction_v = fox_buffer[index].direction_u * sin(random_val) + fox_buffer[index].direction_v * cos(random_val);
		fox_buffer[index].direction_u = temp;


		// In case we hit a border we inverse the direction
		if (fox_buffer[index].u >= 1 || fox_buffer[index].u <= 0) {
			fox_buffer[index].direction_u = 0.0f - fox_buffer[index].direction_u;
		}
		if (fox_buffer[index].v >= 1 || fox_buffer[index].v <= 0) {
			fox_buffer[index].direction_v = 0.0f - fox_buffer[index].direction_v;
		}
		// Moving foxes
		fox_buffer[index].u = fox_buffer[index].u + (fox_buffer[index].direction_u / 1000);
		fox_buffer[index].v = fox_buffer[index].v + (fox_buffer[index].direction_v / 1000);
	}
}

// Kernel to spawn foxes
__global__ void SpawnFoxes(Fox* fox_buffer, int random_age, float random_u, float random_v) {
	int32_t index = threadIdx.x;
	// Add a random offset to the age threshold
	int age_threshold = random_age * 50;
	if (fox_buffer[index].age >= age_threshold) {
		for (int i = 0 ; i < MAX_FOX; i++) {
			if (fox_buffer[i].is_alive != true) {
				// Change previous position to avoid flickering
				fox_buffer[i].u = random_u;
				fox_buffer[i].v = random_v;
				fox_buffer[i].is_alive = true;
				fox_buffer[i].age = 0;
				break;
			}
		}
	}

}


// Kill rabbits (Not atomic)
/*
__global__ void KillRabbits(cudaSurfaceObject_t surface, Rabbit* rabbit_buffer, Fox* fox_buffer) {
	int32_t index = threadIdx.x;

	for (int i = 0; i < MAX_FOX; i++) {

		if (fox_buffer[i].is_alive == true && rabbit_buffer[i].is_alive == true){
			if (hypotf(fox_buffer[i].u - rabbit_buffer[index].u, fox_buffer[i].v - rabbit_buffer[index].v) < fox_buffer[i].radius + 20) {
				rabbit_buffer[index].is_alive = false;
			}
			else{
				fox_buffer[i].death--;
			}
		}
	}

}*/

// Not atomic
/*
__global__ void ChaseRabbitsAndKillFoxes(cudaSurfaceObject_t surface, Fox* fox_buffer) {
	int32_t index = threadIdx.x;

		if (fox_buffer[index].is_alive == true && fox_buffer[index].death == 0){
			fox_buffer[index].is_alive = false;
		}
}*/


// Kernel to chase rabbits
__global__ void ChaseRabbits(Fox* fox_buffer, Rabbit* rabbit_buffer) {
	int32_t index = threadIdx.x;

	// Chase
	for (int i = 0; i < MAX_RABBIT; i++) {

		if (rabbit_buffer[i].is_alive == true && hypotf(fox_buffer[index].u - rabbit_buffer[i].u, fox_buffer[index].v - rabbit_buffer[i].v) < fox_buffer[index].radius + 0.03) {
			fox_buffer[index].direction_u = rabbit_buffer[i].u - fox_buffer[index].u;
			fox_buffer[index].direction_v = rabbit_buffer[i].v - fox_buffer[index].v;
			float norm = sqrt((fox_buffer[index].direction_u * fox_buffer[index].direction_u) + (fox_buffer[index].direction_v * fox_buffer[index].direction_v));
			fox_buffer[index].direction_u = fox_buffer[index].direction_u / norm;
			fox_buffer[index].direction_v = fox_buffer[index].direction_v / norm;
		}
	}
}

// Kernel to chase rabbits and kill foxes if they acheive a certain age
__global__ void KillFoxes(Fox* fox_buffer) {
	int32_t index = threadIdx.x;

	if (fox_buffer[index].is_alive == true && fox_buffer[index].death == 0) {
		atomicExch(&fox_buffer[index].is_alive, 0);
	}
}


// (Atomic) Kernel where foxes kill rabbits if they are in a certain radius
__global__ void KillRabbits(cudaSurfaceObject_t surface, Rabbit* rabbit_buffer, Fox* fox_buffer) {
    int32_t index = threadIdx.x;

    for (int i = 0; i < MAX_FOX; i++) {
        if (fox_buffer[i].is_alive == true && rabbit_buffer[i].is_alive == true){
            if (hypotf(fox_buffer[i].u - rabbit_buffer[index].u, fox_buffer[i].v - rabbit_buffer[index].v) < fox_buffer[i].radius + 0.005) {
				atomicExch(&rabbit_buffer[index].is_alive, 0);      
            }
			else {
				fox_buffer[i].death--;
			}
        }
    }
}



void DrawUVs(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	kernel_uv << <blocks, threads >> > (surface, width, height, time);
}

void CopyTo(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out, int32_t width, int32_t height) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	kernel_copy << <blocks, threads >>> (surface_in, surface_out);
}

void destroy(Fox* device_foxes, Rabbit* device_rabbits) {
	// free device-side memory
	cudaFree(device_foxes);
	cudaFree(device_rabbits);
}


// Function that calls all the used kernels
void DrawMap(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time, int number_foxes, int * number_rabbits) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);

	// Animal colors
	float4 fox_color = make_float4(1.0f, 0.5f, 0.0f, 1.0f);
	float4 rabbit_color = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

	// Draw the green background
	kernel_draw_map << <blocks, threads >> > (surface);

	// Bool used so we can init data only once in the beginning
	static bool is_init = false;

	// Device buffers
	static Fox* device_foxes;
	static Rabbit* device_rabbits;
	// Host buffers
	Fox* fox_buffer = new Fox[MAX_FOX];
	Rabbit* rabbit_buffer = new Rabbit[MAX_RABBIT];

	// Generate random u & v
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> x(0.0, 1.0);
	std::uniform_real_distribution<> y(0.0, 1.0);

	// Init only once then jump always
	if (!is_init) {
		is_init = true;

		// random number to generate random directions
		std::uniform_real_distribution<> direction_x(-1.0, 1.0);
		
		// Init foxes
		for (int i = 0; i < MAX_FOX; i++) {
			fox_buffer[i].u = x(gen);
			fox_buffer[i].v = y(gen);
			float dir_u = direction_x(gen);
			float dir_v = direction_x(gen);
			float norme = sqrt((dir_u * dir_u) + (dir_v * dir_v));
			fox_buffer[i].direction_u = dir_u / norme;
			fox_buffer[i].direction_v = dir_v / norme;
			fox_buffer[i].radius = 0.008;
			if (i < number_foxes){
				fox_buffer[i].is_alive = true;
			}
		}

		// Init rabbits
		for (int i = 0; i < MAX_RABBIT; i++) {
			rabbit_buffer[i].u = x(gen);
			rabbit_buffer[i].v = y(gen);
			float dir_u = direction_x(gen);
			float dir_v = direction_x(gen);
			float norme = sqrt((dir_u * dir_u) + (dir_v * dir_v));
			rabbit_buffer[i].direction_u = dir_u / norme;
			rabbit_buffer[i].direction_v = dir_v / norme;
			rabbit_buffer[i].radius = 0.007;
			if (i < *number_rabbits){
				rabbit_buffer[i].is_alive = true;
			}
		}

		// Allocate device-side memory
		
		cudaMalloc(&device_foxes, sizeof(Fox) * MAX_FOX);

		cudaMalloc(&device_rabbits, sizeof(Rabbit) * MAX_RABBIT);

		// copy data to device
		cudaMemcpy(device_foxes, fox_buffer, sizeof(Fox) * MAX_FOX, cudaMemcpyHostToDevice);
		cudaMemcpy(device_rabbits, rabbit_buffer, sizeof(Rabbit) * MAX_RABBIT, cudaMemcpyHostToDevice);
	}


	DrawFoxes << <blocks, threads >> > (surface, device_foxes, fox_color, width, height);
	DrawRabbits << <blocks, threads >> > (surface, device_rabbits, rabbit_color, width, height);
	assert(*number_rabbits <= 500);
	std::uniform_real_distribution<> random_age(0, 10);
	MoveRabbits <<<1, MAX_RABBIT >>> (device_rabbits, random_age(gen), x(gen), y(gen));
	SpawnRabbits << <1, MAX_RABBIT >> > (device_rabbits, random_age(gen), x(gen), y(gen));
	MoveFoxes << <1, MAX_FOX >> > (device_foxes, random_age(gen));
	SpawnFoxes << <1, MAX_FOX >> > (device_foxes, random_age(gen), x(gen), y(gen));
	ChaseRabbits <<<1, MAX_FOX >>> (device_foxes, device_rabbits);
	KillFoxes << <1, MAX_FOX >> > (device_foxes);
	KillRabbits << <1, MAX_RABBIT >> > (surface, device_rabbits, device_foxes);
}
