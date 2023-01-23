#include <gpgpu.h>
#include <algorithm>
#include <iostream>
#include <random>

//float4 FOX_COLOR = make_float4(1.0f, 0.5f, 0.0f, 1.0f);



// Override - 
__device__ float2 operator-(float2 a, float2 b) {
	return make_float2(a.x - b.x, a.y - b.y);
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

// draw a circle 
/*
__global__ void drawCircleKernel(cudaSurfaceObject_t surface, int centerX, int centerY, int radius, float4 color) {
	// calculate the x and y coordinates for the current thread
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	Fox myfox = { 120.0, 120.0, 10 };
	

	// check if the current pixel is within the circle
	//if (hypotf(centerX - x, centerY - y) < radius) {
		//surf2Dwrite(color, surface, sizeof(float4) * x, y, cudaBoundaryModeTrap);
	//}

	if (hypotf(myfox.u - x, myfox.v - y) < myfox.radius) {
		surf2Dwrite(color, surface, sizeof(float4) * x, y, cudaBoundaryModeTrap);
	}

}
*/

// define the kernel
__global__ void drawCircleKernel(cudaSurfaceObject_t surface, Fox* fox_buffer, Rabbit* rabbit_buffer, float4 fox_color, float4 rabbit_color, int number_foxes, int number_rabbits) {
	// calculate the x and y coordinates for the current thread
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// iterate through the buffer
	for (int i = 0; i < MAX_FOX; i++) {

		if (fox_buffer[i].is_alive == true){
			if (hypotf(fox_buffer[i].u - x, fox_buffer[i].v - y) < fox_buffer[i].radius) {
				surf2Dwrite(fox_color, surface, sizeof(float4) * x, y, cudaBoundaryModeTrap);
			}
			// Changing positions of fox
			if (fox_buffer[i].u >= 1000 || fox_buffer[i].u <= 10){
				fox_buffer[i].direction_u = 0.0f - fox_buffer[i].direction_u;
			}
			// Change of direction
			if (fox_buffer[i].v >= 1000 || fox_buffer[i].v <= 10){
				fox_buffer[i].direction_v = 0.0f - fox_buffer[i].direction_v;
			} 
			fox_buffer[i].u = fox_buffer[i].u + fox_buffer[i].direction_u;
			fox_buffer[i].v = fox_buffer[i].v + fox_buffer[i].direction_v;
		}

	}

	for (int i = 0; i < MAX_RABBIT; i++) {
		if (rabbit_buffer[i].is_alive == true){
			if ((hypotf(rabbit_buffer[i].u - x, rabbit_buffer[i].v - y) < rabbit_buffer[i].radius)) {
				surf2Dwrite(rabbit_color, surface, sizeof(float4) * x, y, cudaBoundaryModeTrap);
			}

			// Changing positions of rabbit
			if (rabbit_buffer[i].u >= 1000 || rabbit_buffer[i].u <= 10){
				rabbit_buffer[i].direction_u = 0.0f - rabbit_buffer[i].direction_u;
			}
			
			if (rabbit_buffer[i].v >= 1000 || rabbit_buffer[i].v <= 10){
				rabbit_buffer[i].direction_v = 0.0f - rabbit_buffer[i].direction_v;
			} 
			// Changing positions of rabbit
			rabbit_buffer[i].u = rabbit_buffer[i].u + rabbit_buffer[i].direction_u;
			rabbit_buffer[i].v = rabbit_buffer[i].v + rabbit_buffer[i].direction_v;
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
/*
// Init foxes 
__global__ void kernel_initFoxes(Fox* foxes, int numFoxes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFoxes)
    {
        std::mt19937_64 gen(clock());
        std::uniform_real_distribution<float> uDist(0.0f, 1024);
        std::uniform_real_distribution<float> vDist(0.0f, 1024);
        std::uniform_real_distribution<float> rDist(0.0f, 50);

        foxes[idx].u = uDist(gen);
        foxes[idx].v = vDist(gen);
        foxes[idx].radius = rDist(gen);
    }
}*/

/* Working one
// Added a draw map 
void DrawMap(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time) {

	dim3 threads(32, 32);
	dim3 blocks(32, 32);

	kernel_draw_map << <blocks, threads >> > (surface);

	// Init foxes
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> x(1.0, 500.0);
	std::uniform_real_distribution<> y(1.0, 500.0);
	Fox* fox_buffer = new Fox[50];
	for (int i = 0; i < 50; i++) {
		fox_buffer[i].u = x(gen);
		fox_buffer[i].v = y(gen);
		fox_buffer[i].radius = 50;
	}




	drawCircleKernel << <blocks, threads >> > (surface, 150, 512, 50, make_float4(1.0f, 0.5f, 0.0f, 1.0f));
	// using buffer
	//drawCircleKernel << <blocks, threads >> > (surface, fox_buffer, make_float4(1.0f, 0.5f, 0.0f, 1.0f));

}*/
/*
void init(Fox* fox_buffer, Rabbit* rabbit_buffer, Fox** device_foxes, Rabbit** device_rabbits) {

	int number_foxes = 6;
	int number_rabbits = 50;
	// Init foxes
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> x(0.0, 1023.0);
	std::uniform_real_distribution<> y(0.0, 1023.0);
	// host foxes
	
	for (int i = 0; i < number_foxes; i++) {
		fox_buffer[i].u = x(gen);
		fox_buffer[i].v = y(gen);
		fox_buffer[i].radius = 15;
	}

	fox_buffer[0].u = 100;
	fox_buffer[0].v = 100;


	for (int i = 0; i < number_rabbits; i++) {
		rabbit_buffer[i].u = x(gen);
		rabbit_buffer[i].v = y(gen);
		rabbit_buffer[i].radius = 15;
	}

	// Allocate device-side memory

	cudaMalloc(device_foxes, sizeof(Fox) * number_foxes);

	
	cudaMalloc(device_rabbits, sizeof(Rabbit) * number_rabbits);

	// copy data to device
	cudaMemcpy(device_foxes, fox_buffer, sizeof(Fox) * number_foxes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_rabbits, rabbit_buffer, sizeof(Rabbit) * number_rabbits, cudaMemcpyHostToDevice);

	// copy data back to host
	//cudaMemcpy(fox_buffer, device_foxes, sizeof(Fox) * number_foxes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(rabbit_buffer, device_rabbits, sizeof(Rabbit) * number_rabbits, cudaMemcpyDeviceToHost);

	
}
*/
void destroy(Fox* device_foxes, Rabbit* device_rabbits) {
	// free device-side memory
	cudaFree(device_foxes);
	cudaFree(device_rabbits);
}



//Trying to make the buffer work
void DrawMap(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time, int number_foxes, int number_rabbits) {

	dim3 threads(32, 32);
	dim3 blocks(32, 32);


	float4 fox_color = make_float4(1.0f, 0.5f, 0.0f, 1.0f);
	float4 rabbit_color = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

	kernel_draw_map << <blocks, threads >> > (surface);
	
	static bool is_init = false;
	static Fox* device_foxes;
	static Rabbit* device_rabbits;
	//init only once then jump always
	if (!is_init) {
		is_init = true;
		// Init foxes
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> x(1.0, 1024.0);
		std::uniform_real_distribution<> y(1.0, 1024.0);
		// direction u v
		std::uniform_real_distribution<> direction_x(-0.001, 0.001);
		std::uniform_real_distribution<> direction_y(-0.001, 0.001);
		// host foxes
		Fox* fox_buffer = new Fox[MAX_FOX];
		for (int i = 0; i < MAX_FOX; i++) {
			fox_buffer[i].u = x(gen);
			fox_buffer[i].v = y(gen);
			fox_buffer[i].direction_u = direction_x(gen);
			fox_buffer[i].direction_v = direction_y(gen);
			fox_buffer[i].radius = 15;
			if (i < number_foxes){
				fox_buffer[i].is_alive = true;
			}
		}

		// host rabbits
		Rabbit* rabbit_buffer = new Rabbit[MAX_RABBIT];
		for (int i = 0; i < MAX_RABBIT; i++) {
			rabbit_buffer[i].u = x(gen);
			rabbit_buffer[i].v = y(gen);
			rabbit_buffer[i].direction_u = direction_x(gen);
			rabbit_buffer[i].direction_v = direction_y(gen);
			rabbit_buffer[i].radius = 15;
			if (i < number_rabbits){
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

	drawCircleKernel << <blocks, threads >> > (surface, device_foxes, device_rabbits, fox_color, rabbit_color, number_foxes, number_rabbits);

	// copy data back to host
	/*cudaMemcpy(fox_buffer, device_foxes, sizeof(Fox) * number_foxes, cudaMemcpyDeviceToHost);
	cudaMemcpy(rabbit_buffer, device_rabbits, sizeof(Rabbit) * number_rabbits, cudaMemcpyDeviceToHost);

	// free device-side memory
	cudaFree(device_foxes);
	cudaFree(device_rabbits);*/
	
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

