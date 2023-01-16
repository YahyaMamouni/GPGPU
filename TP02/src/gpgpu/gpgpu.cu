#include <gpgpu.h>
#include <algorithm>
#include <iostream>
#include <cuComplex.h>


//Kernel gray-scale

__global__ void kernel_gray(float * device_image_float){
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    device_image_float[index] = device_image_float[index] + 0.5f;
}


__global__ void kernel_uv(float * image, int32_t width, int32_t height){
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t y = blockIdx.y * blockDim.y + threadIdx.y;


    int index;
    index = 3 * (width * y + x);
    // Local position
    //x = threadIdx.x;
    //y = threadIdx.y;
    // Local color
    //float u = (float)x / (float)32;
    //float v = (float)y / (float)32;
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;
    int ir = int(255.0 * (1-u));
    int ig = int(255.0 * (1-v));

    image[index++] = ir;
    image[index++] = ig;
    image[index++] = 0;
        
    

}

//kernel_mandelbrot
__global__ void kernel_mandelbrot(float* image) {
    // we need this for the index
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    int index;
    index = 3 * (1024 * y + x);
    // this is the counter that we use in while
    unsigned int cmp = 0u;
    // We limit our x & y to a number between 0 and 1 and we give it to u and v
    // Because the mandelbrot accept values between -2 and 2
    float u = float(x) / 1024;
    float v = float(y) / 1024;
    // c is the complex related to u,v
    // z is an init value of (0.f, 0.f) in the complex
    cuFloatComplex c = make_cuFloatComplex(u, v);
    cuFloatComplex z = make_cuFloatComplex(0.f,0.f);
    // Then we loop 
    while (cuCabsf(z) < 1.f && cmp <= 400) {
        z = cuCaddf(cuCmulf(z, z), c);
        cmp++;
    }

    image[index++] = cuCabsf(z);
    image[index++] = 0;
    image[index++] = 0;


}


void GetGPGPUInfo(){
    cudaDeviceProp cuda_properties;
    cudaGetDeviceProperties(&cuda_properties, 0);
    // Display the maxThreadsPerBlock property
    std::cout << "Max threads per block : \n" << cuda_properties.maxThreadsPerBlock;
}

void GenerateGrayscaleImage(std::vector<uint8_t>& host_image_uint8, int32_t width, int32_t height){
    std::vector<float> host_image_float;
    host_image_float.reserve(host_image_uint8.size());
    // Filling host_image_float with the content of host_image_uint8
    // Using std::transform
    // the y argument will be the corresponding value in the output image.
    for (int i = 0; i < host_image_uint8.size(); i++)
	{
		float f = host_image_uint8.at(i);
		host_image_float.push_back(f / 255);
	}

    // Asking CUDA to duplicate the new image on the GPU

    // Init a temporary image that we will use
    float * device_image_float = nullptr;
    cudaMalloc(&device_image_float, host_image_uint8.size()*sizeof(float));
    // Copy data of host_image_float into device_image_float
    cudaMemcpy(
        device_image_float,
        host_image_float.data(),
        host_image_uint8.size() * sizeof(float),
        cudaMemcpyHostToDevice);
    // At this point host_image_uint8 and host_image_float are only editable by the host and device_image_float only by the device

    // Call to the kernel gray
    //kernel_gray<<<height*3, 1024>>>(device_image_float);

    // Call the kernel uv
    dim3 threads(32,32);
    dim3 blocks(32, 32);
    //kernel_uv <<<blocks, threads>>>(device_image_float, width, height);

    // Mandelbrot
    kernel_mandelbrot <<<blocks, threads>>> (device_image_float);

    cudaMemcpy(
        host_image_float.data(),
        device_image_float,
        host_image_uint8.size() * sizeof(float),
        cudaMemcpyDeviceToHost);

    for (int i = 0; i < host_image_float.size(); i++)
	{
		host_image_uint8.at(i) = host_image_float.at(i) * 255;
	}

    //Free the GPU buffer
    cudaFree(device_image_float);
}