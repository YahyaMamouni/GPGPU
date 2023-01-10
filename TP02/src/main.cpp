#include <iostream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <gpgpu.h>

constexpr int32_t kWidth = 1024;
constexpr int32_t kHeight = 1024;


void main() {

	// Dispaly GPU properties
	GetGPGPUInfo();

	std::vector<uint8_t> image_data(kWidth * kHeight * 3, 42);

	// using the generate gray scale function
	GenerateGrayscaleImage(image_data, kWidth, kHeight);

	stbi_write_png("Hello_World.png", kWidth, kHeight, 3, image_data.data(), kWidth *3);
}
