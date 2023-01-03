#include <iostream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

void main() {
	std::vector<uint8_t> image_data(512*512*3, 128);
	stbi_write_png("Hello_World.png", 512, 512, 3, image_data.data(), 512*3);
}
