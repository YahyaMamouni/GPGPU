#include <iostream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>


void colorate_pixels(std::vector<uint8_t>& image_data, int i, int j, int w, int h, int& index) {
    float u = (float)i / (float)w;
    float v = (float)j / (float)h;
    int ir = int(255.0 * u);
    int ig = int(255.0 * v);

    image_data[index++] = ir;
    image_data[index++] = ig;
    image_data[index++] = 0;
}



void fill_image(std::vector<uint8_t>& image_data, int w, int h) {
    int index = 0;
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; ++i)
        {
            // This is your Kernel function
            colorate_pixels(image_data, i, j, w, h, index);
        }
    }
}


void main() {
	std::vector<uint8_t> image_data(512*512*3, 128);
    float w = 512.0;
    float h = 512.0;


    fill_image(image_data, w, h);

	stbi_write_png("Hello_World.png", 512, 512, 3, image_data.data(), 512*3);



}



// Exam ecrit sur papier 50%
// Projet a rendre 50%