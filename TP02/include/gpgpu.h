#pragma once

#include <vector>

void GetGPGPUInfo();
void GenerateGrayscaleImage(std::vector<uint8_t>& host_image_uint8, int32_t width, int32_t height);