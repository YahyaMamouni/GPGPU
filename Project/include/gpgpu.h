#pragma once

#include <vector>

void GetGPGPUInfo();
void DrawUVs(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time);
void CopyTo(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out, int32_t width, int32_t height);
void DrawMap(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time);