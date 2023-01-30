#pragma once

#include <vector>
#include <include.h>

#define WEED_COLOR  make_float4(0.1, 1, 0, 1.0f);
#define RABIT_COLOR  make_float4(0, 0, 1, 1.0f);
#define FOX_COLOR  make_float4(1, 0, 0, 1.0f);
#define FOX_NUM 50;
#define RABBIT_NUM 500;

void GetGPGPUInfo();
void DrawUVs(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time);
void DrawRabbits(cudaSurfaceObject_t surface, Rabbit* rabbits, int32_t width, int32_t height, float time);
void DrawFoxs(cudaSurfaceObject_t surface,Fox* foxs, int32_t width, int32_t height, float time);
void CopyTo(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out, int32_t width, int32_t height);
void SendToGPU(void** dst, void* src, size_t size);
void move(cudaSurfaceObject_t surface, Fox* foxs, Rabbit* rabbits, int32_t width, int32_t height, float time);

float fracf(float x);