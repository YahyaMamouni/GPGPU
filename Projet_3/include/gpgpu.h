#pragma once

#define MAX_FOX 50
#define MAX_RABBIT 500

// structures
struct Circle {
	float u;
	float v;
	float radius;
};

struct Rabbit {
	float u;
	float v;
	float radius;
	float direction_u;
	float direction_v;
	bool is_alive = false;
	int age = 0;
	//...
};

struct Fox {
	float u;
	float v;
	float radius;
	float direction_u;
	float direction_v;
	bool is_alive = false;
	int death = 500;
	int age = 0;
	//...
};

#include <vector>

void GetGPGPUInfo();
void init(Fox* fox_buffer, Rabbit* rabbit_buffer, Fox** device_foxes, Rabbit** device_rabbits);
void DrawUVs(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time);
void CopyTo(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out, int32_t width, int32_t height);
void DrawMap(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time, int number_foxes, int * number_rabbits);
void destroy(Fox* device_foxes, Rabbit* device_rabbits);