#pragma once

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
	bool is_alive;
	//...
};

struct Fox {
	float u;
	float v;
	float radius;
	float direction_u;
	float direction_v;
	bool is_alive;
	int life_duration;
	//...
};

void init_rabbit_fox(Rabbit** rabits, int lenghtRabbit, Fox** foxs, int lenghtFox);