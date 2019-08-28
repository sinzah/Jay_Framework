#pragma once

#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

#define MIN(x,y) ((x<=y) ? x:y)
#ifdef debug
#define DEBUG(msg) std::cout << "[" << __func__ << "] " << msg << "\n";
#else
#define DEBUG(msg)
#endif

#define MEMSIZE(x) ((x) * sizeof(float))

void print_progress(int count, int max);

void transpose(float* src, float* dst, int height, int width);
double get_time();

#ifdef __cplusplus
}
#endif
