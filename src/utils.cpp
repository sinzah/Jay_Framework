#include "typeinfo.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

double get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}
#else
#include <time.h>
#define CLOCKS_PER_SEC		((clock_t)1000)
double get_time()
{
	return clock() / CLOCKS_PER_SEC;
}
#endif

#define LINE_MAX			80
void print_progress(int count, int max)
{
	float progress = (float)((count * 1000)/max) / 10;
	printf("\rProgress: %4.1f%% (%3d/%3d)", progress, count, max);
	std::cout.flush();
}

void transpose(float* src, float* dst, int height, int width)
{
	for (int i=0;i<height; i++)
		for (int j=0; j<width; j++)
			dst[j*height + i] = src[i*width+j];
}

#ifdef __cplusplus
}
#endif

