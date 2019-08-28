#pragma once

#include "Device.h"

class DevCpu : public Device {
protected:
	void tiled_matmul(float *in_matrix1, float *in_matrix2, float *out_matrix,
		int r_size, int m_size, int c_size, int tile_size);
public:
	DevCpu() {}
	~DevCpu() {}

	void mat_mul(float *in_matrix1, float *in_matrix2, float *out_matrix,
		int r_size, int m_size, int c_size);
};
