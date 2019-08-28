#pragma once
#include "utils.h"
#include <iostream>

class Device {
public:
	const char* devname_;

	Device() {}	
	virtual ~Device() {}

	int fpga_open();
	
	virtual void mat_mul(float *in_matrix1, float *in_matrix2, float *out_matrix,
		int r_size, int m_size, int c_size) = 0;
};
