#pragma once

#include "Device.h"

#include <sys/types.h>

class DevFpga: public Device {
protected:
	int fd_;
	float* data_addr_;
	unsigned int* cmd_addr_;
	
	
	void tiled_matmul(float *in_matrix1, float *in_matrix2, float *out_matrix,
	int r1_size, int r2_size, int row_size, int common_size, int col_size);

public:
	float* mat_tile_;
	float* mat2t_addr_;
	float* vec_addr_;
	short pe_size_;
	
	DevFpga();
	~DevFpga();

	virtual int fpga_open() { return fd_; }
	virtual void fpga_close() {}

	virtual void fpga_buffer_alloc() = 0;
	virtual void fpga_write(void* addr, uint32_t transferSize, void* data_in) = 0;
	virtual void fpga_read(void* addr, uint32_t transferSize, void* data_out) = 0;
	virtual void fpga_read_data(void* data_out) = 0;
	virtual void fpga_write_matrix(float* data_in) = 0;
	virtual void fpga_write_vector(void* data_in, int vec_size) = 0;
	virtual void fpga_cmd_matmul() = 0;

	void mat_mul(float *in_matrix1, float *in_matrix2, float *out_matrix,
		int r_size, int m_size, int c_size);
};
