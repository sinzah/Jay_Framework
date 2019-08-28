#pragma once

#include "Device.h"
#include "DevFpga.h"

class FpgaZed: public DevFpga {
protected:	
	int fpga_bram_size;

public:
	FpgaZed();
	~FpgaZed();
	
	int fpga_open();
	void fpga_close();

	void fpga_buffer_alloc();
	void fpga_write(void* addr, uint32_t transferSize, void* data_in);
	void fpga_read(void* addr, uint32_t transferSize, void* data_out);
	void fpga_read_data(void* data_out);
	void fpga_write_matrix(float* data_in);
	void fpga_write_vector(void* data_in, int vec_size);
	void __attribute__((optimize("O0"))) fpga_cmd_matmul();
};
