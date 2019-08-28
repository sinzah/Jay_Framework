#include <assert.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <string>

#include "utils.h"
#include "FpgaZed.h"


#define MAT_SIZE				64
#define VEC_SIZE				64
#define ADDRESS_DATA			0x40000000
#define ADDRESS_CMD				0x43c00000
#define OP_CODE					0x5555
#define OP_SIZE					4
#define PE_SIZE					64

FpgaZed::FpgaZed()
{
	pe_size_ = 64;
	fpga_buffer_alloc();
	fpga_open();
}

FpgaZed::~FpgaZed()
{
	if (fd_)
		fpga_close();
}

int FpgaZed::fpga_open()
{
	fpga_bram_size = (MAT_SIZE + 1)*(VEC_SIZE) * sizeof(float); // fpga bram data size
	fd_ = open("/dev/mem", O_RDWR);
	data_addr_ = static_cast<float*>(mmap(NULL, fpga_bram_size, 
				PROT_READ | PROT_WRITE, MAP_SHARED, fd_, ADDRESS_DATA));
	cmd_addr_ = static_cast<unsigned int*>(mmap(NULL, sizeof(unsigned int), 
				PROT_READ | PROT_WRITE, MAP_SHARED, fd_, ADDRESS_CMD));
	return fd_;
}

void FpgaZed::fpga_close()
{
	munmap(data_addr_, fpga_bram_size);
	munmap(cmd_addr_, sizeof(unsigned int));
	close(fd_);
}

void FpgaZed::fpga_buffer_alloc()
{
	mat_tile_ = new float[pe_size_*pe_size_];
	mat2t_addr_ = new float[pe_size_*pe_size_];
	vec_addr_ = new float[pe_size_];
}

void FpgaZed::fpga_write(void* addr, uint32_t transferSize, void* data_in)
{
	memcpy(addr, data_in, transferSize);
}

void FpgaZed::fpga_read(void* addr, uint32_t transferSize, void* data_out)
{
	memcpy(data_out, addr, transferSize);
}

void FpgaZed::fpga_read_data(void* data_out)
{
	fpga_read((void*)data_addr_, pe_size_ * sizeof(float), data_out);
}

void FpgaZed::fpga_write_matrix(float* data_in)
{
	int i;
	for (i=0; i<pe_size_; i++)
		fpga_write((void*)(&data_addr_[pe_size_+i*pe_size_]), pe_size_ * sizeof(float), (void*)&data_in[i*pe_size_]);
}

void FpgaZed::fpga_write_vector(void* data_in, int vec_size)
{
	fpga_write((void*)data_addr_, vec_size * sizeof(float), (void*)data_in);
}

void __attribute__((optimize("O0"))) FpgaZed::fpga_cmd_matmul()
{
	uint32_t op_code = OP_CODE;
	fpga_write((void*)cmd_addr_, OP_SIZE, (void*)&op_code);

	while (1) {
		fpga_read((void*)cmd_addr_, OP_SIZE, &op_code);
		if (op_code != OP_CODE)
			break;
	}
}
