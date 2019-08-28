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
#include <unistd.h>

#include "utils.h"
#include "FpgaKcu.h"

#define BRAM_ADDR				0x40000000
#define IP_ADDR 				0x43C00000
#define SIZE					64
#define MAX_SIZE 				128
#define DMA_ALIGN				256
#define OP_CODE					0x5555
#define OP_SIZE					4
#define VECTOR_SIZE				256		// SIZE * 4
#define MATRIX_SIZE				4096	// SIZE * SIZE * 4


FpgaKcu::FpgaKcu()
{
	pe_size_ = 64; 
	data_addr_ = (float*)BRAM_ADDR;
	cmd_addr_ = (unsigned int*)IP_ADDR;
	fpga_buffer_alloc();
}

void FpgaKcu::write_to_channel(char *channelDevice, void* addr, uint32_t transferSize, void* data)
{
	/* local variables */
	int rc;
	char *buffer = NULL;
	char *allocated = NULL;

	/* allocate memory to buffer */
	if (posix_memalign((void **)&allocated, 4096/*alignment*/, transferSize + 4096) != 0) {
		return;
	}
	assert(allocated);
	buffer = allocated;
	//printf("host memory buffer = %p\n", buffer);

	/* first need to copy data to buffer */
	memcpy(buffer, data, transferSize);

	int fpga_fd = open(channelDevice, O_RDWR);
	assert(fpga_fd >= 0);

	/* select AXI MM address */
	off_t off = lseek(fpga_fd, (off_t)addr, SEEK_SET);

	/* Write data to the AXI MM address using SGDMA */
	rc = write(fpga_fd, buffer, transferSize);
	assert(rc == transferSize); // make sure that the entire data is written

	close(fpga_fd);
	free(allocated);
}

/* transferSize of data at addr will be read from device to output ptr
 * returns total execution time of function
 */
//struct timespec read_from_channel(char *channelDevice, uint32_t addr, uint32_t transferSize, void *output){
void FpgaKcu::read_from_channel(char *channelDevice, void* addr, uint32_t transferSize, void *output)
{
	/* local variables */
	int rc;
	char *buffer = NULL;
	char *allocated = NULL;

	if (posix_memalign((void **)&allocated, 4096/*alignment*/, transferSize + 4096) != 0) {
		return;
	}
	assert(allocated);
	buffer = allocated;
	//printf("host memory buffer = %p\n", buffer);

	/* Open device */
	int fpga_fd = open(channelDevice, O_RDWR | O_NONBLOCK);
	assert(fpga_fd >= 0);

	/* zero-initialize buffer and read data from device */
	memset(buffer, 0x00, transferSize);
	off_t off = lseek(fpga_fd, (off_t)addr, SEEK_SET);

	rc = read(fpga_fd, buffer, transferSize);
	if ((rc > 0) && (rc < transferSize)){
		printf("Short read of %d bytes into a %d bytes buffer, could be a packet read?\n", rc, transferSize);
	}

	/* copy data from buffer to output */
	memcpy(output, buffer, transferSize);

	close(fpga_fd);
	free(allocated);
}

void FpgaKcu::fpga_buffer_alloc()
{
	mat_tile_ = new float[pe_size_*pe_size_];
	mat2t_addr_ = new float[pe_size_*pe_size_];
	vec_addr_ = new float[pe_size_];
}

void FpgaKcu::fpga_write(void* addr, uint32_t transferSize, void* data_in)
{
	write_to_channel((char*)"/dev/xdma0_h2c_0", addr, transferSize, data_in);
}

void FpgaKcu::fpga_read(void* addr, uint32_t transferSize, void* data_out)
{
	read_from_channel((char*)"/dev/xdma0_c2h_0", addr, transferSize, data_out);
}

void FpgaKcu::fpga_read_data(void* data_out)
{
	fpga_read(data_addr_, pe_size_ * sizeof(float), data_out);
}

void FpgaKcu::fpga_write_matrix(float* data_in)
{
	fpga_write(data_addr_ + sizeof(float)*pe_size_, sizeof(float) * pe_size_*pe_size_, data_in);
}

void FpgaKcu::fpga_write_vector(void* data_in, int vec_size)
{
	fpga_write((void*)data_addr_, vec_size * sizeof(float), (void*)data_in);
}

void __attribute__((optimize("O0"))) FpgaKcu::fpga_cmd_matmul()
{
	uint32_t op_code = OP_CODE;
	fpga_write((void*)cmd_addr_, OP_SIZE, (void*)&op_code);

	while (1) {
		fpga_read((void*)cmd_addr_, OP_SIZE, &op_code);
		if (op_code != OP_CODE)
			break;
	}
}

