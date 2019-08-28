#pragma once

//#define __FPGA
//#define __ZEDBOARD

extern "C" {
#ifdef __ZEDBOARD
	void fpga_open();
	void fpga_close();
#endif
	void fpga_mat_mul(float *in_matrix1, float *in_matrix2, float *out_matrix, int r_size, int m_size, int c_size);
	void cpu_mat_mul(float *in_matrix1, float *in_matrix2, float *out_matrix, int r_size, int m_size, int c_size);
}
