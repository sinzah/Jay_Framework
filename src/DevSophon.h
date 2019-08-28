#pragma once

#include "Device.h"

#include "bmlib_runtime.h"
#include "bmdnn_api.h"

class Sophon : public Device {
protected:
	bm_handle_t handle_;
	int deviceID_;

	void tiled_matmul(float *in_matrix1, float *in_matrix2, float *out_matrix,
		int r1_size, int r2_size, int row_size, int common_size, int col_size);
public:
	int pe_size_;

	float *mat_tile_;
	float *mat2t_addr_;
	float *vec_addr_;

	Sophon();
	~Sophon();

	int alloc_MatrixBuffer();
	void mat_mul(float *in_matrix1, float *in_matrix2, float *out_matrix,
		int r_size, int m_size, int c_size);
};
