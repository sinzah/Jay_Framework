#include <memory.h>
#include "DevSophon.h"
#include "utils.h"

Sophon::Sophon()
{
	deviceID_ = 0;
	handle_ = nullptr;
	pe_size_ = 8;

	if (bm_dev_request(&handle_, deviceID_) != BM_SUCCESS) {
		handle_ = nullptr;
		DEBUG("Sophon BM device request failed");
	}
	DEBUG("Sophon BM device opened");

	if (alloc_MatrixBuffer() < 0) {
		DEBUG("Failed to memory allocation matrix for tiling");
	}
}

int Sophon::alloc_MatrixBuffer()
{
	mat_tile_ = new float[pe_size_ * pe_size_];
	if (mat_tile_ == nullptr) {
		return -1;
	}
	mat2t_addr_ = new float[pe_size_ * pe_size_];
	if (mat2t_addr_ == nullptr) {
		return -2;
	}
	vec_addr_ = new float[pe_size_];
	if (vec_addr_ == nullptr) {
		return -3;
	}

	return 0;
}

Sophon::~Sophon()
{
	if (handle_ != nullptr) {
		bm_dev_free(handle_);
		DEBUG("Sophon BM device freed");
	}

	if (mat_tile_)
		delete [] mat_tile_;

	if (mat2t_addr_)
		delete [] mat2t_addr_;

	if (vec_addr_)
		delete [] vec_addr_;
}

void Sophon::tiled_matmul(float *in_matrix1, float *in_matrix2, float *out_matrix,
			int r1_size, int r2_size, int row_size, int common_size, int col_size)
{
	int ret;
	int i, j, k;

	memset(mat2t_addr_, 0, sizeof(float)*pe_size_*pe_size_);
	memset(vec_addr_, 0, sizeof(float)*pe_size_);

	/* Transpose matrix B (Weight) */
	for (i = 0; i < common_size; i++)
		for (j = 0; j < col_size; j++)
			mat2t_addr_[j*pe_size_ + i] = in_matrix2[i*r2_size + j];

	for (k = 0; k < row_size; k++)
	{
		memcpy(vec_addr_, in_matrix1 + r1_size * k, common_size * sizeof(float));

		ret = bmdnn_fc_forward(handle_,
				bm_mem_from_system((void *)vec_addr_),       // Input
				bm_mem_from_system((void *)mat2t_addr_),     // Weight
				bm_mem_from_system((void *)BM_MEM_ADDR_NULL),// Bias
				1,                              // Batch size
				MEMSIZE(row_size * col_size),    // Output size
				MEMSIZE(row_size),            // Input size
				0,            // Transpose
				0,            // Using Bias
				0,            // Using Relu
				bm_mem_from_system((void *)(out_matrix + (pe_size_*k))) );
		if (ret != BM_SUCCESS) {
			DEBUG("Failed to run mat_mul on Sophon " << ret);
			break;
		}
	}
}

void Sophon::mat_mul(float *in_matrix1, float *in_matrix2, float *out_matrix,
		int r_size, int m_size, int c_size)
{
#if 0
	int i, j, k, ii, jj;
	int tile_size = pe_size_;
	int col_size = pe_size_;
	int row_size = pe_size_;
	int common_size = pe_size_;

	for (i = 0; i < r_size; i += tile_size)
	{
		row_size = MIN(tile_size, r_size - i);
		for (j = 0; j < c_size; j += tile_size)
		{
			col_size = MIN(tile_size, c_size - j);
			memset(mat_tile_, 0, sizeof(float)*tile_size*tile_size);

			// Initialize outmatrix with zero value
			for (ii = 0; ii < row_size; ii++)
				memset(out_matrix + (i + ii)*c_size + j, 0, sizeof(float)*col_size);

			for (k = 0; k < m_size; k += tile_size)
			{
				common_size = MIN(tile_size, m_size - k);

				tiled_matmul(in_matrix1 + i * m_size + k, in_matrix2 + k * c_size + j,
						(float*)mat_tile_, m_size, c_size, row_size, common_size, col_size);
				for (ii = 0; ii < row_size; ii++)
					for (jj = 0; jj < col_size; jj++)
						out_matrix[(i + ii)*c_size + j + jj] += mat_tile_[ii*tile_size + jj];
			}
		}
	}
#else
	int ret;

	memset(out_matrix, 0, r_size*c_size*sizeof(float));

	ret = bmdnn_fc_forward(handle_,
			bm_mem_from_system((void *)in_matrix1),      // Input
			bm_mem_from_system((void *)in_matrix2),      // Weight
			bm_mem_from_system((void *)BM_MEM_ADDR_NULL),// Bias
			1,                              // Batch size
			MEMSIZE(r_size * c_size),              // Output size
			MEMSIZE(r_size * m_size),              // Input size
			1,            // Transpose
			0,            // Using Bias
			0,            // Using Relu
			bm_mem_from_system((void *)out_matrix) );
#endif
}
