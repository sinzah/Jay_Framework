#include <memory.h>
#include "DevFpga.h"
#include "utils.h"

double time_fpga_transpose = 0;
double time_fpga_write_matrix = 0;
double time_fpga_write_vectors = 0;
double time_fpga_read_matrix = 0;
double time_fpga_tiling_matrix = 0;
double time_fpga_calculation = 0;

DevFpga::DevFpga()
{
	fd_ = 0;
	pe_size_ = 0;
	mat_tile_ = nullptr;
	mat2t_addr_ = nullptr;
	vec_addr_ = nullptr;
}

DevFpga::~DevFpga()
{
	if (mat_tile_ != nullptr)
		delete [] mat_tile_;
	if (mat2t_addr_ != nullptr)
		delete [] mat2t_addr_;
	if (vec_addr_ != nullptr)
		delete [] vec_addr_;
}


// Matrix Multiplication on  FPGA
	void DevFpga::tiled_matmul
(float *in_matrix1, float *in_matrix2, float *out_matrix,
 int r1_size, int r2_size, int row_size, int common_size, int col_size)
{
	int i, j, k;
	memset(mat2t_addr_, 0, sizeof(float)*pe_size_*pe_size_);
	memset(vec_addr_, 0, sizeof(float)*pe_size_);

	double start, end;

	start = get_time();
	/* Transpose matrix B */
	for (i = 0; i < common_size; i++)
		for (j = 0; j < col_size; j++)
			mat2t_addr_[j*pe_size_ + i] = in_matrix2[i*r2_size + j];
	end = get_time();
	time_fpga_transpose += (end - start);

	start = get_time();
	/* Write transposed matrix B to BRAM */
	fpga_write_matrix(mat2t_addr_);
	end = get_time();
	time_fpga_write_matrix += (end - start);

	for (k = 0; k < row_size; k++)
	{
		memcpy(vec_addr_, in_matrix1 + r1_size * k, common_size * sizeof(float));

		/* Write kth row of matrix A to BRAM */
		start = get_time();
		fpga_write_vector((void*)vec_addr_, MIN(col_size, pe_size_));
		end = get_time();
		time_fpga_write_vectors += (end - start);

		start = get_time();
		fpga_cmd_matmul();
		end = get_time();
		time_fpga_calculation += (end - start);

		/* Read kth row of output matrix from BRAM */
		start = get_time();
		fpga_read_data((void*)(out_matrix + (pe_size_ * k)));
		end = get_time();
		time_fpga_read_matrix += (end - start);
	}
}

void DevFpga::mat_mul(float *in_matrix1, float *in_matrix2, float *out_matrix,
		int r_size, int m_size, int c_size)
{
	int i, j, k, ii, jj;
	int tile_size = pe_size_;
	double start, end;
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

				start = get_time();
				tiled_matmul(in_matrix1 + i * m_size + k, in_matrix2 + k * c_size + j,
						(float*)mat_tile_, m_size, c_size, row_size, common_size, col_size);
				end = get_time();
				time_fpga_tiling_matrix += (end - start);

				for (ii = 0; ii < row_size; ii++)
					for (jj = 0; jj < col_size; jj++)
						out_matrix[(i + ii)*c_size + j + jj] += mat_tile_[ii*tile_size + jj];
			}
		}
	}

}

