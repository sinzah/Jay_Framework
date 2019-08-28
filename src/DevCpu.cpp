#include <memory.h>
#include "DevCpu.h"
#include "utils.h"

void DevCpu::tiled_matmul(float *in_matrix1, float *in_matrix2, float *out_matrix,
		int r_size, int m_size, int c_size, int tile_size)
{
	int i,j,k, ii, ij, ik;

	for (i=0;i<r_size;i+=tile_size)
		for(j=0;j<c_size;j+=tile_size)
		{
			out_matrix[i*c_size+j] = 0;
			for (k=0;k<m_size;k+=tile_size)
				for (ii=0;ii<MIN(tile_size, r_size-i);ii++)
					for (ij=0;ij<MIN(tile_size, c_size-j);ij++)
						for(ik=0;ik<MIN(tile_size, m_size-k);ik++)
							out_matrix[(i+ii)*c_size+j+ij] += in_matrix1[(i+ii)*m_size + k+ik] * in_matrix2[(k+ik)*c_size + j+ij];
		}
}

void DevCpu::mat_mul(float *in_matrix1, float *in_matrix2, float *out_matrix,
		int r_size, int m_size, int c_size)
{
	//cpu_tiled_matmul(in_matrix1, in_matrix2, out_matrix, r_size, m_size, c_size, 16);
	//printf("\nAddress of mat1, mat2, out is %p, %p, %p\n", in_matrix1, in_matrix2, out_matrix);

	memset(out_matrix, 0, r_size*c_size*sizeof(float));
	for (int i=0; i<r_size; i++)
		for (int k=0; k<m_size; k++)
			for (int j=0; j<c_size; j++)
				out_matrix[i*c_size+j] += in_matrix1[i*m_size+k] * in_matrix2[k*c_size+j];
}
