#include <stdlib.h>
#include <memory.h>
#include <random>
#include "Layer.h"
#include "common.h"

using namespace std;

Layer::Layer(layerinfo info , Layer* prev_layer, Device* dev=nullptr): dev_(dev), layer_(info)
{
	weights_ = nullptr;
	bias_ = nullptr;
	prevLayer = prev_layer;
	nextLayer = nullptr;
	outMatrix_ = nullptr;
	dWeights_ = nullptr;
	dBias_ = nullptr;
	backprop_ = nullptr;
	dev_ = dev;
}
Layer::~Layer()
{
	if (weights_ != nullptr)
		delete[] weights_;
	if (bias_ != nullptr)        
		delete[] bias_;
	if (dWeights_ != nullptr)
		delete[] dWeights_;
	if (dBias_ != nullptr)
		delete[] dBias_;
	if (outMatrix_ != nullptr)
		delete[] outMatrix_;
	if (backprop_ == nullptr)
		delete[] backprop_;	
}

void Layer::SetDevice(Device* dev)
{
	dev_ = dev;
}
Layer* Layer::GetNextLayer()
{
	return nextLayer;
}
void Layer::SetNextLayer(Layer* layer)
{
	// Currently only support appending to the end, not inserting amid.
	assert(nextLayer == nullptr);
	nextLayer = layer;
}

void Layer::MatrixMultiply(float *mat1, float* mat2, float* output, short r_size, short m_size, short c_size)
{
#if 0
#ifdef __FPGA
	fpga_mat_mul(input, weight, output, r_size, m_size, c_size);
#else
	memset(output, 0, r_size*c_size*sizeof(float));

	for (int i = 0; i < r_size; i++) {
		for (int k = 0; k < m_size; k++)
			for (int j = 0; j < c_size; j++)
				output[i * c_size + j] += input[i * m_size + k] * weight[k * c_size + j];
	}
#endif
#else
	if (dev_ != nullptr)
		dev_->mat_mul(mat1, mat2, output, r_size, m_size, c_size);
#endif
}
