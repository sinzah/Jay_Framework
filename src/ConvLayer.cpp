#include <stdlib.h>
#include <memory.h>
#include <random>
#include "ConvLayer.h"
#include "common.h"
#include "im2col.h"
#include <time.h>
#include "utils.h"

#ifdef WEIGHT_INIT_STD
#undef WEIGHT_INIT_STD
#endif
#define WEIGHT_INIT_STD 0.1

using namespace std;

void ConvLayer::im2col(const float *img, float *col)
{
	conv_t c = layer_.conv;

	__im2col<float>(img, col, c.w, c.h, c.c, c.fw, c.fh, c.p, c.p, c.s, c.s);
}
void ConvLayer::col2im(const float *col, float *img)
{
	conv_t c = layer_.conv;

	__col2im<float>(col, img, c.w, c.h, c.c, c.fw, c.fh, c.p, c.p, c.s, c.s);
}

ConvLayer::ConvLayer(layerinfo info, Layer* prevLayer, Device* dev): Layer(info, prevLayer, dev)
{
	SetOutputImageSize();
	if (info.conv.c != 0)
	{
		AllocateMemoryforWeightsAndBias();
		InitializeWeightsAndBias();
	}

	inMatrix_ = nullptr;
	outMatrixT_ = nullptr;

	outMatrixT_ = new float[output_size_];
}

ConvLayer::~ConvLayer()
{
	if (inMatrix_ != nullptr)
		delete [] inMatrix_;
	if (outMatrixT_ != nullptr)
		delete [] outMatrixT_;
	if (dWT_ != nullptr)
		delete [] dWT_;
	if (unitDW_ != nullptr)
		delete [] unitDW_;
}

void ConvLayer::SetOutputImageSize()
{
	conv_t c = layer_.conv;

	oh_ = (c.h + 2*c.p - c.fh) / c.s + 1;	// Height: (H+2P-FH)/S+1
	ow_ = (c.w + 2*c.p - c.fw) / c.s + 1;	// Width: (W+2P-FW)/S+1

	input_size_ = c.c * c.h * c.w; // C*H*W
	output_size_ = c.fn * oh_ * ow_; // FN*OH*OW
	im2col_size_ = (c.c*c.fh*c.fw) * (oh_*ow_);  // (C*FH*FW, OH*OW)
}

void ConvLayer::AllocateMemoryforWeightsAndBias()
{
	int height = layer_.conv.fn; // FN
	int width = layer_.conv.c * layer_.conv.fh * layer_.conv.fw;  // C*FH*FW
	weight_size_ = height * width; // (FN) * (C*FH*FW)

	if (weights_ != nullptr)
		delete [] weights_;
	weights_ = new float[weight_size_];

	if (bias_ != nullptr)
		delete [] bias_;
	bias_ = new float[layer_.conv.fn];
}

void ConvLayer::InitializeWeightsAndBias()
{
	mt19937 gen(time(NULL));
	normal_distribution<> nd(0.0, 1.0);

	// Initialize Weights
	for (int i=0;i<weight_size_;i++)
		weights_[i] = (float)nd(gen) * WEIGHT_INIT_STD;

	// Initialize Bias
	for (short i=0; i<layer_.conv.fn; i++)
		bias_[i] = 0;
}

void ConvLayer::SetParameterWeight(float* value)
{
	if (weights_ != nullptr)
		delete[] weights_;
	weights_ = std::move(value);
	//memcpy(weights, value, sizeof(float)* weight_size_);
}

void ConvLayer::SetParameterBias(int* value)
{
	// Each output channel has a bias
	memcpy(bias_, value, sizeof(float) * layer_.conv.fn);
}

void ConvLayer::SetBatchSize(short b_size)
{
	if (inMatrix_ != nullptr)
		delete[] inMatrix_;
	inMatrix_ = new float[b_size * im2col_size_];
	assert(inMatrix_ != nullptr);

	if (outMatrix_ != nullptr)
		delete[] outMatrix_;

	outMatrix_ = new float[b_size * output_size_]; // batch*(FN*OH*OW)
	assert(outMatrix_ != nullptr);
	if (nextLayer != nullptr)
		nextLayer->SetBatchSize(b_size);

}

int ConvLayer::GetLayerSize()
{
	return output_size_;
}

/*
P : padding
S : stride
C : input channel 
H : input height
W : input width
OH : output height (H+2P-FH)/S + 1
FH : filter height (W+2P-FW)/S + 1
FW : filter width
FN : output channel (filter number)

Matrix1 : Weights (Filter) (FN, C*FH*FW)
Matrix2 : Im2col of input data (C*FH*FW, OH*OW)
Output  : Output (FN, OH*OW)
 */

float* ConvLayer::Forward(float* input, short b_size)
{
	float* retPtr = outMatrix_;
	conv_t c = layer_.conv;

	// Matrix Multiplication:  W x X(im2col)
	for (short i=0; i<b_size; i++) {
		// convert input to im2col
		im2col(&input[i*input_size_], &inMatrix_[i*im2col_size_]);
		MatrixMultiply(weights_, &inMatrix_[i*im2col_size_], outMatrixT_,
				c.fn, c.c*c.fh*c.fw, oh_*ow_);
		for(short j=0; j<c.fn; j++) {
			if (bias_[j] != 0)
				for (int k=0; k< oh_*ow_;k++)
					outMatrixT_[j*oh_*ow_ + k] += bias_[j];
		}
		// Use Tensorflow format for using some test dataset (FN,C*OH*OW) -> (C*OH*OW, FN)
		transpose(outMatrixT_, &outMatrix_[i*output_size_], c.fn, oh_*ow_);
	}

	if (nextLayer != nullptr)
		retPtr = nextLayer->Forward(outMatrix_, b_size);

	return retPtr;
}

void ConvLayer::Backward(float* delta, short b_size, float learning_rate)
{
	conv_t c = layer_.conv;

	// Backprop will be calculated unless the previous layer is input layer
	if (prevLayer && prevLayer->prevLayer)
	{
		if (backprop_ == nullptr)
			backprop_ = new float[b_size*input_size_];
		memset(backprop_, 0, sizeof(float)*b_size*input_size_);

		float* tmpMatrix = new float[(c.c*c.fh*c.fw) * (oh_*ow_)];  // temp matrix for col2im
		for (int i=0; i<b_size; i++)
		{
			MatrixMultiply(&delta[i*output_size_], weights_, tmpMatrix, oh_*ow_, c.fn, c.c*c.fh*c.fw);
			col2im(tmpMatrix, &backprop_[i*input_size_]);
		}
		delete [] tmpMatrix;
	}

	if (dWeights_ == nullptr)
	{
		dWeights_ = new float[weight_size_];
		dBias_ = new float[c.fn];
		dWT_ = new float[weight_size_];
		unitDW_ = new float[weight_size_];
	}
	memset(dWeights_, 0, sizeof(float) * weight_size_);
	memset(dBias_, 0, sizeof(float) * c.fn);

	for (int i=0;i<b_size;i++)
	{
		MatrixMultiply(&inMatrix_[i*im2col_size_], &delta[i*output_size_], dWT_, c.c*c.fh*c.fw, oh_*ow_, c.fn);
		transpose(dWT_, unitDW_, c.c*c.fh*c.fw, c.fn);

		// Weight accumulation of the batch
		for (int j=0; j<weight_size_; j++)
			dWeights_[j] += unitDW_[j];

		// Bias accumulation of the batch
		for (int k=0; k < oh_*ow_; k++)
			for (int j=0; j < c.fn;j++)
				dBias_[j] += delta[i*output_size_ + k*c.fn + j];
	}

	// Weight Adjustification
	for (int j=0; j<weight_size_; j++)
		weights_[j] -= (dWeights_[j]/b_size) * learning_rate;

	// Bias Adjustification
	for (short j=0; j<c.fn;j++) 
		bias_[j] -= (dBias_[j]/b_size) * learning_rate;

	if (prevLayer && prevLayer->prevLayer) {
		prevLayer->Backward(backprop_, b_size, learning_rate);
	}
}
