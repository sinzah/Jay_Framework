#include <stdlib.h>
#include <memory.h>
#include <random>
#include "FCLayer.h"
#include "common.h"
#include <time.h>

#define WEIGHT_INIT_STD 0.1

using namespace std;

FCLayer::FCLayer(layerinfo info, Layer* prevLayer, Device* dev): 	Layer(info, prevLayer, dev)
{
	short input_size = info.fc.input_size;
	short layer_size = info.fc.layer_size;
	weightT_ = nullptr;

	if (input_size != 0)
	{
		AllocateMemoryforWeightsAndBias();
		InitializeWeightsAndBias();
		weightT_ = new float[input_size*layer_size];
	}
}

FCLayer::~FCLayer()
{
	if (weightT_ != nullptr)
		delete [] weightT_;
	if (inputT_ != nullptr)
		delete [] inputT_;
}

void FCLayer::SetParameterWeight(float* value)
{
	if (weights_ != nullptr)
		delete[] weights_;
	weights_ = new float[layer_.fc.input_size * layer_.fc.layer_size];
	transpose(value, weights_, layer_.fc.layer_size, layer_.fc.input_size);
}

void FCLayer::SetParameterBias(int* value)
{
	memcpy(bias_, value, sizeof(float)* layer_.fc.layer_size);
}

void FCLayer::AllocateMemoryforWeightsAndBias()
{
	if (weights_ != nullptr)
		delete [] weights_;
	weights_ = new float[layer_.fc.input_size * layer_.fc.layer_size];

	if (bias_ != nullptr)
		delete [] bias_;
	bias_ = new float[layer_.fc.layer_size];
}

void FCLayer::InitializeWeightsAndBias()
{
	mt19937 gen(time(NULL));
	normal_distribution<> nd(0.0, 1.0);

	// Initialize Weights
	for (short i=0;i<layer_.fc.input_size;i++)
		for (short j=0;j<layer_.fc.layer_size;j++)
			weights_[i*layer_.fc.layer_size + j] = (float)nd(gen) * WEIGHT_INIT_STD;

	// Initialize Bias
	for (short i=0; i<layer_.fc.layer_size; i++)
		bias_[i] = 0;
}

int FCLayer::GetLayerSize()
{
	return layer_.fc.layer_size;
}

void FCLayer::SetBatchSize(short b_size)
{
	if (outMatrix_ != nullptr)
		delete[] outMatrix_;

	outMatrix_ = new float[b_size * layer_.fc.layer_size];
	assert(outMatrix_ != nullptr);
	if (nextLayer != nullptr)
		nextLayer->SetBatchSize(b_size);
}

extern "C" void transpose(float* src, float* dst, int height, int width);

float* FCLayer::Forward(float* input, short b_size)
{
	fc_t fc = layer_.fc;
	float* retPtr = outMatrix_;
	MatrixMultiply(input, weights_, outMatrix_, b_size, fc.input_size, fc.layer_size);

	for (int i=0;i<b_size;i++)
		for (int j=0;j<fc.layer_size;j++)
			outMatrix_[i*fc.layer_size+j] += bias_[j];
	if (nextLayer != nullptr)
		retPtr = nextLayer->Forward(outMatrix_, b_size);

	return retPtr;
}

void FCLayer::Backward(float* delta, short b_size, float learning_rate)
{
	fc_t fc = layer_.fc;
	short input_size = fc.input_size;
	short layer_size = fc.layer_size;

	// Backprop will be calculated unless the previous layer is input layer
	if (prevLayer && prevLayer->prevLayer)
	{
		if (backprop_ == nullptr)
			backprop_ = new float[b_size*input_size];
		memset(backprop_, 0, sizeof(float)*b_size*input_size);
		MatrixMultiply(delta, weights_, backprop_, b_size, layer_size, input_size);
	}

	// Apply justification to W and B in this layer
	if (dWeights_ == nullptr)
	{
		dWeights_ = new float[input_size * layer_size];
		dBias_ = new float[layer_size];
		inputT_ = new float[b_size*input_size];
		assert(inputT_ != nullptr);
	}


	transpose(prevLayer->outMatrix_, inputT_, b_size, input_size);
	MatrixMultiply(inputT_, delta, dWeights_, input_size, b_size, layer_size);

	memset(dBias_, 0, layer_size * sizeof(float));
	for (short i = 0; i < b_size; i++)
		for (short j = 0; j < layer_size; j++)
			dBias_[j] += delta[i*layer_size + j];

	if (prevLayer && prevLayer->prevLayer) {
		prevLayer->Backward(backprop_, b_size, learning_rate);
	}

	for (int i = 0; i < input_size * layer_size; i++)
		weights_[i] -= dWeights_[i] * learning_rate;

	for (short i = 0; i < layer_size; i++)
		bias_[i] -= dBias_[i] * learning_rate;
}

