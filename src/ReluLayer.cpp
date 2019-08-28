#include <math.h>
#include <memory.h>
#include <assert.h>
#include <algorithm>
#include "ReluLayer.h"

#define max(a,b)		((a>b) ? a:b)

using namespace std;

ReluLayer::ReluLayer(layerinfo info, Layer* prevLayer, Device* dev = nullptr): Layer(info, prevLayer, dev) 
{
	mask_ = nullptr;
}

ReluLayer::~ReluLayer()
{
	if (mask_ != nullptr)
		delete [] mask_;
}

void ReluLayer::SetBatchSize(short b_size)
{
	if (outMatrix_ != nullptr)
		delete[] outMatrix_;
	outMatrix_ = new float[b_size * layer_.relu.layer_size];

	if (mask_ != nullptr)
		delete[] mask_;
	mask_ = new BYTE[b_size * layer_.relu.layer_size];
	memset(mask_, 0, sizeof(BYTE)*b_size*layer_.relu.layer_size);

	assert(outMatrix_ != nullptr);
	assert(mask_ != nullptr);

	if (nextLayer != nullptr)
		nextLayer->SetBatchSize(b_size);

}

float* ReluLayer::Forward(float* input, short size)
{
	float* retPtr = outMatrix_;
	int layer_size = layer_.relu.layer_size;

	memcpy(outMatrix_, input, sizeof(float) * size * layer_size);
	for (int i=0; i<size * layer_size; i++)
		if (input[i] <= 0) {
			outMatrix_[i] = 0;
			mask_[i] = 1;
		}

	if (nextLayer != nullptr)
		retPtr = nextLayer->Forward(outMatrix_, size);

	return retPtr;
}

void ReluLayer::Backward(float* delta, short b_size, float learning_rate)
{
	short layer_size = layer_.relu.layer_size;

	// Backprop will be calculated unless the previous layer is not the input layer
	if (prevLayer && prevLayer->prevLayer)
	{
		for (int i = 0; i < b_size * layer_size; i++) {
			if (mask_[i])
				delta[i] = 0;
		}
		memset(mask_, 0, sizeof(BYTE)*b_size*layer_.relu.layer_size);
		prevLayer->Backward(delta, b_size, learning_rate);
	}
}
