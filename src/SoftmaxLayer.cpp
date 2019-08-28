#include <math.h>
#include <assert.h>
#include <algorithm>
#include "SoftmaxLayer.h"

using namespace std;

SoftmaxLayer::SoftmaxLayer(layerinfo info) : Layer(info, nullptr, nullptr)
{

}

SoftmaxLayer::~SoftmaxLayer()
{

}

void SoftmaxLayer::SetBatchSize(short b_size)
{
	if (backprop_ != nullptr)
		delete [] backprop_;

	backprop_ = new float[b_size * layer_.softmax.layer_size];
	assert(backprop_ != nullptr);

	if (outMatrix_ != nullptr)
		delete[] outMatrix_;

	outMatrix_ = new float[b_size * layer_.softmax.layer_size];
	assert(outMatrix_ != nullptr);

}

float* SoftmaxLayer::Forward(float* input, short size)
{
	int layer_size = layer_.softmax.layer_size;

	for (short i=0;i<size;i++) {
		float max = *max_element(&input[i*layer_size], &input[i*layer_size + layer_size]);
		float exp_sum = 0;
		for (short j=0; j<layer_size; j++) {
			outMatrix_[i*layer_size + j] = (float)exp(input[i*layer_size + j] - max);
			exp_sum += outMatrix_[i*layer_size + j];
		}
		for (int j=0; j<layer_size; j++)
			outMatrix_[i*layer_size + j] /= exp_sum;
	}

	return outMatrix_;
}

float* SoftmaxLayer::GetDifferential(float* delta, short b_size, float learning_rate)
{
	int layer_size = layer_.softmax.layer_size;

	int* label = (int*)delta;
	for (short i=0;i< b_size; i++) {
		for (short j=0; j<layer_size; j++) {
			backprop_[i*layer_size + j] = (outMatrix_[i*layer_size + j] - label[i*layer_size + j]) / layer_size;
		}
	}

	return backprop_;
}
