#include <math.h>
#include <memory.h>
#include <assert.h>
#include <algorithm>
#include "InputLayer.h"

using namespace std;

InputLayer::~InputLayer()
{
}

void InputLayer::SetBatchSize(short b_size)
{
	if (outMatrix_ != nullptr)
		delete[] outMatrix_;

	outMatrix_ = new float[b_size * layer_.start.input_size];
	assert(outMatrix_ != nullptr);
	if (nextLayer != nullptr)
		nextLayer->SetBatchSize(b_size);

}

float* InputLayer::Forward(float* input, short size)
{
	float* retPtr = outMatrix_;

	memcpy(outMatrix_, input, sizeof(float)*size*layer_.start.input_size);
	if (nextLayer != nullptr)
		retPtr = nextLayer->Forward(outMatrix_, size);

	return retPtr;
}
