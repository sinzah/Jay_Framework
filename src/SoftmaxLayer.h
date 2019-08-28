#pragma once

#include "typeinfo.h"
#include "Layer.h"

class SoftmaxLayer : public Layer {
protected:
	float* backprop_;
	
public:
	SoftmaxLayer(layerinfo info);
	~SoftmaxLayer();
	void Backward(float* input, short b_size, float learning_rate) {}

	void SetBatchSize(short b_size);
    float* Forward(float* input, short size);
    float* GetDifferential(float* delta, short b_size, float learning_rate);
};
