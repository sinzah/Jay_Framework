#pragma once

#include "typeinfo.h"
#include "Layer.h"

class ReluLayer : public Layer {
protected:
	BYTE * mask_;
public:
	ReluLayer(layerinfo info, Layer* prevLayer, Device* dev);
	~ReluLayer();

	void SetBatchSize(short size);
    float* Forward(float* input, short size);
    void Backward(float* delta, short b_size, float learning_rate);
};

