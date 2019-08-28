#pragma once
#include "assert.h"
#include "Layer.h"

class FCLayer;
class FCLayer: public Layer {
protected:
	float* weightT_;
	float* inputT_;

public:
	FCLayer(layerinfo info, Layer* prevLayer, Device* dev);
	~FCLayer();

	void SetParameterWeight(float* value);
	void SetParameterBias(int* value);
	void AllocateMemoryforWeightsAndBias();
	void InitializeWeightsAndBias();
	int GetLayerSize();
	void SetBatchSize(short size);

    float* Forward(float* input, short b_size);
	void Backward(float* delta, short b_size, float learning_rate);
};
