#pragma once

#include "Layer.h"

class InputLayer : public Layer {
public:
	InputLayer(layerinfo info, Device* dev=nullptr): Layer(info, nullptr, dev) {}
	~InputLayer();

	void SetBatchSize(short size);
	
    float* Forward(float* input, short size);
    void Backward(float* input, short b_size, float learning_rate) {}
};

