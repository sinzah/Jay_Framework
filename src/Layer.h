#pragma once
#include "assert.h"
#include "typeinfo.h"

class Layer;
class Layer {
protected:
	float* backprop_;
	
public:
	Device*	dev_;
    float* weights_;
    float* bias_;
    float* outMatrix_;
	float* dWeights_;
    float* dBias_;

	layerinfo layer_;
    class Layer* prevLayer;
    class Layer* nextLayer;

protected:
    void MatrixMultiply(float *input, float* weight, float*output, short b_size, short i_size, short l_size);

public:
    Layer(layerinfo info, Layer* prev_layer, Device* dev);
	virtual ~Layer();

	void SetDevice(Device* dev);
    void SetNextLayer(Layer* layer);
    Layer* GetNextLayer();

	virtual int GetLayerSize() { return 0; };
	virtual void SetParameterWeight(float* value) {}
	virtual void SetParameterBias(int* value) {}
	virtual void InitializeWeightsAndBias() {};

	virtual void SetBatchSize(short size) = 0;
	virtual float* Forward(float* input, short b_size) = 0; 
	virtual void Backward(float* input, short b_size, float learning_rate) = 0;
};
