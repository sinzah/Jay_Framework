#pragma once
#include "assert.h"
#include "Layer.h"
#include "SoftmaxLayer.h"
#include "ReluLayer.h"
#include "ConvLayer.h"
#include "Device.h"

class DnnFrmwk {
protected:
	int input_size;
	int output_size;
	Device* dev_;

public:
	Layer* inputLayer;   
	Layer* lastLayer;
	SoftmaxLayer* softmaxLayer;

	Layer* SetInputLayer(int size);
	Layer* AddFCLayer(int size) ;
	Layer* AddConvLayer(short in_channel, short input_height, short input_width, 
		short out_channel, short filter_size, short padding = 0, short stride = 1);
	Layer* AddReluLayer();
	Layer* AddSoftmaxLayer();

	void SetBatchSize(int b_size);
	
	float* Predict(float* data, int size, bool training);
	float CalculateLoss(float* data, int* label, int size);
	float CrossEntropyError(float* prediction, int* label, int size);
	void BackPropagation(int* label, int b_size, float learining_rate);
	int Train(float* data, int* label, int b_size, float learining_rate);
	float GetAccuracy(float* data, int* label, int data_size);

public:
	DnnFrmwk(dev_type dev);
	~DnnFrmwk();    
};
