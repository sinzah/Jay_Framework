#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

#include "typeinfo.h"
#include "Layer.h"
#include "DnnFrmwk.h"
#include "SoftmaxLayer.h"
#include "FCLayer.h"
#include "ConvLayer.h"
#include "InputLayer.h"

#include "Device.h"
#include "DevCpu.h"
#include "DevFpga.h"
#include "FpgaZed.h"
#include "FpgaKcu.h"
#include "DevSophon.h"

using namespace std;

DnnFrmwk::DnnFrmwk(dev_type dev = DEV_CPU)
{
	dev_ = nullptr;
	input_size = 0;
	output_size = 0;
	inputLayer = lastLayer = nullptr;
	softmaxLayer = nullptr;
	switch (dev) {
		case DEV_FPGA_ZED:
			dev_ = new FpgaZed();
			break;
		case DEV_FPGA_KCU1500:
			dev_ = new FpgaKcu();
			break;
		case DEV_SOPHON:
			dev_ = new Sophon();
			break;
		default:
			dev_ = new DevCpu();
			break;
	}
}

DnnFrmwk::~DnnFrmwk()
{
	while (inputLayer) {
		Layer* ptLayer = inputLayer->GetNextLayer();
		delete inputLayer;
		inputLayer = ptLayer;
	}
	if (softmaxLayer != nullptr)
		delete softmaxLayer;
	if (dev_ != nullptr)
		delete dev_;
}

Layer* DnnFrmwk::SetInputLayer(int size)
{
	// This function should be called first when we construct the network as the starting layer
	input_size = size;
	assert(inputLayer == nullptr);
	layerinfo layer;
	layer.start.input_size = size;
	inputLayer = new InputLayer(layer);
	output_size = size;
	lastLayer = inputLayer;
	return inputLayer;
}

Layer* DnnFrmwk::AddFCLayer(int size)
{
	layerinfo layer;
	layer.fc.layer_size = size;

	if (inputLayer == nullptr) {
		// This is the first layer
		layer.fc.input_size = input_size;
		assert(lastLayer == nullptr);
		assert(input_size != 0);
		inputLayer = new FCLayer(layer, nullptr, dev_);
		assert(inputLayer != nullptr);
		lastLayer = inputLayer;
		output_size = size;
		return inputLayer;
	}

	layer.fc.input_size = output_size;
	Layer* newLayer = new FCLayer(layer, lastLayer, dev_);

	lastLayer->SetNextLayer(newLayer);
	lastLayer = newLayer;
	output_size = size;

	return newLayer;
}

Layer* DnnFrmwk::AddConvLayer(short in_channel, short input_height, short input_width, 
		short out_channel, short filter_size, short padding, short stride)
{
	layerinfo layer;
	layer.conv.c = in_channel;
	layer.conv.h = input_height;
	layer.conv.w = input_width;
	layer.conv.fn = out_channel;
	layer.conv.fh = filter_size;
	layer.conv.fw= filter_size;
	layer.conv.p = padding;
	layer.conv.s = stride;

	if (inputLayer == nullptr) {
		// This is the first layer
		assert(lastLayer == nullptr);
		inputLayer = new ConvLayer(layer, nullptr, dev_);
		assert(inputLayer != nullptr);
		lastLayer = inputLayer;
		output_size = inputLayer->GetLayerSize();  // FN*FH*FW
		return inputLayer;
	}

	Layer* newLayer = new ConvLayer(layer, lastLayer, dev_);
	lastLayer->SetNextLayer(newLayer);
	lastLayer = newLayer;
	output_size = newLayer->GetLayerSize();  // FN*C*OH*OW

	return newLayer;
}

Layer* DnnFrmwk::AddReluLayer()
{
	layerinfo layer;

	layer.relu.input_size = output_size;
	layer.relu.layer_size = output_size;

	Layer* newLayer = new ReluLayer(layer, lastLayer, dev_);
	assert(newLayer != nullptr);
	lastLayer->SetNextLayer(newLayer);
	lastLayer = newLayer;

	return newLayer;
}

Layer* DnnFrmwk::AddSoftmaxLayer()
{
	layerinfo layer;
	layer.softmax.input_size = output_size;
	layer.softmax.layer_size = output_size;

	softmaxLayer = new SoftmaxLayer(layer);
	assert(softmaxLayer != nullptr);

	return softmaxLayer;
}

void DnnFrmwk::SetBatchSize(int b_size)
{
	assert(inputLayer != nullptr);
	inputLayer->SetBatchSize(b_size);
	if (softmaxLayer != nullptr)
		softmaxLayer->SetBatchSize(b_size);
}

float* DnnFrmwk::Predict(float* data, int b_size, bool training)
{
	Layer* layer = inputLayer;
	assert(layer != nullptr);
	float* prediction, * softmax;
	prediction = layer->Forward(data, b_size);
	if (training) {
		softmax = softmaxLayer->Forward(prediction, b_size);
		return softmax;
	}
	return prediction;
}

// The value of label must be encoded as one-hot encoding
float DnnFrmwk::CrossEntropyError(float* prediction, int* label, int b_size)
{
	float tmp = 0,entropy = 0;
	const float epsilon = (float)1e-14;
	for (int i=0; i< b_size; i++) {
		tmp = 0;
		for (int j=0; j<output_size; j++)
			tmp += (float)(label[i*output_size + j] * (float)log(prediction[i * output_size + j] + epsilon));
		entropy += tmp;
	}
	return -1.0 * (entropy / b_size);
}

float DnnFrmwk::CalculateLoss(float* prediction, int* label, int b_size)
{
	float loss = CrossEntropyError(prediction, label, b_size);

	return loss;
}

void DnnFrmwk::BackPropagation(int* label, int b_size, float learning_rate)
{
	float* dx = softmaxLayer->GetDifferential((float*)label, b_size, learning_rate);
	lastLayer->Backward(dx, b_size, learning_rate);
}

int DnnFrmwk::Train(float* data, int* label, int b_size, float learning_rate)
{
	int i, j, correct = 0;
	assert(inputLayer && lastLayer);
	float *prediction;

	prediction = Predict(data, b_size, true);
	CalculateLoss(prediction, label, b_size);
	BackPropagation(label, b_size, learning_rate);

	for (i = 0; i < b_size; i++)
	{
		int idx = distance(&prediction[i * output_size],
				max_element(&prediction[i * output_size], &prediction[i * output_size + output_size]));
		for (j = 0; j < output_size; j++)
			if (label[i * output_size + j] != 0)
				break;
		if (idx == j)
			correct++;
	}
	return correct;
}

float DnnFrmwk::GetAccuracy(float* data, int* label, int data_size)
{
	int i, j;
	assert(inputLayer && lastLayer);
	float* prediction;
	int correct = 0;
	prediction = Predict(data, data_size, false);
	for (i = 0; i < data_size; i++)
	{
		int idx = distance(&prediction[i * output_size],
				max_element(&prediction[i * output_size], &prediction[i * output_size + output_size]));
		for (j = 0; j < output_size; j++)
			if (label[i * output_size + j] != 0)
				break;
		if (idx == j)
			correct++;
	}

	return ((float)correct / data_size);
}

