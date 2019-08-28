#pragma once
#include "assert.h"
#include "Layer.h"

class ConvLayer;
class ConvLayer: public Layer {
protected:
	short kn_;
	short kh_;
	short kw_;

	short oh_;
	short ow_;
	int c_;

	int input_size_;
	int	output_size_;
	int weight_size_;
	int im2col_size_;

	float* outMatrixT_;
	float* inMatrix_;	// For training, im2col matrix data for input is preserved for backward propagation
	float* dWT_;
	float* unitDW_;

	void Padding(float* src, float* dest, int pad_size, int batch);
	
public:
	ConvLayer(layerinfo info, Layer* prevLayer, Device* dev);	
	~ConvLayer();

	void im2col(const float *img, float *col);
	void col2im(const float *col, float *img);

	void SetParameterWeight(float* value);
	void SetParameterBias(int* value);
	void AllocateMemoryforWeightsAndBias();
	void InitializeWeightsAndBias();
	void SetOutputImageSize();
	void SetBatchSize(short b_size);
	int GetLayerSize();
	
	float* Forward(float* input, short b_size);
	void Backward(float* input, short b_size, float learning_rate);
}; 
