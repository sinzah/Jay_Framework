#pragma once

#include "Device.h"

typedef unsigned char	BYTE;

typedef struct __conv {
	short h;	// input image height
	short w;	// input image width
	BYTE c;		// input channel
	BYTE fn;	// output channle
	BYTE fh;	// kernel height
	BYTE fw;	// kernel width
	BYTE s;		// stride
	BYTE p;		// padding
} conv_t;

typedef struct __fc {
	short input_size;
	short layer_size;
} fc_t;

typedef struct __softmax {
	short input_size;
	short layer_size;
} softmax_t;

typedef struct __relu {
	short input_size;
	short layer_size;
} relu_t;

typedef struct __input {
	short input_size;
} input_t;

typedef union __layer_info {
	conv_t		conv;
	fc_t		fc;
	softmax_t	softmax;
	relu_t		relu;
	input_t		start;
} layerinfo;

typedef enum
{ 	DEV_CPU,
	DEV_FPGA_ZED,
	DEV_FPGA_KCU1500,
	DEV_SOPHON
} dev_type;

typedef enum { NET_FC, NET_CONV } net_type;
