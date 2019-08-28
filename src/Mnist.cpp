#include <iostream>
#include <string>
#include <fstream>
#include <memory.h>
#include "common.h"
#include "DnnFrmwk.h"
#include "utils.h"

#define BS					100
#define TRAIN_CNT			60000
#define TEST_CNT			10000
#define DATA_SIZE			784
#define CAT_SIZE			10

#define DATA_TRAIN_X		"dataset/train_x.dat"
#define DATA_TRAIN_Y		"dataset/train_y.dat"
#define DATA_TEST_X			"dataset/test_x.dat"
#define DATA_TEST_Y			"dataset/test_y.dat"
#define MODEL_PARAM			"weights.txt"

using namespace std;

dev_type g_dt = DEV_CPU;
net_type g_nt = NET_FC;
string g_model = "";
bool g_infer_only = false;

void ParseArgument(int argc, char* argv[])
{
	for (int i=1; i<argc; i++)
	{
		if (!strcmp(argv[i], "conv")) {
			g_nt = NET_CONV;
			cout << "Conv Network is selected!" << endl;
		} else if (!strcmp(argv[i], "zed")) {
			g_dt = DEV_FPGA_KCU1500;
			cout << "KCU1500 FPGA is selected!" << endl;
		} else if (!strcmp(argv[i], "kcu")) {
			g_dt = DEV_FPGA_ZED;
			cout << "ZedBoard FPGA is selected!" << endl;
		} else if (!strcmp(argv[i], "sophon")) {
			g_dt = DEV_SOPHON;
			cout << "Sophon is selected!" << endl;
		} else if (!strcmp(argv[i], "cpu")) {
			g_dt = DEV_CPU;
			cout << "CPU is selected!" << endl;
		} else if (!strncmp(argv[i], "model=", 6)) {
			g_model = argv[i] + 6;
			cout << "Model file is provided: " << g_model << endl;
		} else if (!strcmp(argv[i], "inference")) {
			g_infer_only = true;
			cout << "Inference Only" << endl;
		} else if (!strcmp(argv[i], "--help")) {
			cout << "Usage: mnist [network type | device type | model file | inference]" << endl;
			cout << "[fc|conv]: Network Type. fc=fully connected, conv=convolution" << endl;
			cout << "[cpu|kcu|zed]: Device Type" << endl;
			cout << "[model=../cnn_weights.txt]: Model parameter file name" << endl;
			cout << "[inference]: inference only execution" << endl;
			exit(0);
		}

	}
	if (g_nt == NET_FC && g_model == "")
		cout << "Fully Connected Network is selected!" << endl;
	if (g_dt == DEV_CPU)
		cout << "CPU is selected!" << endl;

}


bool FCNetwork(DnnFrmwk& net)
{
	net.SetInputLayer(1*28*28);
	net.AddFCLayer(512);
	net.AddReluLayer();
	net.AddFCLayer(10);
	net.AddSoftmaxLayer();

	return true;
}

bool ConvNetwork(DnnFrmwk& net)
{
	net.SetInputLayer(1*28*28);
	net.AddConvLayer(1, 28, 28, 6, 3, 0, 1);
	net.AddReluLayer();
	net.AddFCLayer(30);
	net.AddReluLayer();
	net.AddFCLayer(10);
	net.AddSoftmaxLayer();

	return true;
}

bool NetworkFromFile(DnnFrmwk& net, string model)
{
	std::string layer_name, layer_type, input, output;
	int num_layers;
	float *raw_weights;
	int input_size, output_size;
	Layer* prevLayer = nullptr;

	fstream fin(model.c_str(), ios::in | ios::binary);
	if (!fin.is_open()) {
		return false;
	}
	fin >> num_layers;

	input_size = 1 * 28 * 28;   //Mnist input size
	net.SetInputLayer(input_size);

	for (int i=0;i<num_layers;i++)
	{
		fin >> layer_name >> layer_type >> input >> output;

		if (layer_type.compare("slim.layers.fully_connected") == 0)
		{
			fin >> output_size;
			raw_weights = new float[output_size * input_size];
			memset(raw_weights, 0, sizeof(float) * output_size * input_size);

			for (int onode = 0; onode < output_size; onode++)
				for (int inode = 0; inode < input_size; inode++)
					fin >> raw_weights[onode * input_size + inode];

			prevLayer = net.AddFCLayer(output_size);
			prevLayer->SetParameterWeight(raw_weights);
		}
		else if (layer_type.compare("slim.layers.conv2d") == 0)
		{
			fin >> output_size;
			int input_channel, input_height, input_width;
			fin >> input_channel >> input_height >> input_width;
			int conv_channel, conv_height, conv_width;
			fin >> conv_channel >> conv_height >> conv_width;
			int os = conv_channel*input_channel*conv_height*conv_width;
			raw_weights = new float[conv_channel*input_channel*conv_height*conv_width];
			for (int j=0;j<conv_channel*input_channel*conv_height*conv_width;j++)
				fin >> raw_weights[j];

			// TODO:: Filter should be a square and the size of height and width must be same
			// In this case, there is no padding
			prevLayer = net.AddConvLayer(input_channel, input_height, input_width, conv_channel, conv_height, 0, 1);
			prevLayer->SetParameterWeight(raw_weights);
		}
		else if (layer_type.compare("tf.nn.relu") == 0) {
			prevLayer = net.AddReluLayer();
		}
		else if (layer_type.compare("tf.nn.softmax") == 0) {
			prevLayer = net.AddSoftmaxLayer();
		}
		else if (layer_type.compare("slim.layers.flatten") == 0) {
		}

		input_size = output_size;
	}

	return true;
}

float MeasureAccuracy(DnnFrmwk& net)
{
	ifstream datax, datay;

	datax.open(DATA_TEST_X, ios::binary);
	datay.open(DATA_TEST_Y, ios::binary);
	datax.seekg(0, std::ifstream::end);
	long size = (long)(datax.tellg()) / (DATA_SIZE * 4);
	datax.seekg(0);

	net.SetBatchSize(size);

	float* test_x = new float[DATA_SIZE * size];
	int* test_y = new int[CAT_SIZE * size];

	datax.read((char*)test_x, DATA_SIZE * size * 4);
	datay.read((char*)test_y, CAT_SIZE * size * 4);

	printf("Measuring Accuracy with test dataset [count: %d].\n", (int)size);
	float acc = net.GetAccuracy(test_x, test_y, size);

	delete[] test_x;
	delete[] test_y;

	datax.close();
	datay.close();

	return acc;
}


void TrainStart(DnnFrmwk& mynet)
{
	ifstream datax, datay;
	double start, end;
	const int total_epoch = 10;
	int correct_cnt = 0;

	mynet.SetBatchSize(BS);

	datax.open(DATA_TRAIN_X, ios::binary);
	datay.open(DATA_TRAIN_Y, ios::binary);
	datax.seekg(0, std::ifstream::end);
	long size = (long)(datax.tellg()) / (DATA_SIZE*4);
	datax.seekg(0);

	float* train_x = new float[DATA_SIZE * BS];
	int* train_y = new int[CAT_SIZE * BS];

	printf("Training Begins...\n");
	double t_start = get_time();
	for (int epoch = 0; epoch < total_epoch; epoch++) {
		start = get_time();
		correct_cnt = 0;
		for (int i = 0; i < (size+1) / BS; i++)
		{
			datax.seekg(i * DATA_SIZE * BS * 4);
			datay.seekg(i * BS * 4 * 10);

			datax.read((char*)train_x, DATA_SIZE * BS * 4);
			datay.read((char*)train_y, CAT_SIZE * BS * 4);
			correct_cnt += mynet.Train(train_x, train_y, BS, (float)0.01);

			print_progress(i + 1, (size + 1) / BS);
		}
		end = get_time();
		printf("\nEpoch %d: Training Accuracy = %6.2f%% [Elapsed Time: %.2f seconds]\n",
				epoch+1, ((float)correct_cnt / size) * 100, (end-start));
	}
	double t_end = get_time();

	datax.close();
	datay.close();
	delete[] train_x;
	delete[] train_y;

	printf("\nTraining Finished. [Elapsed Time: %.2f seconds]\n", (t_end-t_start));

}

void TestAccuracy(DnnFrmwk& mynet)
{
	double start, end;
	float acc;

	start = get_time();
	acc = MeasureAccuracy(mynet);
	end = get_time();

	printf("Accuracy is %6.2f%% [Elapsed Time: %.2f seconds]\n", acc*100, (end - start));

#ifdef _WIN32
	system("pause");
#endif
}

bool ConstructNetwork(DnnFrmwk& mynet, net_type nt, string modelFile = "")
{
	bool ret = false;
	if (modelFile.size() != 0)
	{
		ret = NetworkFromFile(mynet, modelFile);
		if (ret == false) {
			printf("Model File is not found!\n");
			return false;
		}
	}
	else {
		if (nt == NET_FC)
			ret = FCNetwork(mynet);
		else if (nt == NET_CONV)
			ret = ConvNetwork(mynet);
		else
			printf("Network is not supported!\n");
	}

	return ret;
}

int main(int argc, char* argv[])
{
	// Parse command line argument
	ParseArgument(argc, argv);

	// Declaration of DNN Framework
	DnnFrmwk mynet(g_dt);

	// Construct Network
	if (ConstructNetwork(mynet, g_nt, g_model) == false)
		return -1;

	// Start Training
	if (!g_infer_only)
		TrainStart(mynet);

	// Accuracy Measure with Test Dataset
	TestAccuracy(mynet);

	return 0;
}
