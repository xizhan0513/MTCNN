#include "mtcnn.h"

void init_npu_device(GxDnnDevice* device)
{
	/* 打开npu设备 */
	GxDnnOpenDevice("/dev/gxnpu", device);
	return ;
}

Mat run_net(GxDnnDevice device, float* input_data, const char* model_name)
{
	int i = 0, j = 0, k = 0;
	int w = 0, l = 0, c = 0;
	int ret = 0;
	int priority = 5;
	int input_num = 0, output_num = 0;
	int input_size = 0, output_size = 0;
	float* result = NULL;
	int result_num = 0;
	GxDnnTask task;
	GxDnnIOInfo *input, *output;
	GxDnnEventHandler event_handler = NULL;

	/* 传入模型文件，获取模型task */
	ret = GxDnnCreateTaskFromFile(device, model_name, &task);
	if (ret != GXDNN_RESULT_SUCCESS) {
		printf("Error: load model fail!\n");
		return -1;
	}

	GxDnnGetTaskIONum(task, &input_num, &output_num);
	input_size = input_num * sizeof(GxDnnIOInfo);
	output_size = output_num * sizeof(GxDnnIOInfo);
	input = (GxDnnIOInfo*)malloc(input_size);
	output = (GxDnnIOInfo*)malloc(output_size);
	if (input == NULL || output == NULL) {
		printf("Error: malloc fail!\n");
		return -1;
	}

	GxDnnGetTaskIOInfo(task, input, input_size, output, output_size);

	memcpy(input[0].dataBuffer, (void*)input_data, input[0].bufferSize);

	GxDnnRunTask(task, priority, event_handler, NULL);

	result = (float*)output[0].dataBuffer;
	result_num = output[0].bufferSize / sizeof(float);

	Mat ret_img = get_pnet_out();

	w = 6;
	l = c = sqrt(result_num / 4);

	float* ptr = NULL;
	int Mat_init_size[3] = {0};
	Mat_init_size[0] = 1;
	Mat_init_size[1] = w;
	Mat_init_size[2] = l;
	Mat tmp = Mat(3, Mat_init_size, CV_32FC(c), Scalar::all(0));


	*out1 = tmp;
	printf("***************\n");
	for (i = 0; i < w; i++) {
		for (j = 0; j < l; j++) {
			ptr = (float*)(out1->data + out1->step[0] * 0 + out1->step[1] * i + out1->step[2] * j);
			for (k = 0; k < c; k++) {
				*ptr = *result;
				printf("----\n");
				ptr++;
				result++;
			}
		}
	}

	printf("***************\n");
	result = (float*)output[1].dataBuffer;
	result_num = output[1].bufferSize / sizeof(float);

	w = 4;
	l = c = sqrt(result_num / 4);
	Mat_init_size[0] = 1;
	Mat_init_size[1] = w;
	Mat_init_size[2] = l;
	*out0 = Mat(3, Mat_init_size, CV_32FC(c), Scalar::all(0));

	for (i = 0; i < w; i++) {
		for (j = 0; j < l; j++) {
			ptr = (float*)(out0->data + out0->step[0] * 0 + out0->step[1] * i + out0->step[2] * j);
			for (k = 0; k < c; k++) {
				*ptr = *result;
				ptr++;
				result++;
			}
		}
	}

	//print_1D(result, result_num);

	GxDnnReleaseTask(task);

	free(input);
	free(output);

	return 0;
}
