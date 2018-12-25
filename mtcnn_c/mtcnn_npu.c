#include "mtcnn.h"

void init_npu_device(GxDnnDevice* device)
{
	/* 打开npu设备 */
	GxDnnOpenDevice("/dev/gxnpu", device);
	return ;
}

Mat run_pnet(GxDnnDevice device, float* net_input_data, const char* model_name)
{
	int i = 0, j = 0, k = 0;
	int ret = 0;
	int priority = 5;
	int input_num = 0, output_num = 0;
	int input_size = 0, output_size = 0;
	float* ptr = NULL;
	float* result = NULL;
	int result_num = 0;
	int Mat_size = 0;
	Mat ret_img;
	GxDnnTask task;
	GxDnnIOInfo *input, *output;
	GxDnnEventHandler event_handler = NULL;

	/* 传入模型文件，获取模型task */
	ret = GxDnnCreateTaskFromFile(device, model_name, &task);
	if (ret != GXDNN_RESULT_SUCCESS) {
		printf("Error: load model fail!\n");
		return ret_img;
	}

	GxDnnGetTaskIONum(task, &input_num, &output_num);
	input_size = input_num * sizeof(GxDnnIOInfo);
	output_size = output_num * sizeof(GxDnnIOInfo);
	input = (GxDnnIOInfo*)malloc(input_size);
	output = (GxDnnIOInfo*)malloc(output_size);

	if (input == NULL || output == NULL) {
		printf("malloc failed in %d lines!\n", __LINE__);
		return ret_img;
	}

	GxDnnGetTaskIOInfo(task, input, input_size, output, output_size);

	memcpy(input[0].dataBuffer, (void*)net_input_data, input[0].bufferSize);

	GxDnnRunTask(task, priority, event_handler, NULL);

	result = (float*)output[1].dataBuffer;
	result_num = output[1].bufferSize / sizeof(float);

	Mat_size = sqrt(result_num / 4);
	Mat out0 = Mat::zeros(4, Mat_size, CV_32FC(Mat_size));

	for (i = 0; i < out0.rows; i++) {
		for (j = 0; j < out0.cols; j++) {
			ptr = out0.ptr<float>(i, j);
			for (k = 0; k < out0.channels(); k++) {
				*ptr = *result;
				ptr++;
				result++;
			}
		}
	}

	result = (float*)output[0].dataBuffer;
	result_num = output[0].bufferSize / sizeof(float);

	Mat_size = sqrt(result_num / 2);
	Mat out1 = Mat::zeros(2, Mat_size, CV_32FC(Mat_size));

	for (i = 0; i < out1.rows; i++) {
		for (j = 0; j < out1.cols; j++) {
			ptr = out1.ptr<float>(i, j);
			for (k = 0; k < out1.channels(); k++) {
				*ptr = *result;
				ptr++;
				result++;
			}
		}
	}

	vconcat(out0, out1, ret_img);

	GxDnnReleaseTask(task);

	free(input);
	free(output);

	return ret_img;
}

Mat run_rnet(GxDnnDevice device, float* net_input_data, const char* model_name, int len)
{
	int i = 0, j = 0;
	int ret = 0;
	int priority = 5;
	int input_num = 0, output_num = 0;
	int input_size = 0, output_size = 0;
	float* result = NULL;
	int result_num = 0;
	float *out_ptr = NULL;
	Mat ret_img = Mat::zeros(len, 6, CV_32FC1);
	GxDnnTask task;
	GxDnnIOInfo *input, *output;
	GxDnnEventHandler event_handler = NULL;

	/* 传入模型文件，获取模型task */
	ret = GxDnnCreateTaskFromFile(device, model_name, &task);
	if (ret != GXDNN_RESULT_SUCCESS) {
		printf("Error: load model fail!\n");
		return ret_img;
	}

	GxDnnGetTaskIONum(task, &input_num, &output_num);
	input_size = input_num * sizeof(GxDnnIOInfo);
	output_size = output_num * sizeof(GxDnnIOInfo);
	input = (GxDnnIOInfo*)malloc(input_size);
	output = (GxDnnIOInfo*)malloc(output_size);

	if (input == NULL || output == NULL) {
		printf("malloc failed in %d lines!\n", __LINE__);
		return ret_img;
	}

	GxDnnGetTaskIOInfo(task, input, input_size, output, output_size);

	out_ptr = (float*)malloc(len * 6 * sizeof(float));
	if (out_ptr == NULL) {
		printf("malloc failed in %d lines!\n", __LINE__);
		return ret_img;
	}
	float* tmp_out = out_ptr;

	while (len > 0) {
		memcpy(input[0].dataBuffer, (void*)net_input_data, input[0].bufferSize);
		GxDnnRunTask(task, priority, event_handler, NULL);

		result = (float*)output[1].dataBuffer;
		result_num = output[1].bufferSize / sizeof(float);
		for (i = 0; i < result_num; i++) {
			*tmp_out = *result;
			tmp_out++;
			result++;
		}

		result = (float*)output[0].dataBuffer;
		result_num = output[0].bufferSize / sizeof(float);
		for (i = 0; i < result_num; i++) {
			*tmp_out = *result;
			tmp_out++;
			result++;
		}

		len--;
		net_input_data += input[0].bufferSize / sizeof(float);
	}

	tmp_out = out_ptr;

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			*ret_img.ptr<float>(i, j) = *tmp_out;
			tmp_out++;
		}
	}

	GxDnnReleaseTask(task);

	free(out_ptr);
	free(input);
	free(output);

	return ret_img;
}

Mat run_onet(GxDnnDevice device, float* net_input_data, const char* model_name, int len)
{
	int i = 0, j = 0;
	int ret = 0;
	int priority = 5;
	int input_num = 0, output_num = 0;
	int input_size = 0, output_size = 0;
	float* result = NULL;
	int result_num = 0;
	float *out_ptr = NULL;
	Mat ret_img = Mat::zeros(len, 16, CV_32FC1);
	GxDnnTask task;
	GxDnnIOInfo *input, *output;
	GxDnnEventHandler event_handler = NULL;

	/* 传入模型文件，获取模型task */
	ret = GxDnnCreateTaskFromFile(device, model_name, &task);
	if (ret != GXDNN_RESULT_SUCCESS) {
		printf("Error: load model fail!\n");
		return ret_img;
	}

	GxDnnGetTaskIONum(task, &input_num, &output_num);
	input_size = input_num * sizeof(GxDnnIOInfo);
	output_size = output_num * sizeof(GxDnnIOInfo);
	input = (GxDnnIOInfo*)malloc(input_size);
	output = (GxDnnIOInfo*)malloc(output_size);

	if (input == NULL || output == NULL) {
		printf("malloc failed in %d lines!\n", __LINE__);
		return ret_img;
	}

	GxDnnGetTaskIOInfo(task, input, input_size, output, output_size);

	out_ptr = (float*)malloc(len * 16 * sizeof(float));
	if (out_ptr == NULL) {
		printf("malloc failed in %d lines!\n", __LINE__);
		return ret_img;
	}

	float* tmp_out = out_ptr;

	while (len > 0) {
		memcpy(input[0].dataBuffer, (void*)net_input_data, input[0].bufferSize);
		GxDnnRunTask(task, priority, event_handler, NULL);

		result = (float*)output[1].dataBuffer;
		result_num = output[1].bufferSize / sizeof(float);
		for (i = 0; i < result_num; i++) {
			*tmp_out = *result;
			tmp_out++;
			result++;
		}

		result = (float*)output[2].dataBuffer;
		result_num = output[2].bufferSize / sizeof(float);
		for (i = 0; i < result_num; i++) {
			*tmp_out = *result;
			tmp_out++;
			result++;
		}

		result = (float*)output[0].dataBuffer;
		result_num = output[0].bufferSize / sizeof(float);
		for (i = 0; i < result_num; i++) {
			*tmp_out = *result;
			tmp_out++;
			result++;
		}

		len--;
		net_input_data += input[0].bufferSize / sizeof(float);
	}

	tmp_out = out_ptr;

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			*ret_img.ptr<float>(i, j) = *tmp_out;
			tmp_out++;
		}
	}

	GxDnnReleaseTask(task);

	free(out_ptr);
	free(input);
	free(output);

	return ret_img;
}

