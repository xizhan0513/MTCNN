#include "mtcnn.h"

void init_npu_device(GxDnnDevice* device)
{
	/* 打开npu设备 */
	GxDnnOpenDevice("/dev/gxnpu", device);
	return ;
}

int load_npu_model(GxDnnDevice device, const char** model_file, struct npu_info* arr_npu, int len)
{
	int i = 0;
	int ret = 0;

	for (i = 0; i < len; i++) {
		ret = GxDnnCreateTaskFromFile(device, model_file[i], &arr_npu[i].task);
		if (ret != GXDNN_RESULT_SUCCESS) {
			printf("Error: load model fail!\n");
			return i;
		}

		GxDnnGetTaskIONum(arr_npu[i].task, &arr_npu[i].input_num, &arr_npu[i].output_num);
		arr_npu[i].input_size = arr_npu[i].input_num * sizeof(GxDnnIOInfo);
		arr_npu[i].output_size = arr_npu[i].output_num * sizeof(GxDnnIOInfo);
		arr_npu[i].priority = 5;
		arr_npu[i].event_handler = NULL;
	}

	return i;
}

Mat run_pnet(float* net_input_data, struct npu_info npu)
{
	int i = 0, j = 0, k = 0;
	float* ptr = NULL;
	float* result = NULL;
	int result_num = 0;
	int Mat_size = 0;
	Mat ret_img;
	GxDnnIOInfo *input, *output;

	input = (GxDnnIOInfo*)malloc(npu.input_size);
	if (input == NULL) {
		printf("malloc failed in %s %d lines!\n", __func__, __LINE__);
		return ret_img;
	}

	output = (GxDnnIOInfo*)malloc(npu.output_size);
	if (output == NULL) {
		printf("malloc failed in %s %d lines!\n", __func__, __LINE__);
		free(input);
		return ret_img;
	}

	GxDnnGetTaskIOInfo(npu.task, input, npu.input_size, output, npu.output_size);

	memcpy(input[0].dataBuffer, (void*)net_input_data, input[0].bufferSize);

	GxDnnRunTask(npu.task, npu.priority, npu.event_handler, NULL);

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

	free(input);
	free(output);

	return ret_img;
}

Mat run_rnet(float* net_input_data, struct npu_info npu, int len)
{
	int i = 0, j = 0;
	float* result = NULL;
	float* out_ptr = NULL;
	int result_num = 0;
	Mat ret_img = Mat::zeros(len, 6, CV_32FC1);
	Mat error_img;
	GxDnnIOInfo *input, *output;

	input = (GxDnnIOInfo*)malloc(npu.input_size);
	if (input == NULL) {
		printf("malloc failed in %s %d lines!\n", __func__, __LINE__);
		return error_img;
	}

	output = (GxDnnIOInfo*)malloc(npu.output_size);
	if (output == NULL) {
		printf("malloc failed in %s %d lines!\n", __func__, __LINE__);
		free(input);
		return error_img;
	}

	GxDnnGetTaskIOInfo(npu.task, input, npu.input_size, output, npu.output_size);

	out_ptr = (float*)malloc(len * 6 * sizeof(float));
	if (out_ptr == NULL) {
		printf("malloc failed in %s %d lines!\n", __func__, __LINE__);
		free(input);
		free(output);
		return error_img;
	}

	float* tmp_out = out_ptr;

	while (len > 0) {
		memcpy(input[0].dataBuffer, (void*)net_input_data, input[0].bufferSize);
		GxDnnRunTask(npu.task, npu.priority, npu.event_handler, NULL);

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

	free(out_ptr);
	free(input);
	free(output);

	return ret_img;
}

Mat run_onet(float* net_input_data, struct npu_info npu, int len)
{
	int i = 0, j = 0;
	float* result = NULL;
	float* out_ptr = NULL;
	int result_num = 0;
	Mat ret_img = Mat::zeros(len, 16, CV_32FC1);
	Mat error_img;
	GxDnnIOInfo *input, *output;

	input = (GxDnnIOInfo*)malloc(npu.input_size);
	if (input == NULL) {
		printf("malloc failed in %s %d lines!\n", __func__, __LINE__);
		return error_img;
	}

	output = (GxDnnIOInfo*)malloc(npu.output_size);
	if (output == NULL) {
		printf("malloc failed in %s %d lines!\n", __func__, __LINE__);
		free(input);
		return error_img;
	}

	GxDnnGetTaskIOInfo(npu.task, input, npu.input_size, output, npu.output_size);

	out_ptr = (float*)malloc(len * 16 * sizeof(float));
	if (out_ptr == NULL) {
		printf("malloc failed in %s %d lines!\n", __func__, __LINE__);
		free(input);
		free(output);
		return error_img;
	}

	float* tmp_out = out_ptr;

	while (len > 0) {
		memcpy(input[0].dataBuffer, (void*)net_input_data, input[0].bufferSize);
		GxDnnRunTask(npu.task, npu.priority, npu.event_handler, NULL);

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


	free(out_ptr);
	free(input);
	free(output);

	return ret_img;
}

void release_npu_model(GxDnnDevice device, struct npu_info* arr_npu, int error_index)
{
	int i = 0;

	for (i = 0; i < error_index; i++) {
		GxDnnReleaseTask(arr_npu[i].task);
	}

	GxDnnCloseDevice(device);
	return ;
}

