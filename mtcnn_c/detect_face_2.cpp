#include "mtcnn.h"

using namespace std;
using namespace cv;

Mat detect_face(Mat* img, float* threshold, float* scales, int scales_len, Mat* ret_points)
{
	int i = 0;
	int ret = 0;
	int load_ret = 0;
	int h = img->rows;
	int w = img->cols;
	int hs = 0, ws = 0;
	int len = 0;
	float scale = 0;
	short* pick = NULL;
	int pick_len = 0;
	int ipass_len = 0;
	int Mat_init_size[3] = {0};
	Mat total_boxes;

	struct npu_info arr_npu[NET_MODEL_NUM] = {0};
	const char* model_file[NET_MODEL_NUM] = {"./npu/pnet49.npu", "./npu/pnet41.npu", "./npu/pnet35.npu", "./npu/pnet30.npu", "./npu/pnet26.npu", "./npu/pnet22.npu", "./npu/pnet19.npu", "./npu/pnet16.npu", "./npu/pnet14.npu", "./npu/rnet.npu", "./npu/onet.npu"};
	int model_file_index = 0;

	int numbox = 0;
	int* dy = NULL;
	int* edy = NULL;
	int* dx = NULL;
	int* edx = NULL;
	int* y = NULL;
	int* ey = NULL;
	int* x = NULL;
	int* ex = NULL;
	int* tmpw = NULL;
	int* tmph = NULL;

	GxDnnDevice npu_device;

	init_npu_device(&npu_device);

	load_ret = load_npu_model(npu_device, model_file, arr_npu, NET_MODEL_NUM);
	if (load_ret != NET_MODEL_NUM) {
		goto EXECUTE_ERROR;
	}

	for (i = 7; i < scales_len; i++) {
		scale = scales[i];
		hs = (int)ceil(h * scale);
		ws = (int)ceil(w * scale);

		Mat im_data = imresample(img, hs, ws, (unsigned char)1);
		float* img_y = get_img_y(&im_data);
		if (img_y == NULL) {
			goto EXECUTE_ERROR;
		}

		/* out = pnet(img_y) */
		Mat out = run_pnet(img_y, arr_npu[model_file_index]);
		if (out.rows * out.cols == 0) {
			free(img_y);
			goto EXECUTE_ERROR;
		}

		model_file_index++;

		Mat out0 = out.rowRange(0, 4).clone();
		Mat out1 = out.rowRange(4, 6).clone();

		Mat in0 = get_in0(&out0);
		Mat in1 = get_in1(&out1);

		Mat boxes = generateBoundingBox(&in1, &in0, scale, threshold[0]);
		if (boxes.rows * boxes.cols == 0) {
			free(img_y);
			continue;
		}

		pick = nms(&boxes, 0.5, "Union", &pick_len);
		if (pick == NULL) {
			free(img_y);
			goto EXECUTE_ERROR;
		}

		if ((boxes.rows * boxes.cols > 0) && (pick_len > 0)) {
			Mat boxes1 = get_boxes_from_pick(&boxes, pick, pick_len);
			total_boxes = append_total_boxes(&total_boxes, &boxes1);
		}

		free(img_y);
		free(pick);
	}

	if (total_boxes.rows * total_boxes.cols == 0)
		return total_boxes;

	numbox = total_boxes.rows;

	if (numbox > 0) {
		pick = nms(&total_boxes, 0.7, "Union", &pick_len);
		if (pick == NULL) {
			goto EXECUTE_ERROR;
		}

		total_boxes = get_boxes_from_pick(&total_boxes, pick, pick_len);
		len = total_boxes.rows;

		float* regw = mat_cols_sub(total_boxes.colRange(2, 3), total_boxes.colRange(0, 1));
		if (regw == NULL) {
			free(pick);
			goto EXECUTE_ERROR;
		}

		float* regh = mat_cols_sub(total_boxes.colRange(3, 4), total_boxes.colRange(1, 2));
		if (regh == NULL) {
			free(regw);
			free(pick);
			goto EXECUTE_ERROR;
		}

		float* qq1 = get_qq(&total_boxes, 0, 5, regw);
		if (qq1 == NULL) {
			free(regh);
			free(regw);
			free(pick);
			goto EXECUTE_ERROR;
		}

		float* qq2 = get_qq(&total_boxes, 1, 6, regh);
		if (qq2 == NULL) {
			free(qq1);
			free(regh);
			free(regw);
			free(pick);
			goto EXECUTE_ERROR;
		}

		float* qq3 = get_qq(&total_boxes, 2, 7, regw);
		if (qq3 == NULL) {
			free(qq2);
			free(qq1);
			free(regh);
			free(regw);
			free(pick);
			goto EXECUTE_ERROR;
		}

		float* qq4 = get_qq(&total_boxes, 3, 8, regh);
		if (qq4 == NULL) {
			free(qq3);
			free(qq2);
			free(qq1);
			free(regh);
			free(regw);
			free(pick);
			goto EXECUTE_ERROR;
		}

		total_boxes = get_vstack_qq_and_transpose(qq1, qq2, qq3, qq4, &total_boxes, 4);

		ret = rerec(&total_boxes);
		if (ret != 0) {
			free(qq4);
			free(qq3);
			free(qq2);
			free(qq1);
			free(regh);
			free(regw);
			free(pick);
			goto EXECUTE_ERROR;
		}

		get_total_boxes_fix(&total_boxes, 0, 4, 0, 4);

		dy = (int*)malloc(len * sizeof(int));
		edy = (int*)malloc(len * sizeof(int));
		dx = (int*)malloc(len * sizeof(int));
		edx = (int*)malloc(len * sizeof(int));
		y = (int*)malloc(len * sizeof(int));
		ey = (int*)malloc(len * sizeof(int));
		x = (int*)malloc(len * sizeof(int));
		ex = (int*)malloc(len * sizeof(int));
		tmpw = (int*)malloc(len * sizeof(int));
		tmph = (int*)malloc(len * sizeof(int));
		if (dy == NULL || edy == NULL || dx == NULL || edx == NULL || y == NULL || ey == NULL || x == NULL || ex == NULL || tmpw == NULL || tmph == NULL) {
			printf("malloc failed in %s %d lines\n", __func__, __LINE__);
			free(qq4);
			free(qq3);
			free(qq2);
			free(qq1);
			free(regh);
			free(regw);
			free(pick);
			goto EXECUTE_ERROR;
			/* 这里可能会导致有些malloc出来的空间没有free */
		}

		pad(&total_boxes, h, w, dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph);

		free(regw);
		free(regh);
		free(qq1);
		free(qq2);
		free(qq3);
		free(qq4);
		free(pick);
	}

	numbox = total_boxes.rows;

	if (numbox > 0) {
		Mat_init_size[0] = 3;
		Mat_init_size[1] = 24;
		Mat_init_size[2] = 24;
		Mat tempimg = Mat(3, Mat_init_size, CV_32FC(numbox), Scalar::all(0));

		for (i = 0; i < numbox; i++) {
			Mat tmp = Mat::zeros(tmph[i], tmpw[i], CV_32FC3);

			buckle_map(img, &tmp, x, ex, y, ey, dx, edx, dy, edy, i);
			if (((tmp.rows > 0) && (tmp.cols > 0)) || ((tmp.rows == 0) && (tmp.cols == 0))) {
				Mat tmp_tempimg = imresample(&tmp, 24, 24, (float)1);
				get_tempimg(&tempimg, &tmp_tempimg, i, Mat_init_size[2]);
			} else {
				printf("buckle map execute failed!\n");
				goto EXECUTE_ERROR;
			}
		}

		Mat tempimg_tmp = transpose3021(&tempimg, Mat_init_size[2]);
		float* tempimg1 = image_normalization(&tempimg_tmp, Mat_init_size[2], (float)1);
		if (tempimg1 == NULL) {
				goto EXECUTE_ERROR;
		}

		/* out = rnet(tempimg1); */
		Mat out = run_rnet(tempimg1, arr_npu[model_file_index], numbox);
		if (out.rows * out.cols == 0) {
			free(tempimg1);
			goto EXECUTE_ERROR;
		}

		model_file_index++;

		Mat out0 = out.colRange(0, 4).clone();
		Mat out1 = out.colRange(4, 6).clone();

		transpose(out0, out0);
		transpose(out1, out1);

		len = out0.cols;
		Mat score = out1.rowRange(1, 2).clone();
		int* ipass = get_ipass(&score, threshold[1], len, &ipass_len);
		if (ipass == NULL) {
			free(tempimg1);
			goto EXECUTE_ERROR;
		}

		total_boxes = get_hstack_ronet(&total_boxes, ipass, 0, 4, &score, ipass_len);

		Mat mv = get_mv(&out0, ipass, ipass_len);

		if (total_boxes.rows > 0) {
			pick = nms(&total_boxes, 0.7, "Union", &pick_len);
			if (pick == NULL) {
				free(ipass);
				free(tempimg1);
				goto EXECUTE_ERROR;
			}

			total_boxes = get_total_boxes_pick(&total_boxes, pick, pick_len);
			mv = transpose_mv_piack(&mv, pick, pick_len);
			ret = bbreg(&total_boxes, &mv);
			if (ret != 0) {
				free(ipass);
				free(tempimg1);
				goto EXECUTE_ERROR;
			}

			ret = rerec(&total_boxes);
			if (ret != 0) {
				free(ipass);
				free(tempimg1);
				goto EXECUTE_ERROR;
			}
		}

		free(tempimg1);
		free(dy);
		free(edy);
		free(dx);
		free(edx);
		free(y);
		free(ey);
		free(x);
		free(ex);
		free(tmpw);
		free(tmph);
		free(pick);
		free(ipass);
	}

	numbox = total_boxes.rows;

	if (numbox > 0) {
		len = total_boxes.rows;
		dy = (int*)malloc(len * sizeof(int));
		edy = (int*)malloc(len * sizeof(int));
		dx = (int*)malloc(len * sizeof(int));
		edx = (int*)malloc(len * sizeof(int));
		y = (int*)malloc(len * sizeof(int));
		ey = (int*)malloc(len * sizeof(int));
		x = (int*)malloc(len * sizeof(int));
		ex = (int*)malloc(len * sizeof(int));
		tmpw = (int*)malloc(len * sizeof(int));
		tmph = (int*)malloc(len * sizeof(int));
		if (dy == NULL || edy == NULL || dx == NULL || edx == NULL || y == NULL || ey == NULL || x == NULL || ex == NULL || tmpw == NULL || tmph == NULL) {
			printf("malloc failed in %s %d lines\n", __func__, __LINE__);
			goto EXECUTE_ERROR;
		}

		total_boxes = fix_total_boxes(&total_boxes);
		pad(&total_boxes, h, w, dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph);
		Mat_init_size[0] = 3;
		Mat_init_size[1] = 48;
		Mat_init_size[2] = 48;
		Mat tempimg = Mat(3, Mat_init_size, CV_32FC(numbox), Scalar::all(0));

		for (i = 0; i < numbox; i++) {
			Mat tmp = Mat::zeros(tmph[i], tmpw[i], CV_32FC3);

			buckle_map(img, &tmp, x, ex, y, ey, dx, edx, dy, edy, i);
			if (((tmp.rows > 0) && (tmp.cols > 0)) || ((tmp.rows == 0) && (tmp.cols == 0))) {
				Mat tmp_tempimg = imresample(&tmp, 48, 48, (float)1);
				get_tempimg(&tempimg, &tmp_tempimg, i, Mat_init_size[2]);
			} else {
				printf("buckle map execute failed!\n");
				goto EXECUTE_ERROR;
			}
		}

		Mat tempimg_tmp = transpose3021(&tempimg, Mat_init_size[2]);
		float* tempimg1 = image_normalization(&tempimg_tmp, Mat_init_size[2], (float)1);
		if (tempimg1 == NULL) {
				goto EXECUTE_ERROR;
		}

		Mat out = run_onet(tempimg1, arr_npu[model_file_index], numbox);
		if (out.rows * out.cols == 0) {
			free(tempimg1);
			goto EXECUTE_ERROR;
		}

		Mat out0 = out.colRange(0, 4).clone();
		Mat out1 = out.colRange(4, 14).clone();
		Mat out2 = out.colRange(14, 16).clone();

		transpose(out0, out0);
		transpose(out1, out1);
		transpose(out2, out2);

		len = out0.cols;
		Mat score = out2.rowRange(1, 2).clone();
		int* ipass = get_ipass(&score, threshold[2], len, &ipass_len);
		if (ipass == NULL) {
			free(tempimg1);
			goto EXECUTE_ERROR;
		}

		Mat points = get_points(&out1, ipass, ipass_len);

		total_boxes = get_hstack_ronet(&total_boxes, ipass, 0, 4, &score, ipass_len);

		Mat mv = get_mv(&out0, ipass, ipass_len);

		float* w = (float*)malloc(ipass_len * sizeof(float));
		if (w == NULL) {
			printf("malloc failed in %s %d lines\n", __func__, __LINE__);
			free(ipass);
			free(tempimg1);
			goto EXECUTE_ERROR;
		}

		float* h = (float*)malloc(ipass_len * sizeof(float));
		if (h == NULL) {
			printf("malloc failed in %s %d lines\n", __func__, __LINE__);
			free(w);
			free(ipass);
			free(tempimg1);
			goto EXECUTE_ERROR;
		}

		get_wh_in_bbreg(&total_boxes, w, h, ipass_len);

		update_points(&points, &total_boxes, w, h, ipass_len);

		if (total_boxes.rows > 0) {
			transpose(mv, mv);
			bbreg(&total_boxes, &mv);
			pick = nms(&total_boxes, 0.7, "Min", &pick_len);
			if (pick == NULL) {
				free(h);
				free(w);
				free(ipass);
				free(tempimg1);
				goto EXECUTE_ERROR;
			}

			total_boxes = get_boxes_from_pick(&total_boxes, pick, pick_len);

			*ret_points = points_pick(&points, pick, pick_len);
		}

		free(w);
		free(h);
		free(ipass);
		free(dy);
		free(edy);
		free(dx);
		free(edx);
		free(y);
		free(ey);
		free(x);
		free(ex);
		free(tmpw);
		free(tmph);
		free(pick);
	}

	release_npu_model(npu_device, arr_npu, load_ret);

	return total_boxes;

EXECUTE_ERROR:
	release_npu_model(npu_device, arr_npu, load_ret);
	return *ret_points;
}
