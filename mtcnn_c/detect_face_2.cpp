#include "mtcnn.h"

using namespace std;
using namespace cv;

Mat detect_face(Mat* img, float* threshold, double* scales, int scales_len, Mat* ret_points)
{
	int i = 0;
	int h = img->rows;
	int w = img->cols;
	int hs = 0, ws = 0;
	int len = 0;
	double scale = 0;
	short* pick = NULL;
	int pick_len = 0;
	int ipass_len = 0;
	Mat total_boxes;

	int Mat_init_size[3] = {0};
	int pnet_out_shape[9][4] = {{1, 4, 20, 20}, {1, 4, 16, 16}, {1, 4, 13, 13}, {1, 4, 10, 10}, {1, 4, 8, 8}, {1, 4, 6, 6}, {1, 4, 5, 5}, {1, 4, 3, 3}, {1, 4, 2, 2}};
	int rnet_out_shape[2] = {17, 4};
	int onet_out_shape[3][2] = {{6, 4}, {6, 10}, {6, 2}};

	const char* pnet_out_file[18] = {"pout0.0", "pout0.1", "pout1.0", "pout1.1", "pout2.0", "pout2.1", "pout3.0", "pout3.1", "pout4.0",\
									"pout4.1", "pout5.0", "pout5.1", "pout6.0", "pout6.1", "pout7.0", "pout7.1", "pout8.0", "pout8.1"};
	const char* rnet_out_file[2] = {"rout0.bin", "rout1.bin"};
	const char* onet_out_file[3] = {"oout0.bin", "oout1.bin", "oout2.bin"};
	int out_file_index = 0;

	unsigned int numbox = 0;
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

	for (i = 7; i < scales_len; i++) {
		scale = scales[i];
		hs = (int)ceil(h * scale);
		ws = (int)ceil(w * scale);

		Mat im_data = imresample(img, hs, ws, (unsigned char)1);
		Mat img_y = get_img_y(&im_data);

		/* out = pnet(img_y) */

		Mat out0 = get_pnet_out(pnet_out_shape[i-7], pnet_out_file[out_file_index], 0);
		out_file_index++;
		Mat out1 = get_pnet_out(pnet_out_shape[i-7], pnet_out_file[out_file_index], 1);
		out_file_index++;

		Mat in0 = get_in0(&out0);
		Mat in1 = get_in1(&out1);

		Mat boxes = generateBoundingBox(&in1, &in0, scale, threshold[0]);
		if (boxes.rows * boxes.cols == 0)
			break;

		pick = nms(&boxes, 0.5, "Union", &pick_len);
		if (pick == NULL) {
			printf("nms failed\n");
			return boxes;
		}

		if ((boxes.rows * boxes.cols > 0) && (pick_len > 0)) {
			Mat boxes1 = get_boxes_from_pick(&boxes, pick, pick_len);
			total_boxes = append_total_boxes(&total_boxes, &boxes1);
		}

		free(pick);
	}

	numbox = total_boxes.rows;

	if (numbox > 0) {
		pick = nms(&total_boxes, 0.7, "Union", &pick_len);
		total_boxes = get_boxes_from_pick(&total_boxes, pick, pick_len);
		len = total_boxes.rows;

		double* regw = mat_cols_sub(total_boxes.colRange(2, 3), total_boxes.colRange(0, 1));
		double* regh = mat_cols_sub(total_boxes.colRange(3, 4), total_boxes.colRange(1, 2));

		double* qq1 = get_qq(&total_boxes, 0, 5, regw);
		double* qq2 = get_qq(&total_boxes, 1, 6, regh);
		double* qq3 = get_qq(&total_boxes, 2, 7, regw);
		double* qq4 = get_qq(&total_boxes, 3, 8, regh);

		total_boxes = get_vstack_qq_and_transpose(qq1, qq2, qq3, qq4, &total_boxes, 4);

		rerec(&total_boxes);

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
			printf("*********************************\n");
			printf("****malloc error in line %d****\n", __LINE__);
			printf("*********************************\n");
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
		Mat tempimg = Mat(3, Mat_init_size, CV_64FC(numbox), Scalar::all(0));

		for (i = 0; i < numbox; i++) {
			Mat tmp = Mat::zeros(tmph[i], tmpw[i], CV_64FC3);

			buckle_map(img, &tmp, x, ex, y, ey, dx, edx, dy, edy, i);
			if (((tmp.rows > 0) && (tmp.cols > 0)) || ((tmp.rows == 0) && (tmp.cols == 0))) {
				Mat tmp_tempimg = imresample(&tmp, 24, 24, (double)1);
				get_tempimg(&tempimg, &tmp_tempimg, i, Mat_init_size[2]);
			} else {
				printf("buckle map execute failed!\n");
				return tmp;
			}
		}

		image_normalization(&tempimg, Mat_init_size[2], (double)1);
		Mat tempimg1 = transpose3021(&tempimg, Mat_init_size[2]);

		/* out = rnet(tempimg1); */

		Mat out0 = get_rnet_out(rnet_out_shape, rnet_out_file[0], 0);
		Mat out1 = get_rnet_out(rnet_out_shape, rnet_out_file[1], 1);

		transpose(out0, out0);
		transpose(out1, out1);

		len = out0.cols;
		Mat score = out1.rowRange(1, 2).clone();
		int* ipass = get_ipass(&score, threshold[1], len, &ipass_len);

		total_boxes = get_hstack_ronet(&total_boxes, ipass, 0, 4, &score, ipass_len);

		Mat mv = get_mv(&out0, ipass, ipass_len);

		if (total_boxes.rows > 0) {
			pick = nms(&total_boxes, 0.7, "Union", &pick_len);
			total_boxes = get_total_boxes_pick(&total_boxes, pick, pick_len);
			mv = transpose_mv_piack(&mv, pick, pick_len);
			bbreg(&total_boxes, &mv);
			rerec(&total_boxes);
		}

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
			printf("*********************************\n");
			printf("****malloc error in line %d****\n", __LINE__);
			printf("*********************************\n");
		}

		total_boxes = fix_total_boxes(&total_boxes);
		pad(&total_boxes, h, w, dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph);
		Mat_init_size[0] = 3;
		Mat_init_size[1] = 48;
		Mat_init_size[2] = 48;
		Mat tempimg = Mat(3, Mat_init_size, CV_64FC(numbox), Scalar::all(0));

		for (i = 0; i < numbox; i++) {
			Mat tmp = Mat::zeros(tmph[i], tmpw[i], CV_64FC3);

			buckle_map(img, &tmp, x, ex, y, ey, dx, edx, dy, edy, i);
			if (((tmp.rows > 0) && (tmp.cols > 0)) || ((tmp.rows == 0) && (tmp.cols == 0))) {
				Mat tmp_tempimg = imresample(&tmp, 48, 48, (double)1);
				get_tempimg(&tempimg, &tmp_tempimg, i, Mat_init_size[2]);
			} else {
				return tmp;
			}
		}

		image_normalization(&tempimg, Mat_init_size[2], (double)1);
		Mat tempimg1 = transpose3021(&tempimg, Mat_init_size[2]);

		Mat out0 = get_onet_out(onet_out_shape[0], onet_out_file[0]);
		Mat out1 = get_onet_out(onet_out_shape[1], onet_out_file[1]);
		Mat out2 = get_onet_out(onet_out_shape[2], onet_out_file[2]);

		transpose(out0, out0);
		transpose(out1, out1);
		transpose(out2, out2);

		len = out0.cols;
		Mat score = out2.rowRange(1, 2).clone();
		int* ipass = get_ipass(&score, threshold[2], len, &ipass_len);
		Mat points = get_points(&out1, ipass, ipass_len);

		total_boxes = get_hstack_ronet(&total_boxes, ipass, 0, 4, &score, ipass_len);

		Mat mv = get_mv(&out0, ipass, ipass_len);

		double* w = (double*)malloc(len * sizeof(double));
		double* h = (double*)malloc(len * sizeof(double));
		if (w == NULL || h == NULL) {
			printf("*********************************\n");
			printf("****malloc error in line %d****\n", __LINE__);
			printf("*********************************\n");
		}

		get_wh_in_bbreg(&total_boxes, w, h, len);

		update_points(&points, &total_boxes, w, h, len);

		if (total_boxes.rows > 0) {
			transpose(mv, mv);
			bbreg(&total_boxes, &mv);
			pick = nms(&total_boxes, 0.7, "Min", &pick_len);
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

	return total_boxes;
}
