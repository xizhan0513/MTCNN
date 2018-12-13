#include "mtcnn.h"

using namespace std;
using namespace cv;

Mat detect_face(Mat* img, float* threshold, double* scales, int scales_len, Mat* ret_points)
{
	int i = 0;
	int h = img->rows;
	int w = img->cols;
	int hs = 0, ws = 0;
	double scale = 0;
	short* pick = NULL;
	int pick_len = 0;
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

		Mat boxes1 = generateBoundingBox(&in1, &in0, scale, threshold[0]);

		pick = nms(&boxes1, 0.5, "Union", &pick_len);
		if (pick == NULL) {
			printf("nms failed\n");
			return boxes1;
		}

		if ((boxes1.rows * boxes1.cols > 0) && (pick_len > 0)) {
			Mat boxes2 = get_boxes_from_pick(&boxes1, pick, pick_len);
			total_boxes = get_total_boxes(&total_boxes, &boxes2);
		}

		free(pick);
	}

	unsigned int numbox = total_boxes.rows;
	double* regw = NULL;
	double* regh = NULL;
	double* qq1 = NULL;
	double* qq2 = NULL;
	double* qq3 = NULL;
	double* qq4 = NULL;
	int* dy = (int*)malloc(total_boxes.rows * sizeof(int));
	int* edy = (int*)malloc(total_boxes.rows * sizeof(int));
	int* dx = (int*)malloc(total_boxes.rows * sizeof(int));
	int* edx = (int*)malloc(total_boxes.rows * sizeof(int));
	int* y = (int*)malloc(total_boxes.rows * sizeof(int));
	int* ey = (int*)malloc(total_boxes.rows * sizeof(int));
	int* x = (int*)malloc(total_boxes.rows * sizeof(int));
	int* ex = (int*)malloc(total_boxes.rows * sizeof(int));
	int* tmpw = (int*)malloc(total_boxes.rows * sizeof(int));
	int* tmph = (int*)malloc(total_boxes.rows * sizeof(int));

	if (numbox > 0) {
		pick = nms(&total_boxes, 0.7, "Union", &pick_len);
		total_boxes = get_boxes_from_pick(&total_boxes, pick, pick_len);
		int len = total_boxes.rows;
		regw = get_reg_wh(&total_boxes, 2, 0);
		regh = get_reg_wh(&total_boxes, 3, 1);
		qq1 = get_qq(&total_boxes, 0, 5, regw);
		qq2 = get_qq(&total_boxes, 1, 6, regh);
		qq3 = get_qq(&total_boxes, 2, 7, regw);
		qq4 = get_qq(&total_boxes, 3, 8, regh);

		total_boxes = get_vstack_qq_and_transpose(qq1, qq2, qq3, qq4, &total_boxes, 4);

		rerec(&total_boxes);

		get_total_boxeses_fix(&total_boxes, 0, 4, 0, 4);

		pad(&total_boxes, h, w, dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph);

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
				return tmp;
			}
		}

		image_normalization(&tempimg, Mat_init_size[2], (double)1);
		Mat tempimg1 = transpose3021(&tempimg, Mat_init_size[2]);

		Mat out0 = get_rnet_out(rnet_out_shape, rnet_out_file[0], 0);
		Mat out1 = get_rnet_out(rnet_out_shape, rnet_out_file[1], 1);

		transpose(out0, out0);
		transpose(out1, out1);
		int len = out0.cols;
		float* score = get_score_out(&out1, 1, len);
		int ipass_len = 0;
		int* ipass = get_ipass(score, threshold[1], len, &ipass_len);

		total_boxes = get_hstack_rnet(&total_boxes, ipass, 0, 4, score, ipass_len);

		Mat mv = get_mv(&out0, ipass, ipass_len);

		if (total_boxes.rows > 0) {
			pick = nms(&total_boxes, 0.7, "Union", &pick_len);
			total_boxes = get_total_boxeses_pick(&total_boxes, pick, pick_len);
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
		free(score);
	}

	numbox = total_boxes.rows;
	if (numbox > 0) {
		dy = (int*)malloc(total_boxes.rows * sizeof(int));
		edy = (int*)malloc(total_boxes.rows * sizeof(int));
		dx = (int*)malloc(total_boxes.rows * sizeof(int));
		edx = (int*)malloc(total_boxes.rows * sizeof(int));
		y = (int*)malloc(total_boxes.rows * sizeof(int));
		ey = (int*)malloc(total_boxes.rows * sizeof(int));
		x = (int*)malloc(total_boxes.rows * sizeof(int));
		ex = (int*)malloc(total_boxes.rows * sizeof(int));
		tmpw = (int*)malloc(total_boxes.rows * sizeof(int));
		tmph = (int*)malloc(total_boxes.rows * sizeof(int));

		total_boxes = fix_total_boxeses(&total_boxes);
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

		int len = out0.cols;
		float* score = get_score_out(&out2, 1, len);
		int ipass_len = 0;
		int* ipass = get_ipass(score, threshold[2], len, &ipass_len);
		Mat points = get_points(&out1, ipass, ipass_len);

		total_boxes = get_hstack_rnet(&total_boxes, ipass, 0, 4, score, ipass_len);

		Mat mv = get_mv(&out0, ipass, ipass_len);

		double* w = (double*)malloc(len * sizeof(double));
		double* h = (double*)malloc(len * sizeof(double));
		get_wh_bbreg(&total_boxes, w, h, len);

		updata_points(&points, &total_boxes, w, h, len);

		if (total_boxes.rows > 0) {
			transpose(mv, mv);
			bbreg(&total_boxes, &mv);
			pick = nms(&total_boxes, 0.7, "Min", &pick_len);
			total_boxes = get_boxes_from_pick(&total_boxes, pick, pick_len);

			*ret_points = points_pick(&points, pick, pick_len);
		}

		free(w);
		free(h);
		free(score);
		free(ipass);
	}
		free(pick);
		free(regw);
		free(regh);
		free(qq1);
		free(qq2);
		free(qq3);
		free(qq4);

	return total_boxes;
}
