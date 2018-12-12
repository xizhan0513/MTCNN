#include <stdio.h>
#include <math.h>
#include "mtcnn.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Mat detect_face(Mat* img, float* threshold, double* scales, int scales_len, Mat* pointss)
{
	int i = 0;
	int h = img->rows;
	int w = img->cols;
	int hs = 0, ws = 0;
	double scale = 0;
	Mat total_box;
	int points[10][6] = {0};
	int Mat_init_size[3] = {0};
	int pnet_init_arr[9][4] = {{1, 4, 20, 20}, {1, 4, 16, 16}, {1, 4, 13, 13}, {1, 4, 10, 10}, {1, 4, 8, 8}, {1, 4, 6, 6}, {1, 4, 5, 5}, {1, 4, 3, 3}, {1, 4, 2, 2}};
	int pack_len = 0;
	short* pack = NULL;

	const char* out_file[18] = {"0.0", "0.1", "1.0", "1.1", "2.0", "2.1", "3.0", "3.1", "4.0", "4.1", "5.0", "5.1", "6.0", "6.1", "7.0", "7.1", "8.0", "8.1"};
	int out_file_index = 0;

	for(i = 7; i < scales_len; i++) {
		scale = scales[i];
		hs = (int)ceil(h * scale);
		ws = (int)ceil(w * scale);

		Mat im_data = imresample_uchar(img, hs, ws);
		Mat img_y = get_img_y(&im_data);

		/* out = pnet(img_y) */

		Mat out0 = get_pnet_out(pnet_init_arr[i-7], out_file[out_file_index], 0);
		out_file_index++;
		Mat out1 = get_pnet_out(pnet_init_arr[i-7], out_file[out_file_index], 1);
		out_file_index++;

		Mat in0 = get_in0(&out0);
		Mat in1 = get_in1(&out1);

		Mat boxes1 = generateBoundingBox(&in1, &in0, scale, threshold[0]);

		pack = nms(&boxes1, 0.5, "Union", &pack_len);
		if (pack == NULL) {
			printf("nms failed\n");
			return boxes1;
		}

		if ((boxes1.rows * boxes1.cols > 0) && (pack_len > 0)) {
			Mat boxes2 = get_boxes_from_pack(&boxes1, pack, pack_len);
			total_box = get_total_box(&total_box, &boxes2);
		}

		free(pack);
	}

	unsigned int numbox = total_box.rows;
	double* regw = NULL;
	double* regh = NULL;
	double* qq1 = NULL;
	double* qq2 = NULL;
	double* qq3 = NULL;
	double* qq4 = NULL;
	int* dy = (int*)malloc(total_box.rows * sizeof(int));
	int* edy = (int*)malloc(total_box.rows * sizeof(int));
	int* dx = (int*)malloc(total_box.rows * sizeof(int));
	int* edx = (int*)malloc(total_box.rows * sizeof(int));
	int* y = (int*)malloc(total_box.rows * sizeof(int));
	int* ey = (int*)malloc(total_box.rows * sizeof(int));
	int* x = (int*)malloc(total_box.rows * sizeof(int));
	int* ex = (int*)malloc(total_box.rows * sizeof(int));
	int* tmpw = (int*)malloc(total_box.rows * sizeof(int));
	int* tmph = (int*)malloc(total_box.rows * sizeof(int));
	printf("%d \n", total_box.rows);

	if (numbox > 0) {
		pack = nms(&total_box, 0.7, "Union", &pack_len);
		total_box = get_boxes_from_pack(&total_box, pack, pack_len);
		int len = total_box.rows;
		regw = get_reg_wh(&total_box, 2, 0);
		regh = get_reg_wh(&total_box, 3, 1);
		qq1 = get_qq(&total_box, 0, 5, regw);
		qq2 = get_qq(&total_box, 1, 6, regh);
		qq3 = get_qq(&total_box, 2, 7, regw);
		qq4 = get_qq(&total_box, 3, 8, regh);

		total_box = get_vstack_qq_and_transpose(qq1, qq2, qq3, qq4, &total_box, 4);

		rerec(&total_box);

		get_total_boxes_fix(&total_box, 0, 4, 0, 4);

		pad(&total_box, h, w, dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph);

		free(pack);
	}

	numbox = total_box.rows;
	if (numbox > 0) {
		Mat_init_size[0] = 3;
		Mat_init_size[1] = 24;
		Mat_init_size[2] = 24;
		Mat tempimg = Mat(3, Mat_init_size, CV_64FC(numbox), Scalar::all(0));

		for (i = 0; i < numbox; i++) {
			Mat tmp = Mat::zeros(tmph[i], tmpw[i], CV_64FC3);

			buckle_map(img, &tmp, x, ex, y, ey, dx, edx, dy, edy, i);
			if (((tmp.rows > 0) && (tmp.cols > 0)) || ((tmp.rows == 0) && (tmp.cols == 0))) {
				Mat tmp_tempimg = imresample_double(&tmp, 24, 24);
				get_tempimg(&tempimg, &tmp_tempimg, i, Mat_init_size[2]);
			} else {
				return tmp;
			}
		}

		image_normalization_double(&tempimg, Mat_init_size[2]);
		Mat tempimg1 = transpose3021(&tempimg, Mat_init_size[2]);

		int rnet_init_arr[2] = {17, 4};
		Mat out0 = get_rnet_out(rnet_init_arr, "rout0.bin", 0);
		Mat out1 = get_rnet_out(rnet_init_arr, "rout1.bin", 1);

		transpose(out0, out0);
		transpose(out1, out1);
		int len = out0.cols;
		float* score = get_score_out(&out1, 1, len);
		int ipass_len = 0;
		int* ipass = get_ipass(score, threshold[1], len, &ipass_len);

		total_box = get_hstack_rnet(&total_box, ipass, 0, 4, score, ipass_len);

		Mat mv = get_mv(&out0, ipass, ipass_len);

		if (total_box.rows > 0) {
			pack = nms(&total_box, 0.7, "Union", &pack_len);
			total_box = get_total_boxes_pick(&total_box, pack, pack_len);
			mv = transpose_mv_piack(&mv, pack, pack_len);
			bbreg(&total_box, &mv);
			rerec(&total_box);
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
		free(pack);
		free(ipass);
		free(score);
	}

	numbox = total_box.rows;
	if (numbox > 0) {
		dy = (int*)malloc(total_box.rows * sizeof(int));
		edy = (int*)malloc(total_box.rows * sizeof(int));
		dx = (int*)malloc(total_box.rows * sizeof(int));
		edx = (int*)malloc(total_box.rows * sizeof(int));
		y = (int*)malloc(total_box.rows * sizeof(int));
		ey = (int*)malloc(total_box.rows * sizeof(int));
		x = (int*)malloc(total_box.rows * sizeof(int));
		ex = (int*)malloc(total_box.rows * sizeof(int));
		tmpw = (int*)malloc(total_box.rows * sizeof(int));
		tmph = (int*)malloc(total_box.rows * sizeof(int));

		total_box = fix_total_boxes(&total_box);
		pad(&total_box, h, w, dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph);
		Mat_init_size[0] = 3;
		Mat_init_size[1] = 48;
		Mat_init_size[2] = 48;
		Mat tempimg = Mat(3, Mat_init_size, CV_64FC(numbox), Scalar::all(0));


		for (i = 0; i < numbox; i++) {
			Mat tmp = Mat::zeros(tmph[i], tmpw[i], CV_64FC3);

			buckle_map(img, &tmp, x, ex, y, ey, dx, edx, dy, edy, i);
			if (((tmp.rows > 0) && (tmp.cols > 0)) || ((tmp.rows == 0) && (tmp.cols == 0))) {
				Mat tmp_tempimg = imresample_double(&tmp, 48, 48);
				get_tempimg(&tempimg, &tmp_tempimg, i, Mat_init_size[2]);
			} else {
				return tmp;
			}
		}

		image_normalization_double(&tempimg, Mat_init_size[2]);
		Mat tempimg1 = transpose3021(&tempimg, Mat_init_size[2]);

		int onet_init_arr[3][2] = {{6, 4}, {6, 10}, {6, 2}};
		Mat out0 = get_onet_out(onet_init_arr[0], "oout0.bin");
		Mat out1 = get_onet_out(onet_init_arr[1], "oout1.bin");
		Mat out2 = get_onet_out(onet_init_arr[2], "oout2.bin");

		transpose(out0, out0);
		transpose(out1, out1);
		transpose(out2, out2);

		int len = out0.cols;
		float* score = get_score_out(&out2, 1, len);
		int ipass_len = 0;
		int* ipass = get_ipass(score, threshold[2], len, &ipass_len);
		Mat points = get_points(&out1, ipass, ipass_len);

		total_box = get_hstack_rnet(&total_box, ipass, 0, 4, score, ipass_len);

		Mat mv = get_mv(&out0, ipass, ipass_len);

		double* w = (double*)malloc(len * sizeof(double));
		double* h = (double*)malloc(len * sizeof(double));
		get_wh_bbreg(&total_box, w, h, len);

		updata_points(&points, &total_box, w, h, len);

		if (total_box.rows > 0) {
			transpose(mv, mv);
			bbreg(&total_box, &mv);
			pack = nms(&total_box, 0.7, "Min", &pack_len);
			total_box = get_boxes_from_pack(&total_box, pack, pack_len);

			*pointss = points_pick(&points, pack, pack_len);
		}

		free(w);
		free(h);
		free(score);
		free(ipass);
	}
		free(pack);
		free(regw);
		free(regh);
		free(qq1);
		free(qq2);
		free(qq3);
		free(qq4);

	return total_box;
}
