#include <stdio.h>
#include <math.h>
#include "mtcnn.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int detect_face(Mat* img, float* threshold, double* scales, int scales_len)
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
			return -1;
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
				return -1;
			}
		}

		image_normalization_double(&tempimg, Mat_init_size[2]);

		Mat tempimg1 = transpose3021(&tempimg, Mat_init_size[2]);
	}



		/*int x = 0;
		for (x = 0; x < len; x++) {
			printf("%.10f ", qq1[x]);
		}
		printf("\n");
*/
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
		free(regw);
		free(regh);
		free(qq1);
		free(qq2);
		free(qq3);
		free(qq4);

	return 0;
}
