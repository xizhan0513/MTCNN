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
	int Mat_init_size [3] = {0};
	int pnet_init_arr[9][4] = {{1, 4, 20, 20}, {1, 4, 16, 16}, {1, 4, 13, 13}, {1, 4, 10, 10}, {1, 4, 8, 8}, {1, 4, 6, 6}, {1, 4, 5, 5}, {1, 4, 3, 3}, {1, 4, 2, 2}};
	int pack_len = 0;

	const char* out_file[18] = {"0.0", "0.1", "1.0", "1.1", "2.0", "2.1", "3.0", "3.1", "4.0", "4.1", "5.0", "5.1", "6.0", "6.1", "7.0", "7.1", "8.0", "8.1"};
	int out_file_index = 0;

	for(i = 7; i < scales_len; i++) {
		scale = scales[i];
		hs = (int)ceil(h * scale);
		ws = (int)ceil(w * scale);

		Mat im_data = get_img_data(img, hs, ws);
		Mat img_y = get_img_y(&im_data);

		/* out = pnet(img_y) */

		Mat out0 = get_pnet_out(pnet_init_arr[i-7], out_file[out_file_index], 0);
		out_file_index++;
		Mat out1 = get_pnet_out(pnet_init_arr[i-7], out_file[out_file_index], 1);
		out_file_index++;

		Mat in0 = get_in0(&out0);
		Mat in1 = get_in1(&out1);

		Mat boxes1 = generateBoundingBox(&in1, &in0, scale, threshold[0]);

		short* pack = nms(&boxes1, 0.5, "Union", &pack_len);
		if (pack == NULL) {
			printf("nms failed\n");
			return -1;
		}

		if ((boxes1.rows * boxes1.cols > 0) && (pack_len > 0)) {
			Mat boxes2 = get_boxes2(&boxes1, pack, pack_len);
			total_box = get_total_box(&total_box, &boxes2);
		}

		free(pack);
	}

	return 0;
}
