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
	double total_box[20][9] = {0};
	int points[10][6] = {0};
	int Mat_init_size [3] = {0};
	int pnet_init_arr[9][4] = {{1, 4, 20, 20}, {1, 4, 16, 16}, {1, 4, 13, 13}, {1, 4, 10, 10}, {1, 4, 8, 8}, {1, 4, 6, 6}, {1, 4, 5, 5}, {1, 4, 3, 3}, {1, 4, 2, 2}};
	int pack_len = 0;

	for(i = 7; i <= scales_len; i++) {
		scale = scales[i];
		hs = (int)ceil(h * scale);
		ws = (int)ceil(w * scale);

		Mat im_data = get_img_data(img, hs, ws);
		Mat img_y = get_img_y(&im_data);

		/* out = pnet(img_y) */

		Mat out0 = get_pnet_out(pnet_init_arr[i-7], "out0.bin", 0);
		Mat out1 = get_pnet_out(pnet_init_arr[i-7], "out1.bin", 1);

		Mat in0 = get_in0(&out0);
		Mat in1 = get_in1(&out1);

		Mat boxes = generateBoundingBox(&in1, &in0, scale, threshold[0]);

		short* pack = nms(&boxes, 0.5, "Union", &pack_len);
		if (pack == NULL) {
			printf("nms failed\n");
			return -1;
		}
		return 0;
	}
}
