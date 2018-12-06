#include <stdio.h>
#include <math.h>
#include "mtcnn.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	if (argc != 4) {
		printf("Usage: ./a.out input_path pb_path output_path\n");
		return -1;
	}

	unsigned int minsize = 20;
	float threshold[3] = {0.8, 0.85, 0.9};
	float factor = 0.85;
	Mat img = imread(argv[1]);

	/* bgr -> rgb */
	cvtColor(img, img, CV_BGR2RGB, 3);

	/* 判断数组维度 */
	if (img.dims != 2) {
		printf("Unable to align %s, img dim error", argv[1]);
	} else {
		int i = 0;
		unsigned int factor_count = 0;
		unsigned int h = img.rows;
		unsigned int w = img.cols;
		long int minl = h < w ? h : w;
		float m = 12.0 / minsize;
		double scales[16] = {0};
		int scales_len = sizeof(scales) / sizeof(scales[0]);

		minl = minl * m;

		while (minl >= 12) {
			scales[i] = m * pow(factor, factor_count);
			minl = minl * factor;
			factor_count += 1;
			i++;
		}

		int ret = detect_face(&img, threshold, scales, scales_len);
	}

	return 0;
}
