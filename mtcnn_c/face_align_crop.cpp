#include "mtcnn.h"
#include <sys/time.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	if (argc != 4) {
		printf("Usage: ./test input_file pb_file output_file\n");
		return -1;
	}

	struct timeval start, end;
	int ret_start = 0, ret_end = 0;
	unsigned int delta_ms = 0;

	ret_start = gettimeofday(&start, NULL);

	int minsize = 20;
	float threshold[3] = {0.8, 0.85, 0.9};
	float factor = 0.85;

	Mat img = imread(argv[1]);
	/* bgr -> rgb */
	cvtColor(img, img, CV_BGR2RGB, img.channels());

	/* 判断数组维度 */
	if (img.dims != 2) {
		printf("Unable to align %s, img dim error", argv[1]);
	} else {
		int i = 0;
		int factor_count = 0;
		int h = img.rows;
		int w = img.cols;
		double minl = h < w ? h : w;
		float m = 12.0 / minsize;
		double scales[SCALES_LEN] = {0};
		int scales_len = SCALES_LEN;

		minl = minl * m;

		while (minl >= 12) {
			scales[i] = m * pow(factor, factor_count);
			minl = minl * factor;
			factor_count += 1;
			i++;
		}

		Mat points;
		Mat bounding_boxes = detect_face(&img, threshold, scales, scales_len, &points);
		if (bounding_boxes.rows * bounding_boxes.cols == 0) {
			printf("No faces in the picture!!!\n");
			return -1;
		}

		print_2D(&bounding_boxes, (double)1);
		printf("----------------------------\n");
		print_2D(&points, (float)1);
		Mat _landmark = points.reshape(0, 2).t();

		Mat warped = face_preprocess(&img, &_landmark);
		cvtColor(warped, warped, CV_RGB2BGR, warped.channels());
		imwrite(argv[3], warped);
	}

	ret_end = gettimeofday(&end, NULL);

	delta_ms = (end.tv_sec * 1000 + end.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);
	printf("time:%dms\n", delta_ms);

	return 0;
}
