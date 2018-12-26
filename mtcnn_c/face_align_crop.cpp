#include "mtcnn.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	if (argc != 3) {
		printf("Usage: ./test input_file output_path\n");
		return -1;
	}

	struct timeval start, end;
	int ret_start = 0, ret_end = 0;
	unsigned int delta_ms = 0;

	ret_start = gettimeofday(&start, NULL);
	if (ret_start != 0) {
		printf("gettimeofday failed!\n");
		return -1;
	}

	int minsize = 20;
	float threshold[3] = {0.8, 0.85, 0.9};
	float factor = 0.85;
	int face_count = 0;
	char output_file_name[50] = {0};

	Mat img = imread(argv[1]);
	/* bgr -> rgb */
	cvtColor(img, img, CV_BGR2RGB, img.channels());

	/* 判断数组维度 */
	if (img.dims != 2) {
		printf("Unable to align %s, img dim error", argv[1]);
		return -1;
	} else {
		int i = 0;
		int factor_count = 0;
		int h = img.rows;
		int w = img.cols;
		float minl = h < w ? h : w;
		float m = 12.0 / minsize;
		float scales[SCALES_LEN] = {0};
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
			printf("detect_face function return error!!!\n");
			return -1;
		}

		print_2D(&bounding_boxes, (float)1);
		printf("----------------------------\n");
		print_2D(&points, (float)1);

		face_count = points.cols > MAX_FACE_NUM ? MAX_FACE_NUM : points.cols;

		Mat sort_result;
		sortIdx(bounding_boxes.colRange(4, 5).clone(), sort_result, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

		for (i = 0; i < points.cols; i++) {
			if (*sort_result.ptr<int>(i) < face_count) {
				Mat _landmark = points.colRange(*sort_result.ptr<int>(i), *sort_result.ptr<int>(i) + 1).clone().reshape(0, 2).t();

				Mat warped = face_preprocess(&img, &_landmark);
				cvtColor(warped, warped, CV_RGB2BGR, warped.channels());
				sprintf(output_file_name, "%s/%d.jpg", argv[2], *sort_result.ptr<int>(i) + 1);
				imwrite(output_file_name, warped);
			}
		}
	}

	ret_end = gettimeofday(&end, NULL);
	if (ret_end != 0) {
		printf("gettimeofday failed!\n");
		return -1;
	}

	delta_ms = (end.tv_sec * 1000 + end.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);
	printf("time:%dms\n", delta_ms);

	return 0;
}
