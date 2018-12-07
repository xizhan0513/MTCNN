#include <stdio.h>
#include <math.h>
#include "mtcnn.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat image_normalization(Mat* img)
{
	int i = 0, j = 0, k = 0;
	unsigned char* ptr_uchar = NULL;
	double* ptr_double = NULL;
	Mat ret_img = Mat::zeros(img->rows, img->cols, CV_64FC(img->channels()));

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			ptr_uchar = img->ptr<uchar>(i, j);
			ptr_double = ret_img.ptr<double>(i ,j);
			for (k = 0; k < ret_img.channels(); k++) {
				*ptr_double = (*ptr_uchar-127.5) * 0.0078125;
				ptr_uchar++;
				ptr_double++;
			}
		}
	}

	return ret_img;
}

Mat transpose_uchar_201(Mat* img)
{
	int i = 0, j = 0, k = 0;
	int x = 0, y = 0, z = 0;
	int xe = 0, ye = 0;
	unsigned char* ptr = NULL;
	Mat ret_img = Mat::zeros(img->channels(), img->rows, CV_8UC(img->cols));

	xe = img->rows;
	ye = img->cols;

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			ptr = ret_img.ptr<uchar>(i, j);
			for (k = 0; k < ret_img.channels(); k++) {
				*ptr = *((img->data + img->step[0]*x + img->step[1]*y) + z);
				ptr++;
				y++;
				if (y == ye) {
					y = 0;
					x++;
					if (x == xe) {
						z++;
						x = 0;
					}
				}
			}
		}
	}

	return image_normalization(&ret_img);
}

Mat transpose_double_021(Mat* img)
{
	int i = 0, j = 0, k = 0;
	int x = 0, y = 0, z = 0;
	int ye = 0, ze = 0;
	double* ptr_double = NULL;
	Mat ret_img = Mat::zeros(img->rows, img->channels(), CV_64FC(img->cols));

	ye = img->cols;
	ze = img->channels();

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			ptr_double = ret_img.ptr<double>(i, j);
			for (k = 0; k < ret_img.channels(); k++) {
				*ptr_double = *((double*)(img->data + img->step[0]*x + img->step[1]*y) + z);
				ptr_double++;
				y++;
				if (y == ye) {
					y = 0;
					z++;
					if (z == ze) {
						x++;
						z = 0;
					}
				}
			}
		}
	}

	return ret_img;
}

Mat transpose_float_021(Mat* img)
{
	int i = 0, j = 0, k = 0;
	int x = 0, y = 0, z = 0;
	int ye = 0, ze = 0;
	float* ptr_float = NULL;
	Mat ret_img = Mat::zeros(img->rows, img->channels(), CV_32FC(img->cols));

	ye = img->cols;
	ze = img->channels();

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			ptr_float = ret_img.ptr<float>(i, j);
			for (k = 0; k < ret_img.channels(); k++) {
				*ptr_float = *((float*)(img->data + img->step[0]*x + img->step[1]*y) + z);
				ptr_float++;
				y++;
				if (y == ye) {
					y = 0;
					z++;
					if (z == ze) {
						x++;
						z = 0;
					}
				}
			}
		}
	}

	return ret_img;
}

Mat expand_dims(Mat* img)
{
	int i = 0, j = 0, k = 0, v = 0;
	double* ptr_double_src = NULL;
	double* ptr_double_dst = NULL;
	int Mat_init_size[3] = {0};
	Mat_init_size[0] = 1;
	Mat_init_size[1] = img->rows;
	Mat_init_size[2] = img->cols;

	Mat ret_img = Mat(3, Mat_init_size, CV_64FC(img->channels()), Scalar::all(0));

	for (i = 0; i < 1; i++) {
		for (j = 0; j < img->rows; j++) {
			for (k = 0; k < img->cols; k++) {
				ptr_double_src = img->ptr<double>(j, k);
				ptr_double_dst = (double*)(ret_img.data + ret_img.step[0]*i + ret_img.step[1]*j + ret_img.step[2]*k);
				for (v = 0; v < img->channels(); v++) {
					*ptr_double_dst = *ptr_double_src;
					ptr_double_src++;
					ptr_double_dst++;
				}
			}
		}
	}

	return ret_img;
}

Mat get_float_2D(Mat* img, int count)
{
	int i = 0, j = 0;
	float* ptr_float = NULL;
	float* ptr_float_ret = NULL;
	Mat ret_img = Mat::zeros(img->cols, img->channels(), CV_32FC1);

	for (i = 0; i < ret_img.rows; i++) {
		ptr_float = img->ptr<float>(count, i);
		ptr_float_ret = ret_img.ptr<float>(i);
		for (j = 0; j < ret_img.cols; j++) {
			*ptr_float_ret = *ptr_float;
			ptr_float++;
			ptr_float_ret++;
		}
	}

	return ret_img;
}

Mat get_img_data(Mat* img, int hs, int ws)
{
	Mat tmp_img;
	resize(*img, tmp_img, Size(ws, hs), 0, 0, INTER_AREA);
	return transpose_uchar_201(&tmp_img);
}

Mat get_img_y(Mat* img)
{
	Mat img_x = transpose_double_021(img);
	return expand_dims(&img_x);
}

Mat get_pnet_out(int* pnet_init_arr, char* str, int flag)
{
	int i = 0, j = 0, k = 0, l = 0;
	float* ptr_float = NULL;

	FILE* f = fopen(str, "rb");
	Mat ret_img = Mat::zeros(flag == 0 ? pnet_init_arr[1] : (pnet_init_arr[1] - 2), pnet_init_arr[2], CV_32FC(pnet_init_arr[3]));

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			ptr_float = ret_img.ptr<float>(i, j);
			for (k = 0; k < ret_img.channels(); k++) {
				fread(ptr_float, 4, 1, f);
				ptr_float++;
			}
		}
	}

	return ret_img;
}

Mat get_in0(Mat* img)
{
	return transpose_float_021(img);
}

Mat get_in1(Mat* img)
{
	Mat tmp_img = transpose_float_021(img);
	return get_float_2D(&tmp_img, 1);
}

void get_xy(Mat* img, int** x, int** y, float t, int* xy_len)
{
	int i = 0, j = 0;
	int count = 0;
	float* ptr_float = NULL;

	for(i = 0; i < img->rows; i++) {
		ptr_float = img->ptr<float>(i);
		for(j = 0; j < img->cols; j++) {
			if (*ptr_float >= t)
				count++;
			ptr_float++;
		}
	}

	*xy_len = count;

	while (*x == NULL) {
		*x = (int*)malloc(count * sizeof(int));
	}
	while (*y == NULL) {
		*y = (int*)malloc(count * sizeof(int));
	}

	int* x_tmp = *x;
	int* y_tmp = *y;

	for(i = 0; i < img->rows; i++) {
		ptr_float = img->ptr<float>(i);
		for(j = 0; j < img->cols; j++) {
			if (*ptr_float >= t) {
				*x_tmp = j;
				*y_tmp = i;
				if (count > 0) {
					x_tmp++;
					y_tmp++;
					count--;
				}
			}
			ptr_float++;
		}
	}

	return ;
}

float* get_score(Mat* img, int* y, int *x, int xy_len)
{
	int i = 0;
	float* ret_ptr = NULL;
	while (ret_ptr == NULL) {
		ret_ptr = (float*)malloc(xy_len * sizeof(float));
	}

	for (i = 0; i < xy_len; i++) {
		ret_ptr[i] = *(img->ptr<float>(y[i], x[i]));
	}

	return ret_ptr;
}

Mat get_vstack(Mat* dx1, Mat* dy1, Mat* dx2, Mat* dy2, int* y, int* x, int xy_len)
{
	int i = 0;
	Mat ret_img = Mat::zeros(xy_len, xy_len, CV_32FC1);

	for (i = 0; i < xy_len; i++) {
		*(ret_img.ptr<float>(0, i)) = *(dx1->ptr<float>(y[i], x[i]));
	}
	for (i = 0; i < xy_len; i++) {
		*(ret_img.ptr<float>(1, i)) = *(dy1->ptr<float>(y[i], x[i]));
	}
	for (i = 0; i < xy_len; i++) {
		*(ret_img.ptr<float>(2, i)) = *(dx2->ptr<float>(y[i], x[i]));
	}
	for (i = 0; i < xy_len; i++) {
		*(ret_img.ptr<float>(3, i)) = *(dy2->ptr<float>(y[i], x[i]));
	}

	return ret_img;
}

Mat get_bb(int* y, int* x, int xy_len)
{
	int i = 0;
	float* ptr_float = NULL;
	Mat ret_img = Mat::zeros(2, xy_len, CV_32FC1);

	for (i = 0; i < xy_len; i++) {
		*(ret_img.ptr<float>(0, i)) = y[i];
	}
	for (i = 0; i < xy_len; i++) {
		*(ret_img.ptr<float>(1, i)) = x[i];
	}

	return ret_img;
}

Mat get_q1(Mat* img, int stride, double scale, int xy_len)
{
	int i = 0, j = 0;
	float* ptr_float = NULL;
	double* ptr_double = NULL;

	Mat ret_img = Mat::zeros(xy_len, 2, CV_64FC1);

	for (i = 0; i < ret_img.rows; i++) {
		ptr_float = img->ptr<float>(i);
		ptr_double = ret_img.ptr<double>(i);
		for (j = 0; j < ret_img.cols; j++) {
			*ptr_double = floor((stride * (*ptr_float)+1) / scale);
			ptr_float++;
			ptr_double++;
		}
	}

	return ret_img;
}

Mat get_q2(Mat* img, int stride, double scale, int cellsize, int xy_len)
{
	int i = 0, j = 0;
	float* ptr_float = NULL;
	double* ptr_double = NULL;

	Mat ret_img = Mat::zeros(xy_len, 2, CV_64FC1);

	for (i = 0; i < ret_img.rows; i++) {
		ptr_float = img->ptr<float>(i);
		ptr_double = ret_img.ptr<double>(i);
		for (j = 0; j < ret_img.cols; j++) {
			*ptr_double = floor((stride * (*ptr_float) + cellsize - 1 + 1) / scale);
			ptr_float++;
			ptr_double++;
		}
	}

	return ret_img;
}

Mat get_dims_0_to_1(float* score, int xy_len)
{
	int i = 0;
	float* ptr_float = NULL;
	Mat ret_img = Mat::zeros(xy_len, 1, CV_32FC1);

	ptr_float = ret_img.ptr<float>(0);
	for (i = 0; i < ret_img.rows; i++) {
		*ptr_float = score[i];
		ptr_float++;
	}

	return ret_img;
}

Mat get_hstack(Mat* q1, Mat* q2, Mat* score_mat, Mat* vstack, int xy_len)
{
	int i = 0, j = 0;
	double* ptr_double = NULL;

	Mat ret_img = Mat::zeros(xy_len, 9, CV_64FC1);

	for (i = 0; i < ret_img.rows; i++) {
		ptr_double = ret_img.ptr<double>(i);
		for (j = 0; j < ret_img.cols; j++) {
			if (j < q1->cols) {
				*ptr_double = *(q1->ptr<double>(i, j));
			}
			if (q1->cols <= j && j < (q2->cols + q1->cols)) {
				*ptr_double = *(q2->ptr<double>(i, j - q1->cols));
			}
			if ((q1->cols + q2->cols) <= j && j < (score_mat->cols + q2->cols + q1->cols)) {
				*ptr_double = *(score_mat->ptr<float>(i, j - q2->cols - q1->cols));
			}
			if ( (q1->cols + q2->cols + score_mat->cols) <= j){
				*ptr_double = *(vstack->ptr<float>(i, j - q2->cols - q1->cols - score_mat->cols));
			}
			ptr_double++;
		}
	}

	return ret_img;
}

Mat generateBoundingBox(Mat* imap, Mat* reg, double scale, float t)
{
	int stride = 2;
	int cellsize = 12;

	transpose(*imap, *imap);
	Mat dx1 = get_float_2D(reg, 0);
	transpose(dx1, dx1);
	Mat dy1 = get_float_2D(reg, 1);
	transpose(dy1, dy1);
	Mat dx2 = get_float_2D(reg, 2);
	transpose(dx2, dx2);
	Mat dy2 = get_float_2D(reg, 3);
	transpose(dy2, dy2);

	int* x = NULL;
	int* y = NULL;
	int xy_len = 0;
	get_xy(imap, &x, &y, t, &xy_len);

	if (xy_len == 1) {
		flip(dx1, dx1, 0);
		flip(dy1, dy1, 0);
		flip(dx2, dx2, 0);
		flip(dy2, dy2, 0);
	}

	float* score = get_score(imap, y, x, xy_len);
	Mat vstack = get_vstack(&dx1, &dy1, &dx2, &dy2, y, x, xy_len);
	transpose(vstack, vstack);
	if (vstack.rows * vstack.cols == 0) {
		resize(vstack, vstack, Size(1, 3), 0, 0, INTER_AREA);
	}

	Mat bb = get_bb(y, x, xy_len);
	transpose(bb, bb);

	Mat q1 = get_q1(&bb, stride, scale, xy_len);
	Mat q2 = get_q2(&bb, stride, scale, cellsize ,xy_len);

	Mat score_mat = get_dims_0_to_1(score, xy_len);

	Mat img_ret = get_hstack(&q1, &q2, &score_mat, &vstack, xy_len);
	free(score);
	free(y);
	free(x);

	return img_ret;
}
