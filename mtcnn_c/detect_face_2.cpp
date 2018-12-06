#include <stdio.h>
#include <math.h>
#include "mtcnn.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Mat opencv3_transpose_201(Mat* img)
{
	int i = 0, j = 0, k = 0;
	int x = 0, y = 0, z = 0;
	int xe = 0, ye = 0;
	unsigned char* ptr = NULL;
	Mat img_ret = Mat::zeros(img->channels(), img->rows, CV_8UC(img->cols));

	xe = img->rows;
	ye = img->cols;

    for (i = 0; i < img_ret.rows; i++) {
        for (j = 0; j < img_ret.cols; j++) {
            ptr = img_ret.ptr<uchar>(i, j);
            for (k = 0; k < img_ret.channels(); k++) {
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

	return img_ret;
}

Mat opencv3_transpose_021(Mat* img)
{
	int i = 0, j = 0, k = 0;
	int x = 0, y = 0, z = 0;
	int ye = 0, ze = 0;
	double* ptr = NULL;
	Mat img_ret = Mat::zeros(img->rows, img->channels(), CV_64FC(img->cols));

	ye = img->cols;
	ze = img->channels();

    for (i = 0; i < img_ret.rows; i++) {
        for (j = 0; j < img_ret.cols; j++) {
            ptr = img_ret.ptr<double>(i, j);
            for (k = 0; k < img_ret.channels(); k++) {
                *ptr = *((double*)(img->data + img->step[0]*x + img->step[1]*y) + z);
				ptr++;
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

	return img_ret;
}

Mat opencv3_transpose_0132(Mat* img, int len)
{
	int i = 0, j = 0, k = 0, l = 0;
	int x = 0, y = 0, z = 0, w = 0;
	int ye = 0, ze = 0;
	float* ptr = NULL;

	int size[2] = {0};
	size[0] = img->size().height;
	size[1] = img->size().width;
	size[2] = img->channels();
	Mat img_ret = Mat(3, size, CV_32FC(len), Scalar::all(0));

	ye = len;
	ze = img->channels();
	for (i = 0; i < img_ret.size().height; i++) {
		for (j = 0; j < img_ret.size().width; j++) {
			for (k = 0; k < len; k++) {
				ptr = (float*)(img_ret.data + img_ret.step[0]*i + img_ret.step[1]*j + img_ret.step[2]*k);
				for (l = 0; l < img_ret.channels(); l++) {
					*ptr = *((float*)(img->data + img->step[0]*w + img->step[1]*x + img->step[2]*y) + z);
					ptr++;
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
	}
	return img_ret;
}

Mat imresample(Mat* img, int hs, int ws)
{
	Mat res_img;
	resize(*img, res_img, Size(ws, hs), 0, 0, INTER_AREA);
	return opencv3_transpose_201(&res_img);
}

void image_normalization(Mat* img, Mat* img_double)
{
    int i = 0, j = 0, k = 0;
    unsigned char* ptr = NULL;
	double* ptr_double = NULL;

    for (i = 0; i < img->rows; i++) {
        for (j = 0; j < img->cols; j++) {
            ptr = img->ptr<uchar>(i, j);
			ptr_double = img_double->ptr<double>(i ,j);
            for (k = 0; k < img->channels(); k++) {
                    *ptr_double = (*ptr-127.5) * 0.0078125;
					ptr++;
					ptr_double++;
            }
        }
    }

    return ;
}

void expand_dims(Mat* src, Mat*dst)
{
	int i = 0, j = 0, k = 0, v = 0;
	double* pSrc = NULL;
    double* pDst = NULL;

	for (i = 0; i < 1; i++) {
        for (j = 0; j < src->rows; j++) {
            for (k = 0; k < src->cols; k++) {
                pSrc = src->ptr<double>(j, k);
                pDst = (double*)(dst->data + dst->step[0]*i + dst->step[1]*j + dst->step[2]*k);
				for (v = 0; v < src->channels(); v++) {
					*pDst = *pSrc;
                    pSrc++;
                    pDst++;
                }
            }
        }
    }

	return ;
}

void expand_dims_dump(Mat* src, Mat*dst)
{
	int i = 0, j = 0, k = 0, v = 0;
	float* pSrc = NULL;
    float* pDst = NULL;

	for (i = 0; i < 1; i++) {
        for (j = 0; j < dst->rows; j++) {
            for (k = 0; k < dst->cols; k++) {
                pDst = dst->ptr<float>(j, k);
                pSrc = (float*)(src->data + src->step[0]*i + src->step[1]*j + src->step[2]*k);
				for (v = 0; v < dst->channels(); v++) {
					*pDst = *pSrc;
                    pSrc++;
                    pDst++;
                }
            }
        }
    }

	return ;
}

Mat get_pnet_out(int x, int y, int z, int v, const char* out_name)
{
	int i = 0, j = 0, k = 0, l = 0;
	float* ptr = NULL;
	int size[2] = {0};
	size[0] = x;
	size[1] = y;
	size[2] = z;

	Mat img_ret = Mat(3, size, CV_32FC(v), Scalar::all(0));
	FILE* f = fopen(out_name, "rb");

	for (i = 0; i < size[0]; i++) {
		for (j = 0; j < size[1]; j++) {
			for (k = 0; k < size[2]; k++) {
				ptr = (float*)(img_ret.data + img_ret.step[0]*i + img_ret.step[1]*j + img_ret.step[2]*k);
				for(l = 0; l < img_ret.channels(); l++) {
					fread(ptr, 4, 1, f);
					ptr++;
				}
			}
		}
	}

	fclose(f);
	return img_ret;
}

void get_2D(Mat* src, Mat* dst, int count)
{
	int i = 0, j = 0, k = 0;
	float* pSrc = NULL;
	float* pDst = NULL;

	for (i = 0; i < dst->rows; i++) {
		pDst = dst->ptr<float>(i);
		pSrc = src->ptr<float>(count, i);
		for (j = 0; j < dst->cols; j++) {
			*pDst = *pSrc;
			pSrc++;
			pDst++;
		}
	}
}

void where(Mat* img, float t, int* x, int* y)
{
	int i = 0, j = 0;
	float* ptr = NULL;

	for (i = 0; i < img->rows; i++) {
		ptr = img->ptr<float>(i);
		for (j = 0; j < img->cols; j++) {
			if (*ptr >= t) {
				*x = j;
				*y = i;
				x++;
				y++;
			}
			ptr++;
		}
	}

	return ;
}

void get_value(Mat* img, int* y, int* x, float* score, int len){
	int i = 0;
	for (i = 0; i < len; i++) {
		score[i] = *(img->ptr<float>(y[i], x[i]));
	}
}

void vstack_out(Mat* img, Mat* dx1, Mat* dy1, Mat* dx2, Mat* dy2, int*y, int* x, int len)
{
	int i = 0, j = 0;

	for (i = 0; i < len; i++) {
		*(img->ptr<float>(0,i)) = *(dx1->ptr<float>(y[i], x[i]));
	}
	for (i = 0; i < len; i++) {
		*(img->ptr<float>(1,i)) = *(dy1->ptr<float>(y[i], x[i]));
	}
	for (i = 0; i < len; i++) {
		*(img->ptr<float>(2,i)) = *(dx2->ptr<float>(y[i], x[i]));
	}
	for (i = 0; i < len; i++) {
		*(img->ptr<float>(3,i)) = *(dy2->ptr<float>(y[i], x[i]));
	}

}

void vstack_xy(Mat* img, int*y, int*x, int len)
{
	int i = 0;
	float* ptr = NULL;
	for (i = 0; i < len; i++) {
		*(img->ptr<float>(0, i)) = y[i];
	}
	for (i = 0; i < len; i++) {
		*(img->ptr<float>(1, i)) = x[i];
	}
	return ;
}

void get_q1(Mat* q1, Mat* bb, int stride, double scale)
{
	int i = 0, j = 0;
	double* pDst = NULL;
	float* pSrc = NULL;
 	for (i = 0; i < q1->rows; i++) {
		pDst = q1->ptr<double>(i);
		pSrc = bb->ptr<float>(i);
		for (j = 0; j < q1->cols; j++) {
			*pDst = floor((stride * (*pSrc)+1) / scale);
			pDst++;
			pSrc++;
		}
	}
	return ;
}

void get_q2(Mat* q2, Mat* bb, int stride, double scale, int cellsize)
{
	int i = 0, j = 0;
	double* pDst = NULL;
	float* pSrc = NULL;
 	for (i = 0; i < q2->rows; i++) {
		pDst = q2->ptr<double>(i);
		pSrc = bb->ptr<float>(i);
		for (j = 0; j < q2->cols; j++) {
			*pDst = floor((stride * (*pSrc) + cellsize - 1 + 1) / scale);
			pDst++;
			pSrc++;
		}
	}
	return ;
}

void expand_dims_0_to_1(float* src, Mat* dst)
{
	int i = 0;
	float* pDst = NULL;
	pDst = dst->ptr<float>(0);
	for (i = 0; i < dst->rows; i++) {
		*pDst = src[i];
		pDst++;
	}
	return ;
}

void get_hstack(Mat* dst, Mat* q1, Mat* q2, Mat* score2, Mat* reg)
{
	int i = 0, j = 0;
	double* pDst = NULL;
	for (i = 0; i < dst->rows; i++) {
		pDst = dst->ptr<double>(i);
		for (j = 0; j < dst->cols; j++) {
			if (j < q1->cols) {
				*pDst = *(q1->ptr<double>(i, j));
			}
			if (q1->cols <= j && j < (q2->cols+q1->cols)) {
				*pDst = *(q2->ptr<double>(i, j-q1->cols));
			}
			if ((q1->cols+q2->cols) <= j && j < (score2->cols+q2->cols+q1->cols)) {
				*pDst = *(score2->ptr<float>(i, j-q2->cols-q1->cols));
			}
			if ( (q1->cols+q2->cols+score2->cols) <= j){
				*pDst = *(reg->ptr<float>(i, j-q2->cols-q1->cols-score2->cols));
			}
			pDst++;
		}
	}
	return ;
}

Mat generateBoundingBox(Mat imap, Mat reg, double scale, float t, Mat* vstack)
{
	int stride = 2;
	int cellsize = 12;
	int x[4] = {0};
	int y[4] = {0};
	float score[4] = {0};
	Mat img_ret = Mat(4, 9, CV_64FC1, Scalar::all(0));
	Mat bb = Mat::zeros(2, 4, CV_32FC1);
	Mat q1 = Mat::zeros(4, 2, CV_64FC1);
	Mat q2 = Mat::zeros(4, 2, CV_64FC1);
	Mat score2 = Mat::zeros(4, 1, CV_32FC1);
	Mat dx1 = Mat::zeros(reg.cols, reg.channels(), CV_32FC1);
	Mat dx2 = Mat::zeros(reg.cols, reg.channels(), CV_32FC1);
	Mat dy1 = Mat::zeros(reg.cols, reg.channels(), CV_32FC1);
	Mat dy2 = Mat::zeros(reg.cols, reg.channels(), CV_32FC1);

	transpose(imap, imap);

	get_2D(&reg, &dx1, 0);
	transpose(dx1, dx1);

	get_2D(&reg, &dy1, 1);
	transpose(dy1, dy1);

	get_2D(&reg, &dx2, 2);
	transpose(dx2, dx2);

	get_2D(&reg, &dy2, 3);
	transpose(dy2, dy2);

	where(&imap, t, x, y);

	if (sizeof(x)/sizeof(x[0]) == 1) {
		flip(dx1, dx1, 0);
		flip(dy1, dy1, 0);
		flip(dx2, dx2, 0);
		flip(dy2, dy2, 0);
	}
	get_value(&imap, y, x, score, 4);
	vstack_out(vstack, &dx1, &dy1, &dx2, &dy2, y, x, 4);
	transpose(*vstack, *vstack);
	if (vstack->rows * vstack->cols == 0) {
		/* reg = np.empty((0,3)); */
	}

	vstack_xy(&bb, y, x, 4);
	transpose(bb, bb);

	get_q1(&q1, &bb, stride, scale);
	get_q2(&q2, &bb, stride, scale, cellsize);
	expand_dims_0_to_1(score, &score2);

	get_hstack(&img_ret, &q1, &q2, &score2, vstack);

	return img_ret;
}

void get_boxes(Mat* img, double* x1, double* y1, double* x2, double* y2, double* s)
{
	int i = 0;

	for (i = 0; i < img->rows; i++) {
		x1[i] = *(img->ptr<double>(i, 0));
	}
	for (i = 0; i < img->rows; i++) {
		y1[i] = *(img->ptr<double>(i, 1));
	}
	for (i = 0; i < img->rows; i++) {
		x2[i] = *(img->ptr<double>(i, 2));
	}
	for (i = 0; i < img->rows; i++) {
		y2[i] = *(img->ptr<double>(i, 3));
	}
	for (i = 0; i < img->rows; i++) {
		s[i] = *(img->ptr<double>(i, 4));
	}

}

void get_area(double* area, double* x1, double* y1, double* x2, double* y2)
{
	int i = 0;
	for (i = 0; i < 4; i++) {
		area[i] = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] +1);
	}
}

void get_argsort(double* s, int* I)
{
	int i = 0;
	int j = 0;
	int count = 0;

	for (i = 0; i < 4; i++) {
		for(j = 0; j < 4; j++) {
			if (s[i] > s[j]) {
				count++;
			}
		}
		I[i] = count;
		count = 0;
	}
}

void maximum(double* xx1, double* x1, double i, int* idx, int len)
{
	int v = 0;

	for (v = 0; v < len; v++) {
		if (i >=x1[idx[v]]) {
			xx1[v] = i;
		}else {
			xx1[v] = x1[idx[v]];
		}
	}
}

void minimum(double* xx1, double* x1, double i, int* idx, int len)
{
	int v = 0;

	for (v = 0; v < len; v++) {
		if (i <=x1[idx[v]]) {
			xx1[v] = i;
		}else {
			xx1[v] = x1[idx[v]];
		}
	}
}

int* get_idx(int * I, int len)
{
	int i = 0;
	int* tmp = (int*)malloc(sizeof(int)*(len));

	for (i = 0; i < len; i++) {
		tmp[i] = I[i];

	}
	return tmp;
}

void get_wh(double* tmp, double* xx1, double* xx2, double cmp)
{
	int i = 0;
	for (i = 0; i < 3; i++) {
		tmp[i] = xx2[i] - xx1[i] + 1;
		if (cmp >= tmp[i]) {
			tmp[i] = cmp;
		}
	}
}

void get_inter(double* inter, double* w, double* h)
{
	int i = 0;
	for (i = 0; i < 3; i++) {
		inter[i] = w[i] * h[i];
	}
}

void get_o(double* o, double* inter, double* area, int i, int* idx)
{
	int v = 0;
	for (v = 0; v < 3; v++) {
		o[v] = area[idx[v]];
	}

	for (v = 0; v < 3; v++) {
		o[v] = inter[v] / (area[i] + o[v] - inter[v]);
	}
}

short* nms(Mat* img, float threshold, const char* str)
{
	short* ret = (short*)malloc(2*sizeof(short));

	if ((img->rows * img->cols) == 0) {
		/* return np.empty((0,3)) */
	}

	double x1[4] = {0};
	double y1[4] = {0};
	double x2[4] = {0};
	double y2[4] = {0};
	double s[4] = {0};
	double area[4] = {0};
	int I[4] = {0};
	int i = 0;
	short pick[4] = {0};
	int counter = 0;
	int len = 4;
	double xx1[3] = {0};
	double yy1[3] = {0};
	double xx2[3] = {0};
	double yy2[3] = {0};
	double tmp[3] = {0};
	double w[3] = {0};
	double h[3] = {0};
	double inter[3] = {0};
	double o[3] = {0};

	get_boxes(img, x1, y1, x2, y2, s);
	get_area(area, x1, y1, x2, y2);

	get_argsort(s, I);

	while (len > 0) {
		i = I[len - 1];
		pick[counter] = i;
		counter++;

		len = len - 1;
		int* idx = get_idx(I, len);
		maximum(xx1, x1, x1[i], idx, len);
		maximum(yy1, y1, y1[i], idx, len);
		minimum(xx2, x2, x2[i], idx, len);
		minimum(yy2, y2, y2[i], idx, len);

		get_wh(w, xx1, xx2, 0.0);
		get_wh(h, yy1, yy2, 0.0);

		get_inter(inter, w, h);

		get_o(o, inter, area, i, idx);
		printf("%f\n", threshold);
		//get_I(o, I, threshold);
		return 0;
	}

	return ret;
}

int detect_face(Mat* img, float* threshold, double* scales, int scales_len)
{
	int i = 0;
	int h = img->rows;
	int w = img->cols;
	int hs = 0, ws = 0;
	double scale = 0;
	int total_box[9] = {0};
	int points = 0;
	int size [3] = {0};

	for(i = 7; i <= scales_len; i++) {
		scale = scales[i];
		hs = (int)ceil(h * scale);
		ws = (int)ceil(w * scale);

		Mat im_data = Mat::zeros(img->channels(), hs, CV_8UC(ws));
		Mat im_data_double = Mat::zeros(img->channels(), hs, CV_64FC(ws));

		im_data = imresample(img, hs, ws);
		image_normalization(&im_data, &im_data_double);

		Mat img_x = Mat::zeros(im_data_double.rows, im_data_double.channels(), CV_64FC(im_data_double.cols));
		size[0] = 1;
		size[1] = img_x.rows;
		size[2] = img_x.cols;
		Mat img_y = Mat(3, size, CV_64FC(ws), Scalar::all(0));

		img_x = opencv3_transpose_021(&im_data_double);
		expand_dims(&img_x, &img_y);
		/* out = pnet(img_y) */

		int size[2] = {0};
		size[0] = 1;
		size[1] = 4;
		size[2] = 20;
		Mat out0 = Mat(3, size, CV_32FC(20), Scalar::all(0));
		Mat in0 = Mat::zeros(size[1], size[2], CV_32FC(out0.channels()));

		out0 = get_pnet_out(1, 4, 20, 20, "out0.bin");

		size[0] = 1;
		size[1] = 2;
		size[2] = 20;
		Mat out1 = Mat(3, size, CV_32FC(20), Scalar::all(0));
		Mat in1_tmp = Mat::zeros(size[1], size[2], CV_32FC(out1.channels()));
		Mat in1 = Mat::zeros(size[2], out1.channels(), CV_32FC1);

		out1 = get_pnet_out(1, 2, 20, 20, "out1.bin");

		out0 = opencv3_transpose_0132(&out0, size[2]);
		out1 = opencv3_transpose_0132(&out1, size[2]);

		expand_dims_dump(&out0, &in0);
		expand_dims_dump(&out1, &in1_tmp);

		get_2D(&in1_tmp, &in1, 1);


		Mat ret = Mat(4, 9, CV_64FC1, Scalar::all(0));
		Mat reg = Mat::zeros(4, 4, CV_32FC1);
		ret = generateBoundingBox(in1, in0, scale, threshold[0], &reg);
		save_diff_file(&ret);
		short* pack = nms(&ret, 0.5, "Union");

		return 0;
	}
}
