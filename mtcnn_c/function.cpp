#include "mtcnn.h"

using namespace std;
using namespace cv;

Mat image_normalization(Mat* img, unsigned char type)
{
    int i = 0, j = 0, k = 0;
    unsigned char* ptr_src = NULL;
    double* ptr_dst = NULL;
    Mat ret_img = Mat::zeros(img->rows, img->cols, CV_64FC(img->channels()));

    for (i = 0; i < ret_img.rows; i++) {
        for (j = 0; j < ret_img.cols; j++) {
            ptr_src = img->ptr<uchar>(i, j);
            ptr_dst = ret_img.ptr<double>(i ,j);
            for (k = 0; k < ret_img.channels(); k++) {
                *ptr_dst = (*ptr_src-127.5) * 0.0078125;
                ptr_src++;
                ptr_dst++;
            }
        }
    }

    return ret_img;
}

void image_normalization(Mat* img, int len, double type)
{
    int i = 0, j = 0, k = 0, v = 0;
    double* ptr = NULL;

    for (i = 0; i < img->size().height; i++) {
        for (j = 0; j < img->size().width; j++) {
            for (k = 0; k < len; k++) {
				ptr = (double*)(img->data + img->step[0] * i + img->step[1] * j + img->step[2] * k);
				for (v = 0; v < img->channels(); v++) {
                *ptr = (*ptr - 127.5) * 0.0078125;
                ptr++;
				}
            }
        }
    }

    return ;
}

Mat transpose_201(Mat* img, unsigned char type)
{
    int i = 0, j = 0, k = 0;
    unsigned char* ptr = NULL;
    Mat ret_img = Mat::zeros(img->channels(), img->rows, CV_8UC(img->cols));

    for (i = 0; i < ret_img.rows; i++) {
        for (j = 0; j < ret_img.cols; j++) {
            ptr = ret_img.ptr<uchar>(i, j);
            for (k = 0; k < ret_img.channels(); k++) {
				*ptr = *(img->ptr<uchar>(j, k) + i);
                ptr++;
            }
        }
    }

    return image_normalization(&ret_img, type);
}

Mat transpose_201(Mat* img, double type)
{
    int i = 0, j = 0, k = 0;
    double* ptr = NULL;
    Mat ret_img = Mat::zeros(img->channels(), img->rows, CV_64FC(img->cols));

    for (i = 0; i < ret_img.rows; i++) {
        for (j = 0; j < ret_img.cols; j++) {
            ptr = ret_img.ptr<double>(i, j);
            for (k = 0; k < ret_img.channels(); k++) {
				*ptr = *(img->ptr<double>(j, k) + i);
                ptr++;
            }
        }
    }

    return ret_img;
}

Mat transpose_021(Mat* img, float type)
{
    int i = 0, j = 0, k = 0;
    float* ptr = NULL;
    Mat ret_img = Mat::zeros(img->rows, img->channels(), CV_64FC(img->cols));

    for (i = 0; i < ret_img.rows; i++) {
        for (j = 0; j < ret_img.cols; j++) {
            ptr = ret_img.ptr<float>(i, j);
            for (k = 0; k < ret_img.channels(); k++) {
                *ptr = *(img->ptr<float>(i, k) + j);
				ptr++;
            }
        }
    }

    return ret_img;
}

Mat transpose_021(Mat* img, double type)
{
    int i = 0, j = 0, k = 0;
    double* ptr = NULL;
    Mat ret_img = Mat::zeros(img->rows, img->channels(), CV_64FC(img->cols));

    for (i = 0; i < ret_img.rows; i++) {
        for (j = 0; j < ret_img.cols; j++) {
            ptr = ret_img.ptr<double>(i, j);
            for (k = 0; k < ret_img.channels(); k++) {
                *ptr = *(img->ptr<double>(i, k) + j);
				ptr++;
            }
        }
    }

    return ret_img;
}

Mat expand_dims(Mat* img)
{
    int i = 0, j = 0, k = 0, v = 0;
    double* ptr_src = NULL;
    double* ptr_dst = NULL;
    int Mat_init_size[3] = {0};
    Mat_init_size[0] = 1;
    Mat_init_size[1] = img->rows;
    Mat_init_size[2] = img->cols;

    Mat ret_img = Mat(3, Mat_init_size, CV_64FC(img->channels()), Scalar::all(0));

    for (i = 0; i < ret_img.size().height; i++) {
        for (j = 0; j < ret_img.size().width; j++) {
            for (k = 0; k < Mat_init_size[2]; k++) {
                ptr_src = img->ptr<double>(j, k);
                ptr_dst = (double*)(ret_img.data + ret_img.step[0] * i + ret_img.step[1] * j + ret_img.step[2] * k);
                for (v = 0; v < ret_img.channels(); v++) {
                    *ptr_dst = *ptr_src;
                    ptr_src++;
                    ptr_dst++;
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

Mat imresample(Mat* img, int hs, int ws, unsigned char type)
{
    Mat tmp_img;
    resize(*img, tmp_img, Size(ws, hs), 0, 0, INTER_AREA);
    return transpose_201(&tmp_img, type);
}

Mat imresample(Mat* img, int hs, int ws, double type)
{
    Mat tmp_img;
    resize(*img, tmp_img, Size(ws, hs), 0, 0, INTER_AREA);
	return transpose_201(&tmp_img, type);
}

Mat get_img_y(Mat* img)
{
    Mat img_x = transpose_021(img, (double)1);
    return expand_dims(&img_x);
}

Mat get_pnet_out(int* pnet_out_shape, const char* str, int flag)
{
    int i = 0, j = 0, k = 0, l = 0;
    float* ptr_float = NULL;

    FILE* f = fopen(str, "rb");
    Mat ret_img = Mat::zeros(flag == 0 ? pnet_out_shape[1] : (pnet_out_shape[1] - 2), pnet_out_shape[2], CV_32FC(pnet_out_shape[3]));

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

Mat get_rnet_out(int* pnet_out_shape, const char* str, int flag)
{
    int i = 0, j = 0, k = 0, l = 0;
    float* ptr_float = NULL;

    FILE* f = fopen(str, "rb");
    Mat ret_img = Mat::zeros(pnet_out_shape[0], flag == 0 ? pnet_out_shape[1] : (pnet_out_shape[1] - 2), CV_32FC1);

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

Mat get_onet_out(int* pnet_out_shape, const char* str)
{
    int i = 0, j = 0, k = 0, l = 0;
    float* ptr_float = NULL;

    FILE* f = fopen(str, "rb");
    Mat ret_img = Mat::zeros(pnet_out_shape[0], pnet_out_shape[1], CV_32FC1);

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
    return transpose_021(img, (float)1);
}

Mat get_in1(Mat* img)
{
    Mat tmp_img = transpose_021(img, (float)1);
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
    Mat ret_img = Mat::zeros(4, xy_len, CV_32FC1);

	printf("vstack = %d\n", xy_len);
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

Mat get_hstack_pnet(Mat* q1, Mat* q2, Mat* score_mat, Mat* vstack, int xy_len)
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

    Mat img_ret = get_hstack_pnet(&q1, &q2, &score_mat, &vstack, xy_len);
    free(score);
    free(y);
    free(x);

    return img_ret;
}

void get_boxes(Mat* img, double** x1, double** y1, double** x2, double** y2, double** s)
{
    int i = 0, j = 0;
    *x1 = (double*)malloc(img->rows * sizeof(double));
    *y1 = (double*)malloc(img->rows * sizeof(double));
    *x2 = (double*)malloc(img->rows * sizeof(double));
    *y2 = (double*)malloc(img->rows * sizeof(double));
    *s = (double*)malloc(img->rows * sizeof(double));
    double* tmp_ptr = NULL;

    for (i = 0; i < 5; i++) {
        switch (i) {
            case 0:
                tmp_ptr = *x1;
                break;
            case 1:
                tmp_ptr = *y1;
                break;
            case 2:
                tmp_ptr = *x2;
                break;
            case 3:
                tmp_ptr = *y2;
                break;
            case 4:
                tmp_ptr = *s;
                break;
        }
        for (j = 0; j < img->rows; j++) {
            *tmp_ptr = *(img->ptr<double>(j, i));
            tmp_ptr++;
        }
    }

    return ;
}

double* get_area(double* x1, double* y1, double* x2, double* y2, int len)
{
    int i = 0;
    double* ret_ptr = (double*)malloc(len * sizeof(double));

    for (i = 0; i < len; i++) {
        ret_ptr[i] = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] +1);
    }

    return ret_ptr;
}

int* get_I(double* s, int len)
{
    int i = 0, j = 0;
    int* ret_ptr = (int*)malloc(len * sizeof(int));
	double tmp_arr[len] = {0};
	double tmp_var = 0;

	for (i = 0; i < len; i++) {
		tmp_arr[i] = s[i];
	}

	for (i = 0; i < len - 1; i++) {
		for (j = 0; j < len - 1 - i; j++) {
			if (tmp_arr[j] > tmp_arr[j + 1]) {
				tmp_var = tmp_arr[j];
				tmp_arr[j] = tmp_arr[j + 1];
				tmp_arr[j + 1] = tmp_var;
			}
		}
	}

    for (i = 0; i < len; i++) {
        for (j = 0; j < len; j++) {
            if (s[j] == tmp_arr[i])
                ret_ptr[i] = j;
        }
    }

    return ret_ptr;
}

int* get_idx(int* I, int len)
{
    int i = 0;
    int* ret_ptr = (int*)malloc(len * sizeof(int));

    for (i = 0; i < len; i++) {
		ret_ptr[i] = I[i];
    }

    return ret_ptr;
}

double* maximum(double* x1, double x, int* idx, int len)
{
    int i = 0;
    double* ret_ptr = (double*)malloc(len * sizeof(double));

    for (i = 0; i < len; i++) {
        if (x >= x1[idx[i]]) {
            ret_ptr[i] = x;
        }else {
            ret_ptr[i] = x1[idx[i]];
        }
    }

    return ret_ptr;
}

double* minimum(double* x1, double x, int* idx, int len)
{
    int i = 0;
    double* ret_ptr = (double*)malloc(len * sizeof(double));

    for (i = 0; i < len; i++) {
        if (x <= x1[idx[i]]) {
            ret_ptr[i] = x;
        }else {
            ret_ptr[i] = x1[idx[i]];
        }
    }

    return ret_ptr;
}

double* get_wh(double* xx1, double* xx2, double cmp, int len)
{
    int i = 0;
    double* ret_ptr = (double*)malloc(len * sizeof(double));

    for (i = 0; i < len; i++) {
        ret_ptr[i] = xx2[i] - xx1[i] + 1;
        if (cmp >= ret_ptr[i]) {
            ret_ptr[i] = cmp;
        }
    }

    return ret_ptr;
}

double* get_inter(double* w, double* h, int len)
{
    int i = 0;
    double* ret_ptr = (double*)malloc(len * sizeof(double));

    for (i = 0; i < len; i++) {
        ret_ptr[i] = w[i] * h[i];
    }

    return ret_ptr;
}

double* get_o(double* inter, double* area, int x, int* idx, int len)
{
    int i = 0;
    double* ret_ptr = (double*)malloc(len * sizeof(double));

    for (i = 0; i < len; i++) {
        ret_ptr[i] = area[idx[i]];
    }
    for (i = 0; i < len; i++) {
        ret_ptr[i] = inter[i] / (area[x] + ret_ptr[i] - inter[i]);
    }

    return ret_ptr;
}

void updata_I(int** I, double* o, float threshold, int* len)
{
    int i = 0;
    int count = 0;

	if (*len == 0) {
		return ;
	} else {
		int tmp_arr[*len] = {0};

		for (i = 0; i < *len; i++) {
			tmp_arr[i] = (*I)[i];
			if (o[i] <= threshold)
			    count++;
		}

		if (count == 0) {
			*len = count;
			return ;
		}

		free(*I);

		*I = (int*)malloc(count * sizeof(int));
		int* tmp_I = *I;

		for (i = 0; i < *len; i++) {
			if (o[i] <= threshold) {
			    *tmp_I = tmp_arr[i];
				tmp_I++;
			}
		}

		*len = count;
	}

    return ;
}

short* get_pick(short* pick, int counter)
{
	int i = 0;
	short* ret_ptr = (short*)malloc(counter * sizeof(short));

	for (i = 0; i < counter; i++) {
		ret_ptr[i] = pick[i];
	}

	return ret_ptr;
}

short* nms(Mat* img, float threshold, const char* str, int* pick_len)
{
    if (img->cols * img->rows == 0) {
        return NULL;
    }

    int len = img->rows;
    short pick[100] = {0};
    int counter = 0;

    double* x1 = NULL, *x2 = NULL, *y1 = NULL, *y2 = NULL, *s = NULL;
    get_boxes(img, &x1, &y1, &x2, &y2, &s);
    double* area = get_area(x1, y1, x2, y2, len);
	int* I = get_I(s, len);
	while (len > 0) {
        int i = I[len - 1];
        pick[counter] = i;
        counter++;
        len -= 1;
        int* idx = get_idx(I, len);

		double* xx1 = maximum(x1, x1[i], idx, len);
        double* yy1 = maximum(y1, y1[i], idx, len);
        double* xx2 = minimum(x2, x2[i], idx, len);
        double* yy2 = minimum(y2, y2[i], idx, len);

		double* w = get_wh(xx1, xx2, 0.0, len);
        double* h = get_wh(yy1, yy2, 0.0, len);

        double* inter = get_inter(w, h, len);

        double* o = NULL;
		if (!strcmp(str, "Min")) {
			double* tmp = minimum(area, area[i], idx, len);
			o = get_o_Min(inter, tmp, len);
			free(tmp);
		}else {
			o = get_o(inter, area, i, idx, len);
		}
		updata_I(&I, o, threshold, &len);

		free(o);
        free(inter);
        free(h);
        free(w);
        free(yy2);
        free(xx2);
        free(yy1);
        free(xx1);
        free(idx);
    }
	*pick_len = counter;

    free(I);
    free(area);
    free(s);
    free(y2);
    free(y1);
    free(x2);
    free(x1);
    return get_pick(pick, counter);
}

Mat get_boxes_from_pick(Mat* img, short* pick, int pick_len)
{
	int i = 0, j = 0;
	double* ptr_double_src = NULL;
	double* ptr_double_dst = NULL;

	Mat ret_img = Mat::zeros(pick_len, img->cols, CV_64FC1);

	for (i = 0; i < ret_img.rows; i++) {
		ptr_double_src = img->ptr<double>(pick[i]);
		ptr_double_dst = ret_img.ptr<double>(i);
		for (j = 0; j < ret_img.cols; j++) {
			*ptr_double_dst = *ptr_double_src;
			ptr_double_src++;
			ptr_double_dst++;
		}
	}

	return ret_img;
}

Mat get_total_boxes(Mat* total_boxes, Mat* boxes2)
{
	int i = 0, j = 0;
	double* ptr_src = NULL;
	double* ptr_dst = NULL;

	Mat ret_img = Mat::zeros((total_boxes->rows + boxes2->rows), boxes2->cols, CV_64FC1);

	for (i = 0; i < ret_img.rows; i++) {
		if (i < total_boxes->rows) {
			ptr_src = total_boxes->ptr<double>(i);
		} else {
			ptr_src = boxes2->ptr<double>(i - total_boxes->rows);
		}
		ptr_dst = ret_img.ptr<double>(i);
		for (j = 0; j < ret_img.cols; j++) {
			*ptr_dst = *ptr_src;
			ptr_src++;
			ptr_dst++;
		}
	}

	return ret_img;
}

double* get_reg_wh(Mat* img, int x, int y)
{
	int i = 0;
	double* ret_ptr = (double*)malloc(img->rows * sizeof(double));

	for (i = 0; i < img->rows; i++) {
		ret_ptr[i] = *(img->ptr<double>(i, x)) - *(img->ptr<double>(i, y));
	}

	return ret_ptr;
}

double* get_qq(Mat* img, int x, int y, double* reg)
{
	int i = 0;
	double* ret_ptr = (double*)malloc(img->rows * sizeof(double));

	for (i = 0; i < img->rows; i++) {
		ret_ptr[i] = *(img->ptr<double>(i, x)) + *(img->ptr<double>(i, y)) * reg[i];
	}

	return ret_ptr;
}

Mat get_vstack_qq_and_transpose(double* qq1, double* qq2, double* qq3, double* qq4, Mat* total_boxes, int index)
{
	int i = 0, j = 0;
	double* ptr_double = NULL;
	Mat ret_img = Mat::zeros(total_boxes->rows, 5, CV_64FC1);

	for (i = 0; i < ret_img.rows; i++) {
		*(ret_img.ptr<double>(i, 0))= qq1[i];
	}
	for (i = 0; i < ret_img.rows; i++) {
		*(ret_img.ptr<double>(i, 1))= qq2[i];
	}
	for (i = 0; i < ret_img.rows; i++) {
		*(ret_img.ptr<double>(i, 2))= qq3[i];
	}
	for (i = 0; i < ret_img.rows; i++) {
		*(ret_img.ptr<double>(i, 3))= qq4[i];
	}
	for (i = 0; i < ret_img.rows; i++) {
		*(ret_img.ptr<double>(i, 4))= *(total_boxes->ptr<double>(i, index));
	}

	return ret_img;
}

void get_bboxA(Mat* img, double* x, double* y, int index)
{
	int i = 0;

	for (i = 0; i < img->rows; i++) {
		*(img->ptr<double>(i, index)) = *(img->ptr<double>(i, index)) + x[i] * 0.5 - y[i] * 0.5;
	}

	return ;
}

Mat tile(double* l, int y, int x, int len)
{
	int i = 0, j = 0;
	double* ptr_double = NULL;
	Mat ret_img = Mat::zeros(y, len * x, CV_64FC1);

	for (i = 0; i < ret_img.rows; i++) {
			ptr_double = (double*)ret_img.ptr<double>(i);
		for (j = 0; j < ret_img.cols; j++) {
			*ptr_double = l[j];
			ptr_double++;
		}
	}

	transpose(ret_img, ret_img);

	return ret_img;
}

void get_ret_rerec(Mat* img, int xs, int xe, int ys, int ye, Mat* tmp)
{
	int i = 0, j = 0;
	double* ptr_double_src = NULL;
	double* ptr_double_dst = NULL;

	for (i = 0; i < img->rows; i++) {
		ptr_double_src = (double*)img->ptr<double>(i, xs);
		ptr_double_dst = (double*)img->ptr<double>(i, ys);
		for (j = 0; j < (xe-xs); j++) {
			*ptr_double_dst = *ptr_double_src + *(tmp->ptr<double>(i, j));
			ptr_double_src++;
			ptr_double_dst++;
		}
	}

	return ;
}

void rerec(Mat* img)
{
	int i = 0;
	double* h = get_reg_wh(img, 3, 1);
	double* w = get_reg_wh(img, 2, 0);

	double* l = (double*)malloc(img->rows * sizeof(double));
	for (i = 0; i < img->rows; i++) {
		l[i] = (h[i] >= w[i] ? h[i] : w[i]);
	}

	get_bboxA(img, w, l, 0);
	get_bboxA(img, h, l, 1);

	Mat tmp = tile(l, 2, 1, img->rows);

	get_ret_rerec(img, 0, 2, 2, 4, &tmp);

	free(l);
	free(h);
	free(w);

	return ;
}

void get_total_boxeses_fix(Mat* img, int xs, int xe, int ys, int ye)
{
	int i = 0, j = 0;
	double* ptr_double_src = NULL;
	double* ptr_double_dst = NULL;

	for (i = 0; i < img->rows; i++) {
		ptr_double_src = (double*)img->ptr<double>(i, xs);
		ptr_double_dst = (double*)img->ptr<double>(i, ys);
		for (j = 0; j < (xe - xs); j++) {
			*ptr_double_dst = floor(*ptr_double_src);
			ptr_double_src++;
			ptr_double_dst++;
		}
	}

	return ;
}

void get_tmpwh(Mat* img, int* tmp, int x, int y)
{
	int i = 0, j = 0;

	for (i = 0; i < img->rows; i++) {
		tmp[i] = *(img->ptr<double>(i, x)) - *(img->ptr<double>(i, y)) + 1;
	}

	return ;
}

void init_dx_dy_edx_edy(int* dx, int* dy, int* edx, int* edy, int* tmpw, int* tmph, int numbox)
{
	int i = 0;

	for (i = 0; i < numbox; i++) {
		dx[i] = 1;
		dy[i] = 1;
		edx[i] = tmpw[i];
		edy[i] = tmph[i];
	}

	return ;
}

void init_x_y_ex_ey(Mat* img, int* x, int* y, int* ex, int* ey)
{
	int i = 0, j = 0;;

	for (i = 0; i < img->rows; i++) {
		x[i] = *(img->ptr<double>(i, 0));
	}
	for (i = 0; i < img->rows; i++) {
		y[i] = *(img->ptr<double>(i, 1));
	}
	for (i = 0; i < img->rows; i++) {
		ex[i] = *(img->ptr<double>(i, 2));
	}
	for (i = 0; i < img->rows; i++) {
		ey[i] = *(img->ptr<double>(i, 3));
	}

	return ;
}

void set_exy(int* edx, int* ex, int w, int* tmpw, int len)
{
	int i = 0;
	int tmp[len] = {0};

	for (i = 0; i < len; i++) {
		tmp[i] = ex[i] > w ? i : -1;
	}

	for (i = 0; i < len; i++) {
		if (tmp[i] >= 0) {
			edx[i] = 0 - ex[i] + w + tmpw[i];
			ex[i] = w;
		}
	}

	return ;
}

void set_xy(int* dx, int* x, int w, int len)
{
	int i = 0;
	int tmp[len] = {0};

	for (i = 0; i < len; i++) {
		tmp[i] = x[i] < w ? i : -1;
	}

	for (i = 0; i < len; i++) {
		if (tmp[i] >= 0) {
			dx[i] = 2 - x[i];
			x[i] = w;
		}
	}

	return ;
}

void pad(Mat* img, int h, int w, int* dy, int* edy, int* dx, int* edx, int* y, int* ey, int* x, int* ex, int* tmpw, int* tmph)
{
	int i = 0;

	get_tmpwh(img, tmpw, 2, 0);
	get_tmpwh(img, tmph, 3, 1);

	int numbox = img->rows;

	init_dx_dy_edx_edy(dx, dy, edx, edy, tmpw, tmph, numbox);
	init_x_y_ex_ey(img, x, y, ex, ey);

	set_exy(edx, ex, w, tmpw, numbox);
	set_exy(edy, ey, h, tmph, numbox);
	set_xy(dx, x, 1, numbox);
	set_xy(dy, y, 1, numbox);

	return ;
}

void buckle_map(Mat* img, Mat* tmp, int* x, int* ex, int* y, int* ey, int* dx, int* edx, int* dy, int* edy, int k)
{
	int i = 0, j = 0, v = 0;
	uchar* src = NULL;
	double* dst = NULL;

	for (i = 0; i < (ey[k] - y[k] + 1); i++) {
		for (j = 0; j < (ex[k] - x[k] + 1); j++) {
			src = (unsigned char*)img->ptr<uchar>((y[k] - 1 + i), (x[k] - 1 + j));
			dst = (double*)tmp->ptr<double>((dy[k] - 1 + i), (dx[k] - 1 + j));
			for (v = 0; v < img->channels(); v++) {
				*dst = *src;
				src++;
				dst++;
			}
		}
	}

	return ;
}

void get_tempimg(Mat* tempimg, Mat* tmp_tempimg, int k, int len)
{
	int i = 0, j = 0, v = 0, z = 0;
	double* src = NULL;
	double* dst = (double*)tempimg->data + k;

	for (i = 0; i < tempimg->size().height; i++) {
		for (j = 0; j < tempimg->size().width; j++) {
			src = (double*)tmp_tempimg->ptr<double>(i, j);
			for (v = 0; v < len; v++) {
				*dst = *src;
				src++;
				dst += tempimg->channels();
			}
		}
	}

	return ;
}

Mat transpose3021(Mat* img, int len)
{
	int i = 0, j = 0, k = 0, v = 0;
	double* src = NULL;
	double* dst = NULL;

	int Mat_init_size[3] = {0};
	Mat_init_size[0] = img->channels();
	Mat_init_size[1] = img->size().height;
	Mat_init_size[2] = len;
	Mat ret_img = Mat(3, Mat_init_size, CV_64FC(img->size().width));

	for (i = 0; i < ret_img.size().height; i++) {
		for (j = 0; j < ret_img.size().width; j++) {
			for (k = 0; k < Mat_init_size[2]; k++) {
				dst = (double*)(ret_img.data + ret_img.step[0] * i + ret_img.step[1] * j + ret_img.step[2] * k);
				for (v = 0; v < ret_img.channels(); v++) {
					*dst = *(((double*)(img->data + img->step[0] * j + img->step[1] * v + img->step[2] * k)) + i);
					dst++;
				}
			}
		}
	}

	return ret_img;
}

float* get_score_out(Mat* img, int index, int len)
{
	int i = 0;
	float* ret_ptr = (float*)malloc(len * sizeof(float));

	for (i = 0; i < len; i++) {
		ret_ptr[i] = *(img->ptr<float>(index, i));
	}

	return ret_ptr;
}

int* get_ipass(float* score, float threshold, int len, int* ipass_len)
{
	int i = 0;
	int count = 0;

	for (i = 0; i < len; i++) {
		if (score[i] > threshold)
			count++;
	}

	*ipass_len = count;
	int* ret_ptr = (int*)malloc(count * sizeof(int));
	int* tmp_ptr = ret_ptr;

	for (i = 0; i < len; i++) {
		if (score[i] > threshold) {
			*tmp_ptr = i;
			tmp_ptr++;
		}
	}

	return ret_ptr;
}

Mat get_hstack_rnet(Mat* img, int* ipass, int x, int y, float* score, int ipass_len)
{
	int i = 0, j = 0;

	Mat ret_img = Mat::zeros(ipass_len, img->cols, CV_64FC1);

	for(i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			if (x <= j && j < y) {
				*(ret_img.ptr<double>(i, j)) = *(img->ptr<double>(ipass[i], j));
			} else {
				*(ret_img.ptr<double>(i, j)) = score[ipass[i]];
			}
		}
	}

	return ret_img;
}

Mat get_mv(Mat* img, int* ipass, int ipass_len)
{
	int i = 0, j = 0;

	Mat ret_img = Mat::zeros(img->rows, ipass_len, CV_32FC1);

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			*(ret_img.ptr<float>(i, j)) = *(img->ptr<float>(i, ipass[j]));
		}
	}

	return ret_img;
}

Mat get_total_boxeses_pick(Mat* img, short* pick, int len)
{
	int i = 0, j = 0;
	Mat ret_img = Mat::zeros(len, img->cols, CV_64FC1);

	for (i = 0; i < len; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			*(ret_img.ptr<double>(i, j)) = *(img->ptr<double>(pick[i], j));
		}
	}

	return ret_img;
}

Mat transpose_mv_piack(Mat* img, short* pick, int len)
{
	int i = 0, j = 0;
	Mat ret_img = Mat::zeros(img->rows, len, CV_32FC1);

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < len; j++) {
			*(ret_img.ptr<float>(i, j)) = *(img->ptr<float>(i, pick[j]));
		}
	}

	transpose(ret_img, ret_img);
	return ret_img;
}

void get_wh_bbreg(Mat* img, double* w, double* h, int len)
{
	int i = 0;
	for (i = 0; i < len; i++) {
		w[i] = *(img->ptr<double>(i, 2)) - *(img->ptr<double>(i, 0)) + 1;
		h[i] = *(img->ptr<double>(i, 3)) - *(img->ptr<double>(i, 1)) + 1;
	}

	return ;
}

void get_b_bbreg(Mat* img, Mat* mv, double* b1, double* b2, double* b3, double* b4, double* w, double* h, int len)
{
	int i = 0;
	for (i = 0; i < len; i++) {
		b1[i] = *(img->ptr<double>(i, 0)) + *(mv->ptr<float>(i, 0)) * w[i];
		b2[i] = *(img->ptr<double>(i, 1)) + *(mv->ptr<float>(i, 1)) * h[i];
		b3[i] = *(img->ptr<double>(i, 2)) + *(mv->ptr<float>(i, 2)) * w[i];
		b4[i] = *(img->ptr<double>(i, 3)) + *(mv->ptr<float>(i, 3)) * h[i];
	}

	return ;
}

void get_bbreg_return(Mat* img, double* b1, double* b2, double* b3, double* b4, int len)
{
	int i = 0, j = 0;
	Mat tmp = Mat::zeros(4, len, CV_64FC1);
	for (i = 0; i < tmp.cols; i++) {
		*(tmp.ptr<double>(0, i)) = b1[i];
		*(tmp.ptr<double>(1, i)) = b2[i];
		*(tmp.ptr<double>(2, i)) = b3[i];
		*(tmp.ptr<double>(3, i)) = b4[i];
	}

	transpose(tmp, tmp);

	for (i = 0; i < tmp.rows; i++) {
		for (j = 0; j < 4; j++) {
			*(img->ptr<double>(i, j)) = *(tmp.ptr<double>(i, j));
		}
	}

	return ;
}

void bbreg(Mat* img, Mat* mv)
{
	if (mv->cols == 1) {
		/* reg = np.reshape(reg, (reg.shape[2], reg.shape[3])) */
	}

	int len = mv->rows;

	double* w = (double*)malloc(len * sizeof(double));
	double* h = (double*)malloc(len * sizeof(double));
	double* b1 = (double*)malloc(len * sizeof(double));
	double* b2 = (double*)malloc(len * sizeof(double));
	double* b3 = (double*)malloc(len * sizeof(double));
	double* b4 = (double*)malloc(len * sizeof(double));
	get_wh_bbreg(img, w, h, len);
	get_b_bbreg(img, mv, b1, b2, b3, b4, w, h, len);

	get_bbreg_return(img, b1, b2, b3, b4, len);

	free(w);
	free(h);
	free(b1);
	free(b2);
	free(b3);
	free(b4);
	return ;
}

Mat fix_total_boxeses(Mat* img)
{
	int i = 0, j = 0;
	//Mat ret_img = Mat::zeros(img->rows, img->cols, CV_32SC1);
	Mat ret_img = Mat::zeros(img->rows, img->cols, CV_64FC1);
	for (i = 0; i < img->rows; i++) {
		for (j = 0; j < img->cols; j++) {
			//*(ret_img.ptr<int>(i, j)) = floor(*(img->ptr<double>(i, j)));
			*(ret_img.ptr<double>(i, j)) = floor(*(img->ptr<double>(i, j)));
		}
	}

	return ret_img;
}

Mat get_points(Mat* img, int* ipass, int len)
{
	int i = 0, j = 0;
	Mat ret_img = Mat::zeros(img->rows, len, CV_32FC1);

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			*(ret_img.ptr<float>(i, j)) = *(img->ptr<float>(i, ipass[j]));
		}
	}

	return ret_img;
}

void updata_points(Mat* img, Mat* total_boxes, double* w, double* h, int len)
{
	int i = 0, j = 0;

	for (i = 0; i < 5; i++) {
		for (j = 0; j < len; j++) {
			*(img->ptr<float>(i, j)) = (w[j] * *(img->ptr<float>(i, j))) + (*(total_boxes->ptr<double>(j, 0)) - 1);
			*(img->ptr<float>(i + 5, j)) = (h[j] * *(img->ptr<float>(i + 5, j))) + (*(total_boxes->ptr<double>(j, 1)) - 1);
		}
	}

	return ;
}

double* get_o_Min(double* inter, double* tmp, int len)
{
	int i = 0;
	double* ret_ptr = (double*)malloc(len * sizeof(double));

	for (i = 0; i < len; i++) {
		ret_ptr[i] = inter[i] / tmp[i];
	}

	return ret_ptr;
}

Mat  points_pick(Mat* img, short* pick, int pick_len)
{
	int i = 0, j = 0;
	Mat ret_img = Mat::zeros(img->rows, pick_len, CV_32FC1);
	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			*(ret_img.ptr<float>(i, j))= *(img->ptr<float>(i, pick[j]));
		}
	}

	return ret_img;
}

Mat face_preprocess(Mat* img, Mat* landmark)
{
	int i = 0, j = 0;
	float* ptr = NULL;
	Mat ret_img;

#if 0
	float src_arr[5][2] = {{38.2946, 51.6963}, {73.5318, 51.5014}, {56.0252, 71.7366}, {41.5493, 92.3655}, {70.7299, 92.2041}};
	Mat src = Mat::zeros(5, 2, CV_32FC1);

	for (i = 0; i < 5; i++) {
		ptr = src.ptr<float>(i);
		for (j = 0; j < 2; j++) {
			*ptr = src_arr[i][j];
			ptr++;
		}
	}

	Mat M = estimateRigidTransform(*landmark, src, false);
	warpAffine(*img, ret_img, M, Size(112, 112), 1, 0);
#endif

#if 1
	vector<Point2d> source_pts(5);

	for (i = 0; i < 5; i++) {
		ptr = landmark->ptr<float>(i);
		source_pts[i] = Point2d(*ptr, *(ptr + 1));
	}

	int ret = Align(*img, ret_img, source_pts);
#endif

	return ret_img;
}
