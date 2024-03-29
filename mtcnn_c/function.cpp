#include "mtcnn.h"

using namespace std;
using namespace cv;

Mat image_normalization(Mat* img, unsigned char type)
{
    int i = 0, j = 0, k = 0;
    unsigned char* ptr_src = NULL;
    float* ptr_dst = NULL;
    Mat ret_img = Mat::zeros(img->rows, img->cols, CV_32FC(img->channels()));

    for (i = 0; i < ret_img.rows; i++) {
        for (j = 0; j < ret_img.cols; j++) {
            ptr_src = img->ptr<uchar>(i, j);
            ptr_dst = ret_img.ptr<float>(i, j);
            for (k = 0; k < ret_img.channels(); k++) {
                *ptr_dst = (*ptr_src-127.5) * 0.0078125;
                ptr_src++;
                ptr_dst++;
            }
        }
    }

    return ret_img;
}

float* image_normalization(Mat* img, int len, float type)
{
    int i = 0, j = 0, k = 0, v = 0;
    float* ptr = NULL;
	float* ret_ptr = (float*)malloc(img->size().height * img->size().width * len * img->channels() * sizeof(float));
	if (ret_ptr == NULL ) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

	float* tmp_ptr = ret_ptr;

    for (i = 0; i < img->size().height; i++) {
        for (j = 0; j < img->size().width; j++) {
            for (k = 0; k < len; k++) {
				ptr = (float*)(img->data + img->step[0] * i + img->step[1] * j + img->step[2] * k);
				for (v = 0; v < img->channels(); v++) {
                *tmp_ptr = (*ptr - 127.5) * 0.0078125;
                ptr++;
				tmp_ptr++;
				}
            }
        }
    }

    return ret_ptr;
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

Mat transpose_201(Mat* img, float type)
{
    int i = 0, j = 0, k = 0;
    float* ptr = NULL;
    Mat ret_img = Mat::zeros(img->channels(), img->rows, CV_32FC(img->cols));

    for (i = 0; i < ret_img.rows; i++) {
        for (j = 0; j < ret_img.cols; j++) {
            ptr = ret_img.ptr<float>(i, j);
            for (k = 0; k < ret_img.channels(); k++) {
				*ptr = *(img->ptr<float>(j, k) + i);
                ptr++;
            }
        }
    }

    return ret_img;
}

Mat transpose_021(Mat* img)
{
    int i = 0, j = 0, k = 0;
    float* ptr = NULL;
    Mat ret_img = Mat::zeros(img->rows, img->channels(), CV_32FC(img->cols));

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

Mat transpose3021(Mat* img, int len)
{
	int i = 0, j = 0, k = 0, v = 0;
	float* ptr_dst = NULL;

	int Mat_init_size[3] = {0};
	Mat_init_size[0] = img->channels();
	Mat_init_size[1] = img->size().height;
	Mat_init_size[2] = len;
	Mat ret_img = Mat(3, Mat_init_size, CV_32FC(img->size().width));

	for (i = 0; i < ret_img.size().height; i++) {
		for (j = 0; j < ret_img.size().width; j++) {
			for (k = 0; k < Mat_init_size[2]; k++) {
				ptr_dst = (float*)(ret_img.data + ret_img.step[0] * i + ret_img.step[1] * j + ret_img.step[2] * k);
				for (v = 0; v < ret_img.channels(); v++) {
					*ptr_dst = *(((float*)(img->data + img->step[0] * j + img->step[1] * v + img->step[2] * k)) + i);
					ptr_dst++;
				}
			}
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

Mat imresample(Mat* img, int hs, int ws, float type)
{
    Mat tmp_img;
    resize(*img, tmp_img, Size(ws, hs), 0, 0, INTER_AREA);
	return transpose_201(&tmp_img, type);
}

float* get_img_y(Mat* img)
{
	int i = 0, j = 0, k = 0;
	float* ptr = NULL;
    Mat img_x = transpose_021(img);

	float* ret_ptr = (float*)malloc(img_x.rows * img_x.cols * img_x.channels() * sizeof(float));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

	float* tmp_ptr = ret_ptr;

	for (i = 0; i < img_x.rows; i++) {
		for (j = 0; j < img_x.cols; j++) {
			ptr = img_x.ptr<float>(i, j);
			for (k = 0; k < img_x.channels(); k++) {
				*tmp_ptr = *ptr;
				tmp_ptr++;
				ptr++;
			}
		}
	}

	return ret_ptr;
}

Mat get_in0(Mat* img)
{
    return transpose_021(img);
}

Mat get_in1(Mat* img)
{
    Mat tmp_img = transpose_021(img);

	return from_3DMat_select_rows(&tmp_img, 1);
}

int get_xy(Mat* img, int** x, int** y, float t, int* xy_len)
{
    int i = 0, j = 0, k = 0;
    int count = 0;
    float* ptr = NULL;

    for (i = 0; i < img->rows; i++) {
        ptr = img->ptr<float>(i);
        for(j = 0; j < img->cols; j++) {
            if (*ptr >= t)
                count++;
            ptr++;
        }
    }

	if (count == 0)
		return -1;

    *xy_len = count;

    *x = (int*)malloc(count * sizeof(int));
	if (*x == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return -1;
	}
    *y = (int*)malloc(count * sizeof(int));
	if (*y == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		free(*x);
		return -1;
	}

    for (i = 0; i < img->rows; i++) {
        ptr = img->ptr<float>(i);
        for(j = 0; j < img->cols; j++) {
            if (*ptr >= t) {
                *(*x + k) = j;
                *(*y + k) = i;
				k++;
			}
            ptr++;
		}
    }

    return 0;
}

Mat get_score_in_gBB(Mat* img, int* y, int *x, int xy_len)
{
    int i = 0;
    float* ptr = 0;
	Mat ret_img = Mat::zeros(xy_len, 1, CV_32FC1);

	ptr = ret_img.ptr<float>(0);
    for (i = 0; i < ret_img.rows; i++) {
        *ptr = *(img->ptr<float>(y[i], x[i]));
		ptr++;
	}

    return ret_img;
}

Mat get_vstack_in_gBB(Mat* dx1, Mat* dy1, Mat* dx2, Mat* dy2, int* y, int* x, int xy_len)
{
    int i = 0;
    Mat ret_img = Mat::zeros(4, xy_len, CV_32FC1);

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

Mat get_bb_in_gBB(int* y, int* x, int xy_len)
{
    int i = 0;
    Mat ret_img = Mat::zeros(2, xy_len, CV_32SC1);

    for (i = 0; i < ret_img.cols; i++) {
        *(ret_img.ptr<int>(0, i)) = y[i];
    }
    for (i = 0; i < ret_img.cols; i++) {
        *(ret_img.ptr<int>(1, i)) = x[i];
    }

    return ret_img;
}

Mat get_q1(Mat* img, int stride, float scale, int xy_len)
{
    int i = 0, j = 0;
	int tmp_var = 0;
    int* ptr_src = NULL;
    float* ptr_dst = NULL;

    Mat ret_img = Mat::zeros(xy_len, 2, CV_32FC1);

    for (i = 0; i < ret_img.rows; i++) {
        ptr_src = img->ptr<int>(i);
        ptr_dst = ret_img.ptr<float>(i);
        for (j = 0; j < ret_img.cols; j++) {
            tmp_var = (stride * (*ptr_src) + 1) / scale;
			*ptr_dst = tmp_var > 0 ? floor(tmp_var) : ceil(tmp_var);
            ptr_src++;
            ptr_dst++;
        }
    }

    return ret_img;
}

Mat get_q2(Mat* img, int stride, float scale, int cellsize, int xy_len)
{
    int i = 0, j = 0;
	int tmp_var = 0;
    int* ptr_src = NULL;
    float* ptr_dst = NULL;

    Mat ret_img = Mat::zeros(xy_len, 2, CV_32FC1);

    for (i = 0; i < ret_img.rows; i++) {
        ptr_src = img->ptr<int>(i);
        ptr_dst = ret_img.ptr<float>(i);
        for (j = 0; j < ret_img.cols; j++) {
            tmp_var = (stride * (*ptr_src) + cellsize - 1 + 1) / scale;
			*ptr_dst = tmp_var > 0 ? floor(tmp_var) : ceil(tmp_var);
            ptr_src++;
            ptr_dst++;
        }
    }

    return ret_img;
}

Mat get_hstack_pnet(Mat* q1, Mat* q2, Mat* score, Mat* vstack, int xy_len)
{
    int i = 0, j = 0;
    float* ptr = NULL;

    Mat ret_img = Mat::zeros(xy_len, 9, CV_32FC1);

    for (i = 0; i < ret_img.rows; i++) {
        ptr = ret_img.ptr<float>(i);
        for (j = 0; j < ret_img.cols; j++) {
            if (j < q1->cols) {
                *ptr = *(q1->ptr<float>(i, j));
            }
            if (q1->cols <= j && j < (q2->cols + q1->cols)) {
                *ptr = *(q2->ptr<float>(i, j - q1->cols));
            }
            if ((q1->cols + q2->cols) <= j && j < (score->cols + q2->cols + q1->cols)) {
                *ptr = *(score->ptr<float>(i, j - q2->cols - q1->cols));
            }
            if ( (q1->cols + q2->cols + score->cols) <= j){
                *ptr = *(vstack->ptr<float>(i, j - q2->cols - q1->cols - score->cols));
            }
            ptr++;
        }
    }

    return ret_img;
}

Mat from_3DMat_select_rows(Mat* img, int index)
{
    int i = 0, j = 0;
    float* ptr_src = NULL;
    float* ptr_dst = NULL;
    Mat ret_img = Mat::zeros(img->cols, img->channels(), CV_32FC1);

    for (i = 0; i < ret_img.rows; i++) {
        ptr_src = img->ptr<float>(index, i);
        ptr_dst = ret_img.ptr<float>(i);
        for (j = 0; j < ret_img.cols; j++) {
            *ptr_dst = *ptr_src;
            ptr_src++;
            ptr_dst++;
        }
    }

    return ret_img;
}

Mat generateBoundingBox(Mat* imap, Mat* reg, float scale, float t)
{
    int stride = 2;
    int cellsize = 12;

    transpose(*imap, *imap);

	Mat dx1 = from_3DMat_select_rows(reg, 0);
    transpose(dx1, dx1);
    Mat dy1 = from_3DMat_select_rows(reg, 1);
    transpose(dy1, dy1);
    Mat dx2 = from_3DMat_select_rows(reg, 2);
    transpose(dx2, dx2);
    Mat dy2 = from_3DMat_select_rows(reg, 3);
    transpose(dy2, dy2);

    int* x = NULL;
    int* y = NULL;
    int xy_len = 0;
    int ret = get_xy(imap, &x, &y, t, &xy_len);

	if (ret != 0) {
		Mat ret_img = Mat::zeros(0, 0, CV_8UC1);
		return ret_img;
	}

    if (xy_len == 1) {
        flip(dx1, dx1, 0);
        flip(dy1, dy1, 0);
        flip(dx2, dx2, 0);
        flip(dy2, dy2, 0);
    }

    Mat score = get_score_in_gBB(imap, y, x, xy_len);
    Mat vstack = get_vstack_in_gBB(&dx1, &dy1, &dx2, &dy2, y, x, xy_len);
    transpose(vstack, vstack);

	if (vstack.rows * vstack.cols == 0) {
		*reg = Mat::zeros(1, 3, CV_32FC1);
    }

    Mat bb = get_bb_in_gBB(y, x, xy_len);
    transpose(bb, bb);

    Mat q1 = get_q1(&bb, stride, scale, xy_len);
    Mat q2 = get_q2(&bb, stride, scale, cellsize ,xy_len);

    Mat ret_img = get_hstack_pnet(&q1, &q2, &score, &vstack, xy_len);

    free(y);
    free(x);

    return ret_img;
}

float* get_area(Mat* x1, Mat* y1, Mat* x2, Mat* y2, int len)
{
    int i = 0;
    float* ret_ptr = (float*)malloc(len * sizeof(float));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

    for (i = 0; i < len; i++) {
        ret_ptr[i] = (*(x2->ptr<float>(i)) - *(x1->ptr<float>(i)) + 1) * (*(y2->ptr<float>(i)) - *(y1->ptr<float>(i)) + 1);
    }

    return ret_ptr;
}

int* get_idx(Mat* I)
{
    int i = 0;
    int* ret_ptr = (int*)malloc((I->rows - 1) * sizeof(int));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

    for (i = 0; i < (I->rows - 1); i++) {
		ret_ptr[i] = *(I->ptr<int>(i));
    }

    return ret_ptr;
}

float* maximum(Mat* x1, int cmp_idx, int* idx, int len)
{
    int i = 0;
    float* ret_ptr = (float*)malloc(len * sizeof(float));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

    for (i = 0; i < len; i++) {
        if (*(x1->ptr<float>(cmp_idx)) >= *(x1->ptr<float>(idx[i]))) {
            ret_ptr[i] = *(x1->ptr<float>(cmp_idx));
        }else {
            ret_ptr[i] = *(x1->ptr<float>(idx[i]));
        }
    }

    return ret_ptr;
}

float* minimum(Mat* x1, int cmp_idx, int* idx, int len)
{
    int i = 0;
    float* ret_ptr = (float*)malloc(len * sizeof(float));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

    for (i = 0; i < len; i++) {
        if (*(x1->ptr<float>(cmp_idx)) <= *(x1->ptr<float>(idx[i]))) {
            ret_ptr[i] = *(x1->ptr<float>(cmp_idx));
        }else {
            ret_ptr[i] = *(x1->ptr<float>(idx[i]));
        }
    }

    return ret_ptr;
}

float* minimum(float* x1, float cmp, int* idx, int len)
{
    int i = 0;
    float* ret_ptr = (float*)malloc(len * sizeof(float));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

    for (i = 0; i < len; i++) {
        if (cmp <= x1[idx[i]]) {
            ret_ptr[i] = cmp;
        }else {
            ret_ptr[i] = x1[idx[i]];
        }
    }

    return ret_ptr;
}

float* get_wh_in_nms(float* xx1, float* xx2, float cmp, int len)
{
    int i = 0;
    float* ret_ptr = (float*)malloc(len * sizeof(float));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

    for (i = 0; i < len; i++) {
        ret_ptr[i] = xx2[i] - xx1[i] + 1;
        if (cmp >= ret_ptr[i]) {
            ret_ptr[i] = cmp;
        }
    }

    return ret_ptr;
}

float* get_inter(float* w, float* h, int len)
{
    int i = 0;
    float* ret_ptr = (float*)malloc(len * sizeof(float));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

    for (i = 0; i < len; i++) {
        ret_ptr[i] = w[i] * h[i];
    }

    return ret_ptr;
}

float* get_o(float* inter, float* area, int x, int* idx, int len)
{
    int i = 0;
    float* ret_ptr = (float*)malloc(len * sizeof(float));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

    for (i = 0; i < len; i++) {
        ret_ptr[i] = area[idx[i]];
    }
    for (i = 0; i < len; i++) {
        ret_ptr[i] = inter[i] / (area[x] + ret_ptr[i] - inter[i]);
    }

    return ret_ptr;
}

Mat update_I(Mat* I, float* o, float threshold)
{
    int i = 0, j = 0;
    int count = 0;

	for (i = 0; i < I->rows - 1; i++) {
		if (o[i] <= threshold)
			count++;
	}

	if (count == 0) {
		Mat ret_img = Mat::zeros(count, 0, CV_32SC1);
		return ret_img;
	}

	Mat ret_img = Mat::zeros(count, 1, CV_32SC1);

	for (i = 0; i < I->rows - 1; i++) {
		if (o[i] <= threshold) {
			*(ret_img.ptr<int>(j)) = *(I->ptr<int>(i));
			j++;
		}
	}

	return ret_img;
}

short* get_pick_in_nms(short* pick, int counter)
{
	int i = 0;
	short* ret_ptr = (short*)malloc(counter * sizeof(short));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

	for (i = 0; i < counter; i++) {
		ret_ptr[i] = pick[i];
	}

	return ret_ptr;
}

short* nms(Mat* img, float threshold, const char* method, int* pick_len)
{
    if (img->cols * img->rows == 0) {
        return NULL;
    }

    int counter = 0;
    int len = img->rows;
    short pick[len] = {0};
	Mat I;

	float* area = NULL;
	int* idx = NULL;
	float* xx1 = NULL;
	float* yy1 = NULL;
	float* xx2 = NULL;
	float* yy2 = NULL;
	float* w = NULL;
	float* h = NULL;
	float* inter = NULL;
	float* o = NULL;

	Mat x1 = img->colRange(0, 1).clone();
	Mat y1 = img->colRange(1, 2).clone();
	Mat x2 = img->colRange(2, 3).clone();
	Mat y2 = img->colRange(3, 4).clone();
	Mat s = img->colRange(4, 5).clone();

    area = get_area(&x1, &y1, &x2, &y2, len);
	if (area == NULL) {
		goto AREA_MALLOC_ERROR;
	}

	sortIdx(s, I, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

	while (len > 0) {
		int i = *(I.ptr<int>(I.rows - 1));
        pick[counter] = i;
        counter++;
        len = I.rows - 1;

        idx = get_idx(&I);
		if (idx == NULL) {
			goto IDX_MALLOC_ERROR;
		}

		xx1 = maximum(&x1, i, idx, len);
		if (xx1 == NULL) {
			goto XX1_MALLOC_ERROR;
		}

        yy1 = maximum(&y1, i, idx, len);
		if (yy1 == NULL) {
			goto YY1_MALLOC_ERROR;
		}

        xx2 = minimum(&x2, i, idx, len);
		if (xx2 == NULL) {
			goto XX2_MALLOC_ERROR;
		}

        yy2 = minimum(&y2, i, idx, len);
		if (yy2 == NULL) {
			goto YY2_MALLOC_ERROR;
		}

		w = get_wh_in_nms(xx1, xx2, 0.0, len);
		if (w == NULL) {
			goto W_MALLOC_ERROR;
		}

        h = get_wh_in_nms(yy1, yy2, 0.0, len);
		if (h == NULL) {
			goto H_MALLOC_ERROR;
		}

        inter = get_inter(w, h, len);
		if (inter == NULL) {
			goto INTER_MALLOC_ERROR;
		}

		if (!strcmp(method, "Min")) {
			float* tmp = minimum(area, area[i], idx, len);
			if (tmp == NULL) {
				goto O_MALLOC_ERROR;
			}

			o = get_o_Min(inter, tmp, len);
			if (o == NULL) {
				free(tmp);
				goto O_MALLOC_ERROR;
			}

			free(tmp);
		}else {
			o = get_o(inter, area, i, idx, len);
			if (o == NULL) {
				goto O_MALLOC_ERROR;
			}
		}

		I = update_I(&I, o, threshold);
		len = I.rows;

		free(o);
        free(inter);
        free(h);
        free(w);
		free(xx1);
		free(xx2);
		free(yy1);
		free(yy2);
        free(idx);
    }

	*pick_len = counter;

    free(area);
    return get_pick_in_nms(pick, counter);

O_MALLOC_ERROR:
		free(inter);
INTER_MALLOC_ERROR:
		free(h);
H_MALLOC_ERROR:
		free(w);
W_MALLOC_ERROR:
		free(yy2);
YY2_MALLOC_ERROR:
		free(xx2);
XX2_MALLOC_ERROR:
		free(yy1);
YY1_MALLOC_ERROR:
		free(xx1);
XX1_MALLOC_ERROR:
		free(idx);
IDX_MALLOC_ERROR:
		free(area);
AREA_MALLOC_ERROR:
		return NULL;
}

Mat get_boxes_from_pick(Mat* img, short* pick, int pick_len)
{
	int i = 0, j = 0;
	float* ptr_src = NULL;
	float* ptr_dst = NULL;

	Mat ret_img = Mat::zeros(pick_len, img->cols, CV_32FC1);

	for (i = 0; i < ret_img.rows; i++) {
		ptr_src = img->ptr<float>(pick[i]);
		ptr_dst = ret_img.ptr<float>(i);
		for (j = 0; j < ret_img.cols; j++) {
			*ptr_dst = *ptr_src;
			ptr_src++;
			ptr_dst++;
		}
	}

	return ret_img;
}

Mat append_total_boxes(Mat* total_boxes, Mat* boxes)
{
	int i = 0, j = 0;
	float* ptr_src = NULL;
	float* ptr_dst = NULL;

	Mat ret_img = Mat::zeros((total_boxes->rows + boxes->rows), boxes->cols, CV_32FC1);

	for (i = 0; i < ret_img.rows; i++) {
		if (i < total_boxes->rows) {
			ptr_src = total_boxes->ptr<float>(i);
		} else {
			ptr_src = boxes->ptr<float>(i - total_boxes->rows);
		}
		ptr_dst = ret_img.ptr<float>(i);
		for (j = 0; j < ret_img.cols; j++) {
			*ptr_dst = *ptr_src;
			ptr_src++;
			ptr_dst++;
		}
	}

	return ret_img;
}

float* mat_cols_sub(Mat subed, Mat sub)
{
	int i = 0;
	float* ret_ptr = (float*)malloc(subed.rows * sizeof(float));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

	for (i = 0; i < subed.rows; i++) {
		ret_ptr[i] = *(subed.ptr<float>(i)) - *(sub.ptr<float>(i));
	}

	return ret_ptr;
}

float* get_qq(Mat* img, int x, int y, float* reg)
{
	int i = 0;
	float* ret_ptr = (float*)malloc(img->rows * sizeof(float));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

	for (i = 0; i < img->rows; i++) {
		ret_ptr[i] = *(img->ptr<float>(i, x)) + *(img->ptr<float>(i, y)) * reg[i];
	}

	return ret_ptr;
}

Mat get_vstack_qq_and_transpose(float* qq1, float* qq2, float* qq3, float* qq4, Mat* total_boxes, int index)
{
	int i = 0;
	Mat ret_img = Mat::zeros(total_boxes->rows, 5, CV_32FC1);

	for (i = 0; i < ret_img.rows; i++) {
		*(ret_img.ptr<float>(i, 0))= qq1[i];
	}
	for (i = 0; i < ret_img.rows; i++) {
		*(ret_img.ptr<float>(i, 1))= qq2[i];
	}
	for (i = 0; i < ret_img.rows; i++) {
		*(ret_img.ptr<float>(i, 2))= qq3[i];
	}
	for (i = 0; i < ret_img.rows; i++) {
		*(ret_img.ptr<float>(i, 3))= qq4[i];
	}
	for (i = 0; i < ret_img.rows; i++) {
		*(ret_img.ptr<float>(i, 4))= *(total_boxes->ptr<float>(i, index));
	}

	return ret_img;
}

void get_bboxA(Mat* img, float* x, float* y, int index)
{
	int i = 0;

	for (i = 0; i < img->rows; i++) {
		*(img->ptr<float>(i, index)) = *(img->ptr<float>(i, index)) + x[i] * 0.5 - y[i] * 0.5;
	}

	return ;
}

Mat tile(float* l, int y, int x, int len)
{
	int i = 0, j = 0;
	float* ptr = NULL;
	Mat ret_img = Mat::zeros(y, len * x, CV_32FC1);

	for (i = 0; i < ret_img.rows; i++) {
		ptr = ret_img.ptr<float>(i);
		for (j = 0; j < ret_img.cols; j++) {
			*ptr = l[j];
			ptr++;
		}
	}

	transpose(ret_img, ret_img);

	return ret_img;
}

void get_ret_rerec(Mat* img, int xs, int xe, int ys, int ye, Mat* tmp)
{
	int i = 0, j = 0;
	float* ptr_src = NULL;
	float* ptr_dst = NULL;

	for (i = 0; i < img->rows; i++) {
		ptr_src = img->ptr<float>(i, xs);
		ptr_dst = img->ptr<float>(i, ys);
		for (j = 0; j < (xe-xs); j++) {
			*ptr_dst = *ptr_src + *(tmp->ptr<float>(i, j));
			ptr_src++;
			ptr_dst++;
		}
	}

	return ;
}

int rerec(Mat* img)
{
	int i = 0;
	float* h = mat_cols_sub(img->colRange(3, 4), img->colRange(1, 2));
	if (h == NULL) {
		return -1;
	}

	float* w = mat_cols_sub(img->colRange(2, 3), img->colRange(0, 1));
	if (w == NULL) {
		free(h);
		return -1;
	}

	float* l = (float*)malloc(img->rows * sizeof(float));
	if (l == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		free(w);
		free(h);
		return -1;
	}

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

	return 0;
}

void get_total_boxes_fix(Mat* img, int xs, int xe, int ys, int ye)
{
	int i = 0, j = 0;
	int tmp_var = 0;
	float* ptr_src = NULL;
	float* ptr_dst = NULL;

	for (i = 0; i < img->rows; i++) {
		ptr_src = img->ptr<float>(i, xs);
		ptr_dst = img->ptr<float>(i, ys);
		for (j = 0; j < (xe - xs); j++) {
			tmp_var = *ptr_src;
			*ptr_dst = tmp_var > 0 ? floor(tmp_var) : ceil(tmp_var);
			ptr_src++;
			ptr_dst++;
		}
	}

	return ;
}

void get_tmpwh(Mat* img, int* tmp, int x, int y)
{
	int i = 0;

	for (i = 0; i < img->rows; i++) {
		tmp[i] = *(img->ptr<float>(i, x)) - *(img->ptr<float>(i, y)) + 1;
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
	int i = 0;;

	for (i = 0; i < img->rows; i++) {
		x[i] = *(img->ptr<float>(i, 0));
	}
	for (i = 0; i < img->rows; i++) {
		y[i] = *(img->ptr<float>(i, 1));
	}
	for (i = 0; i < img->rows; i++) {
		ex[i] = *(img->ptr<float>(i, 2));
	}
	for (i = 0; i < img->rows; i++) {
		ey[i] = *(img->ptr<float>(i, 3));
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
	float* dst = NULL;

	for (i = 0; i < (ey[k] - y[k] + 1); i++) {
		for (j = 0; j < (ex[k] - x[k] + 1); j++) {
			src = img->ptr<uchar>((y[k] - 1 + i), (x[k] - 1 + j));
			dst = tmp->ptr<float>((dy[k] - 1 + i), (dx[k] - 1 + j));
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
	int i = 0, j = 0, v = 0;
	float* ptr_src = NULL;
	float* ptr_dst = (float*)tempimg->data + k;

	for (i = 0; i < tempimg->size().height; i++) {
		for (j = 0; j < tempimg->size().width; j++) {
			ptr_src = tmp_tempimg->ptr<float>(i, j);
			for (v = 0; v < len; v++) {
				*ptr_dst = *ptr_src;
				ptr_src++;
				ptr_dst += tempimg->channels();
			}
		}
	}

	return ;
}

int* get_ipass(Mat* score, float threshold, int len, int* ipass_len)
{
	int i = 0;
	int count = 0;
	float* ptr = score->ptr<float>(0);

	for (i = 0; i < len; i++) {
		if (*ptr > threshold)
			count++;
		ptr++;
	}

	if (count == 0)
		return NULL;

	*ipass_len = count;
	int* ret_ptr = (int*)malloc(count * sizeof(int));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

	int* tmp_ptr = ret_ptr;
	ptr = score->ptr<float>(0);

	for (i = 0; i < len; i++) {
		if (*ptr > threshold) {
			*tmp_ptr = i;
			tmp_ptr++;
		}
		ptr++;
	}

	return ret_ptr;
}

Mat get_hstack_ronet(Mat* img, int* ipass, int x, int y, Mat* score, int ipass_len)
{
	int i = 0, j = 0;
	float* ptr = score->ptr<float>(0);

	Mat ret_img = Mat::zeros(ipass_len, img->cols, CV_32FC1);

	for(i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			if (x <= j && j < y) {
				*(ret_img.ptr<float>(i, j)) = *(img->ptr<float>(ipass[i], j));
			} else {
				*(ret_img.ptr<float>(i, j)) = *(ptr + ipass[i]);
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

Mat get_total_boxes_pick(Mat* img, short* pick, int len)
{
	int i = 0, j = 0;
	Mat ret_img = Mat::zeros(len, img->cols, CV_32FC1);

	for (i = 0; i < len; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			*(ret_img.ptr<float>(i, j)) = *(img->ptr<float>(pick[i], j));
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

void get_wh_in_bbreg(Mat* img, float* w, float* h, int len)
{
	int i = 0;
	for (i = 0; i < len; i++) {
		w[i] = *(img->ptr<float>(i, 2)) - *(img->ptr<float>(i, 0)) + 1;
		h[i] = *(img->ptr<float>(i, 3)) - *(img->ptr<float>(i, 1)) + 1;
	}

	return ;
}

void get_b_in_bbreg(Mat* img, Mat* mv, float* b1, float* b2, float* b3, float* b4, float* w, float* h, int len)
{
	int i = 0;
	for (i = 0; i < len; i++) {
		b1[i] = *(img->ptr<float>(i, 0)) + *(mv->ptr<float>(i, 0)) * w[i];
		b2[i] = *(img->ptr<float>(i, 1)) + *(mv->ptr<float>(i, 1)) * h[i];
		b3[i] = *(img->ptr<float>(i, 2)) + *(mv->ptr<float>(i, 2)) * w[i];
		b4[i] = *(img->ptr<float>(i, 3)) + *(mv->ptr<float>(i, 3)) * h[i];
	}

	return ;
}

void get_bbreg_return(Mat* img, float* b1, float* b2, float* b3, float* b4, int len)
{
	int i = 0, j = 0;
	Mat tmp = Mat::zeros(4, len, CV_32FC1);
	for (i = 0; i < tmp.cols; i++) {
		*(tmp.ptr<float>(0, i)) = b1[i];
		*(tmp.ptr<float>(1, i)) = b2[i];
		*(tmp.ptr<float>(2, i)) = b3[i];
		*(tmp.ptr<float>(3, i)) = b4[i];
	}

	transpose(tmp, tmp);

	for (i = 0; i < tmp.rows; i++) {
		for (j = 0; j < 4; j++) {
			*(img->ptr<float>(i, j)) = *(tmp.ptr<float>(i, j));
		}
	}

	return ;
}

int bbreg(Mat* boundingbox, Mat* reg)
{
	if (reg->cols == 1) {
		/* reg = np.reshape(reg, (reg.shape[2], reg.shape[3])); */
		return -1;
	}

	int len = reg->rows;

	float* w = (float*)malloc(len * sizeof(float));
	float* h = (float*)malloc(len * sizeof(float));
	float* b1 = (float*)malloc(len * sizeof(float));
	float* b2 = (float*)malloc(len * sizeof(float));
	float* b3 = (float*)malloc(len * sizeof(float));
	float* b4 = (float*)malloc(len * sizeof(float));
	if (w == NULL || h == NULL || b1 == NULL || b2 == NULL || b3 == NULL || b4 == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return -1;
	}

	get_wh_in_bbreg(boundingbox, w, h, len);
	get_b_in_bbreg(boundingbox, reg, b1, b2, b3, b4, w, h, len);

	get_bbreg_return(boundingbox, b1, b2, b3, b4, len);

	free(w);
	free(h);
	free(b1);
	free(b2);
	free(b3);
	free(b4);
	return 0;
}

Mat fix_total_boxes(Mat* img)
{
	int i = 0, j = 0;
	int tmp_var = 0;
	Mat ret_img = Mat::zeros(img->rows, img->cols, CV_32FC1);
	for (i = 0; i < img->rows; i++) {
		for (j = 0; j < img->cols; j++) {
			tmp_var = *(img->ptr<float>(i, j));
			*(ret_img.ptr<float>(i, j)) = tmp_var > 0 ? floor(tmp_var) : ceil(tmp_var);
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

void update_points(Mat* img, Mat* total_boxes, float* w, float* h, int len)
{
	int i = 0, j = 0;

	for (i = 0; i < 5; i++) {
		for (j = 0; j < len; j++) {
			*(img->ptr<float>(i, j)) = (w[j] * *(img->ptr<float>(i, j))) + (*(total_boxes->ptr<float>(j, 0)) - 1);
			*(img->ptr<float>(i + 5, j)) = (h[j] * *(img->ptr<float>(i + 5, j))) + (*(total_boxes->ptr<float>(j, 1)) - 1);
		}
	}

	return ;
}

float* get_o_Min(float* inter, float* tmp, int len)
{
	int i = 0;
	float* ret_ptr = (float*)malloc(len * sizeof(float));
	if (ret_ptr == NULL) {
		printf("malloc failed in %s %d lines\n", __func__, __LINE__);
		return NULL;
	}

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
	int i = 0;
	float* ptr = NULL;
	Mat ret_img;

#if 0
	int j = 0;
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

	Align(*img, ret_img, source_pts);
#endif

	return ret_img;
}
