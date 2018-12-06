#ifndef __MTCNN_H__
#define __MTCNN_H__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int detect_face(Mat*, float*, double*, int);

void print_Mat(Mat*);

void print_Mat_uchar(Mat*);

void print_Mat_double(Mat*);

void print_Mat_3DMat(Mat*, int);

void save_diff_file_uchar(Mat*);

void save_diff_file_double(Mat*);

void save_diff_file_3DMat(Mat*, int);

Mat opencv3_transpose_201(Mat*);

Mat opencv3_transpose_021(Mat*);

Mat imresample(Mat*, int, int);

void image_normalization(Mat*, Mat*);

void expand_dims(Mat* ,Mat*);
void expand_dims_dump(Mat* ,Mat*);

Mat get_pnet_out(int, int, int, int, const char*);

Mat opencv3_transpose_0132(Mat*, int);

void get_2D(Mat*, Mat*, int);

void save_diff_file(Mat*);

Mat generateBoundingBox(Mat, Mat, double, float, Mat*);

void where(Mat*, float, int*, int*);

void get_value(Mat*, int*, int*, float*, int);

void vstack_out(Mat*, Mat*, Mat*, Mat*, Mat*, int*, int* y, int);

void vstack(Mat*, int*, int*, int);

void get_q1(Mat*, Mat*, int, double);

void get_q2(Mat*, Mat*, int, double, int);

void expand_dims_0_to_1(float*, Mat*);

void get_hstack(Mat*, Mat*, Mat*, Mat*, Mat*);

short* nms(Mat*, float, const char*);

void get_boxes(Mat*, double*, double*, double*, double*, double*);

void get_area(double*, double*, double*, double*, double*);

void get_argsort(double*, int*);

void maximum(double*, double*, double, int*, int);

void minimum(double*, double*, double, int*, int);
int* get_idx(double*, int);
void get_wh(double*, double*, double*, double);

void get_inter(double*, double*, double*);

void get_o(double, double, double, int, int*);
#endif
