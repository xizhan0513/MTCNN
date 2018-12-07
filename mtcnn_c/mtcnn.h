#ifndef __MTCNN_H__
#define __MTCNN_H__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int detect_face(Mat*, float*, double*, int);

Mat get_img_data(Mat*, int, int);

Mat transpose_uchar_201(Mat*);

Mat  image_normalization(Mat*);

Mat get_img_y(Mat*);

Mat transpose_double_021(Mat);

Mat expand_dims(Mat*);

Mat get_pnet_out(int*, char*, int);

Mat get_in0(Mat*);

Mat get_in1(Mat*);

Mat get_float_2D(Mat*, int);

Mat generateBoundingBox(Mat*, Mat*, double, float);

void get_xy(Mat*, int**, int**, float, int*);

float* get_score(Mat*, int*, int*, int);

Mat get_vstack(Mat*, Mat*, Mat*, Mat*, int*, int*, int);

Mat get_bb(int*, int*, int);

Mat get_q1(Mat*, int, double, int);

Mat get_q2(Mat*, int, double, int, int);

Mat get_dims_0_to_1(float*, int);

Mat get_hstack(Mat*, Mat*, Mat*, Mat*, int);







void print_Mat(Mat*);

void print_Mat_uchar(Mat*);

void print_Mat_double(Mat*);

void print_Mat_3DMat(Mat*, int);

void save_diff_file_uchar(Mat*);

void save_diff_file_double(Mat*);

void save_diff_file_3DMat(Mat*, int);

void save_diff_file(Mat*);




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
