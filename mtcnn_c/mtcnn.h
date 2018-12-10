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

Mat get_pnet_out(int*, const char*, int);

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

short* nms(Mat*, float, const char*, int*);

void get_boxes(Mat*, double**, double**, double**, double**, double**);

double* get_area(double*, double*, double*, double*, int);

int* get_I(double*, int);

int* get_idx(int*, int);

void print_Mat(Mat*);

void print_Mat_uchar(Mat*);

void print_Mat_double(Mat*);

void print_Mat_3DMat(Mat*, int);

void save_diff_file_uchar(Mat*);

void save_diff_file_double(Mat*);

void save_diff_file_3DMat(Mat*, int);

void save_diff_file(Mat*);

double* maximum(double*, double, int*, int);

double* minimum(double*, double, int*, int);

void get_boxes(Mat*, double*, double*, double*, double*, double*);

double* get_wh(double*, double*, double, int);

double* get_inter(double*, double*, int);

double* get_o(double*, double*, int, int*, int);

void updata_I(int**, double*, float, int*);

short* get_pick(short*, int);

Mat get_boxes_from_pack(Mat*, short*, int);

Mat get_total_box(Mat*, Mat*);

double* get_reg_wh(Mat*, int, int);

double* get_qq(Mat*, int, int, double*);

Mat get_vstack_qq_and_transpose(double*, double*, double*, double*, Mat*, int);

void rerec(Mat*);

void get_bboxA(Mat*, double*, double*, int);

Mat tile(double*, int, int, int);

void get_ret_rerec(Mat*, int, int, int, int, Mat*);

void get_total_boxes_fix(Mat*, int, int, int, int);

void pad(Mat*, int, int, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);

void init_dx_dy_edx_edy(int*, int*, int*, int*, int*, int*, int);

void init_x_y_ex_ey(Mat*, int*, int*, int*, int*);

#endif
