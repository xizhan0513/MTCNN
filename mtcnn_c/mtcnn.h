#ifndef __MTCNN_H__
#define __MTCNN_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>	/* 包含所有opencv模块，影响编译速度，不影响运行速度的 */
#include "align.h"
#include "gxdnn.h"

using namespace std;
using namespace cv;

#define SCALES_LEN 16

Mat detect_face(Mat*, float*, double*, int, Mat*);

Mat imresample(Mat*, int, int, unsigned char);

Mat imresample(Mat*, int, int, double);

Mat transpose_201(Mat*, unsigned char);

Mat transpose_201(Mat*, double);

Mat transpose_021(Mat*, float);

Mat transpose_021(Mat*, double);

Mat image_normalization(Mat*, unsigned char type);

void image_normalization(Mat*, int, double type);

Mat get_img_y(Mat*);

Mat expand_dims(Mat*);

Mat get_pnet_out(int*, const char*, int);

Mat get_rnet_out(int*, const char*, int);

Mat get_onet_out(int*, const char*);

Mat get_in0(Mat*);

Mat get_in1(Mat*);

Mat from_3DMat_select_rows(Mat*, int);

Mat generateBoundingBox(Mat*, Mat*, double, float);

int get_xy(Mat*, int**, int**, float, int*);

Mat get_score_in_gBB(Mat*, int*, int*, int);

Mat get_vstack_in_gBB(Mat*, Mat*, Mat*, Mat*, int*, int*, int);

Mat get_bb(int*, int*, int);

Mat get_q1(Mat*, int, double, int);

Mat get_q2(Mat*, int, double, int, int);

Mat get_hstack_pnet(Mat*, Mat*, Mat*, Mat*, int);

Mat get_hstack_ronet(Mat*, int*, int, int, Mat*, int);

short* nms(Mat*, float, const char*, int*);

int* get_idx(Mat*);

void print_1D(short*, int);

void print_1D(int*, int);

void print_1D(float*, int);

void print_1D(double*, int);

void print_2D(Mat*, int);

void print_2D(Mat*, float);

void print_2D(Mat*, double);

void print_3D(Mat*, unsigned char);

void print_3D(Mat*, float);

void print_3D(Mat*, double);

void print_4D(Mat*, int, float);

void print_4D(Mat*, int, double);

void save_diff_file_2D(Mat*, float);

void save_diff_file_2D(Mat*, double);

void save_diff_file_3D(Mat*, unsigned char);

void save_diff_file_3D(Mat*, float);

void save_diff_file_3D(Mat*, double);

void save_diff_file_4D(Mat*, int, float);

void save_diff_file_4D(Mat*, int, double);

double* maximum(Mat*, int, int*, int);

double* minimum(Mat*, int, int*, int);

double* minimum(double*, double, int*, int);

double* get_wh(double*, double*, double, int);

double* get_inter(double*, double*, int);

double* get_o(double*, double*, int, int*, int);

Mat update_I(Mat*, double*, float);

short* get_pick_in_nms(short*, int);

double* get_area(Mat*, Mat*, Mat*, Mat*, int);

Mat get_boxes_from_pick(Mat*, short*, int);

Mat append_total_boxes(Mat*, Mat*);

double* mat_cols_sub(Mat, Mat);

double* get_qq(Mat*, int, int, double*);

Mat get_vstack_qq_and_transpose(double*, double*, double*, double*, Mat*, int);

void rerec(Mat*);

void get_bboxA(Mat*, double*, double*, int);

Mat tile(double*, int, int, int);

void get_ret_rerec(Mat*, int, int, int, int, Mat*);

void get_total_boxes_fix(Mat*, int, int, int, int);

void get_tmpwh(Mat*, int*, int, int);

void pad(Mat*, int, int, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);

void init_dx_dy_edx_edy(int*, int*, int*, int*, int*, int*, int);

void init_x_y_ex_ey(Mat*, int*, int*, int*, int*);

void set_exy(int*, int*, int, int*, int);

void set_xy(int*, int*, int, int);

void buckle_map(Mat*, Mat*, int*, int*, int*, int*, int*, int*, int*, int*, int);

void get_tempimg(Mat*, Mat*, int, int);

Mat transpose3021(Mat*, int);

int* get_ipass(Mat*, float, int, int*);

Mat get_mv(Mat*, int*, int);

Mat get_total_boxes_pick(Mat*, short*, int len);

Mat transpose_mv_piack(Mat*, short*, int);

void bbreg(Mat*, Mat*);

void get_wh_in_bbreg(Mat*, double*, double*, int);

void get_b_in_bbreg(Mat*, Mat*, double*, double*, double*, double*, double*, double*, int);

void get_bbreg_return(Mat*, double*, double*, double*, double*, int);

Mat fix_total_boxes(Mat*);

Mat get_points(Mat*, int*, int);

void update_points(Mat*, Mat*, double*, double*, int);

double* get_o_Min(double*, double*, int);

Mat  points_pick(Mat*, short*, int);

Mat face_preprocess(Mat*, Mat*);

#endif
