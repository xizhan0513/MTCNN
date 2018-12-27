#ifndef __MTCNN_H__
#define __MTCNN_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>	/* 包含所有opencv模块，影响编译速度，不影响运行速度的 */
#include "align.h"
#include "gxdnn.h"

using namespace std;
using namespace cv;

#define SCALES_LEN 16			/* 生成图像金字塔的个数 */
#define MAX_FACE_NUM 2			/* 检测人脸最大数量 */
#define NET_MODEL_NUM 11		/* 模型个数 */

struct npu_info {
	int priority;
	int input_num;
	int output_num;
	int input_size;
	int output_size;
	GxDnnTask task;
	GxDnnEventHandler event_handler;
};


Mat detect_face(Mat*, float*, float*, int, Mat*);

Mat imresample(Mat*, int, int, unsigned char);

Mat imresample(Mat*, int, int, float);

Mat transpose_201(Mat*, unsigned char);

Mat transpose_201(Mat*, float);

Mat transpose_021(Mat*);

Mat transpose3021(Mat*, int);

Mat image_normalization(Mat*, unsigned char type);

float* image_normalization(Mat*, int, float type);

float* get_img_y(Mat*);

Mat get_in0(Mat*);

Mat get_in1(Mat*);

Mat from_3DMat_select_rows(Mat*, int);

Mat generateBoundingBox(Mat*, Mat*, float, float);

int get_xy(Mat*, int**, int**, float, int*);

Mat get_score_in_gBB(Mat*, int*, int*, int);

Mat get_vstack_in_gBB(Mat*, Mat*, Mat*, Mat*, int*, int*, int);

Mat get_bb_in_gBB(int*, int*, int);

Mat get_q1(Mat*, int, float, int);

Mat get_q2(Mat*, int, float, int, int);

Mat get_hstack_pnet(Mat*, Mat*, Mat*, Mat*, int);

Mat get_hstack_ronet(Mat*, int*, int, int, Mat*, int);

short* nms(Mat*, float, const char*, int*);

int* get_idx(Mat*);

void print_1D(short*, int);

void print_1D(int*, int);

void print_1D(float*, int);

void print_2D(Mat*, int);

void print_2D(Mat*, float);

void print_3D(Mat*, unsigned char);

void print_3D(Mat*, float);

void print_4D(Mat*, int, float);

void save_diff_file_2D(Mat*, float);

void save_diff_file_3D(Mat*, unsigned char);

void save_diff_file_3D(Mat*, float);

void save_diff_file_4D(Mat*, int, float);

float* maximum(Mat*, int, int*, int);

float* minimum(Mat*, int, int*, int);

float* minimum(float*, float, int*, int);

float* get_wh_in_nms(float*, float*, float, int);

float* get_inter(float*, float*, int);

float* get_o(float*, float*, int, int*, int);

Mat update_I(Mat*, float*, float);

short* get_pick_in_nms(short*, int);

float* get_area(Mat*, Mat*, Mat*, Mat*, int);

Mat get_boxes_from_pick(Mat*, short*, int);

Mat append_total_boxes(Mat*, Mat*);

float* mat_cols_sub(Mat, Mat);

float* get_qq(Mat*, int, int, float*);

Mat get_vstack_qq_and_transpose(float*, float*, float*, float*, Mat*, int);

int rerec(Mat*);

void get_bboxA(Mat*, float*, float*, int);

Mat tile(float*, int, int, int);

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

int* get_ipass(Mat*, float, int, int*);

Mat get_mv(Mat*, int*, int);

Mat get_total_boxes_pick(Mat*, short*, int len);

Mat transpose_mv_piack(Mat*, short*, int);

int bbreg(Mat*, Mat*);

void get_wh_in_bbreg(Mat*, float*, float*, int);

void get_b_in_bbreg(Mat*, Mat*, float*, float*, float*, float*, float*, float*, int);

void get_bbreg_return(Mat*, float*, float*, float*, float*, int);

Mat fix_total_boxes(Mat*);

Mat get_points(Mat*, int*, int);

void update_points(Mat*, Mat*, float*, float*, int);

float* get_o_Min(float*, float*, int);

Mat  points_pick(Mat*, short*, int);

Mat face_preprocess(Mat*, Mat*);

Mat run_pnet(float*, struct npu_info);

Mat run_rnet(float*, struct npu_info, int);

Mat run_onet(float*, struct npu_info, int);

void init_npu_device(GxDnnDevice*);

int load_npu_model(GxDnnDevice, const char**, struct npu_info*, int);

void release_npu_model(GxDnnDevice, struct npu_info*, int);

#endif
