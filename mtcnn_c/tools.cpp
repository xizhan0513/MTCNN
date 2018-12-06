#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include "mtcnn.h"

using namespace std;
using namespace cv;

void print_Mat_uchar(Mat* img)
{
    int i = 0, j = 0, k = 0;
    unsigned char* ptr = NULL;
    printf("rows = %d, cols = %d, channels = %d\n", img->rows, img->cols, img->channels());
    printf("[");
    for (i = 0; i < img->rows; i++) {
        printf("[");
        for (j = 0; j < img->cols; j++) {
            ptr = img->ptr<uchar>(i, j);
            printf("[");
            for(k = 0; k < img->channels(); k++) {
                printf("%d ", *ptr);
				ptr++;
            }
            printf("]\n");
        }
        printf("]\n");
    }
    printf("]\n");
}

void print_Mat_double(Mat* img)
{
    int i = 0, j = 0, k = 0;
    float* ptr = NULL;
    printf("rows = %d, cols = %d, channels = %d\n", img->rows, img->cols, img->channels());
    printf("[");
    for (i = 0; i < img->rows; i++) {
        printf("[");
        for (j = 0; j < img->cols; j++) {
            ptr = img->ptr<float>(i, j);
            printf("[");
            for(k = 0; k < img->channels(); k++) {
                printf("%.8f ", *ptr);
				ptr++;
            }
            printf("]\n");
        }
        printf("]\n");
    }
    printf("]\n");
}

void print_Mat_3DMat(Mat* img, int img_len)
{
    int i = 0, j = 0, k = 0, v = 0;
    float* ptr = NULL;
    printf("[");
    for (i = 0; i < img->size().height; i++) {
        printf("[");
        for (j = 0; j < img->size().width; j++) {
            printf("[");
            for(k = 0; k < img_len; k++) {
				ptr = (float*)(img->data + img->step[0]*i + img->step[1]*j + img->step[2]*k);
				printf("[");
				for (v = 0; v < img->channels(); v++) {
					printf("%.10f ", *ptr);
					ptr++;
				}
				printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
    }
    printf("]\n");
}
void save_diff_file_uchar(Mat* img)
{
	int i = 0, j = 0, k = 0;
    unsigned char* ptr = NULL;
	FILE* f = fopen("uchar.bin", "wb+");

    for (i = 0; i < img->rows; i++) {
        for (j = 0; j < img->cols; j++) {
            ptr = img->ptr<uchar>(i, j);
            for (k = 0; k < img->channels(); k++) {
                    fwrite(ptr, 1, 1, f);
                    ptr++;
                }
        }
    }
    fclose(f);
    return ;
}

void save_diff_file_double(Mat* img)
{
	int i = 0, j = 0, k = 0;
    float* ptr = NULL;
	FILE* f = fopen("double.bin", "wb+");

    for (i = 0; i < img->rows; i++) {
        for (j = 0; j < img->cols; j++) {
            ptr = img->ptr<float>(i, j);
            for (k = 0; k < img->channels(); k++) {
                    fwrite(ptr, 4, 1, f);
                    ptr++;
                }
        }
    }
    fclose(f);
    return ;
}

void save_diff_file_3DMat(Mat* img, int img_len)
{
	int i = 0, j = 0, k = 0, v = 0;
    float* ptr = NULL;
	FILE* f = fopen("3DMat.bin", "wb+");

    for (i = 0; i < img->size().height; i++) {
        for (j = 0; j < img->size().width; j++) {
            for (k = 0; k < img_len; k++) {
				ptr = (float*)(img->data + img->step[0]*i + img->step[1]*j + img->step[2]*k);
				for (v = 0; v < img->channels(); v++) {
					fwrite(ptr, 4, 1, f);
                    ptr++;
				}
            }
        }
    }
    fclose(f);
    return ;
}

void save_diff_file(Mat* img)
{
	int i = 0, j = 0, k = 0;
    double* ptr = NULL;
	FILE* f = fopen("tmp.bin", "wb+");

    for (i = 0; i < img->rows; i++) {
        ptr = img->ptr<double>(i);
        for (j = 0; j < img->cols; j++) {
			fwrite(ptr, 8, 1, f);
            ptr++;
        }
    }
    fclose(f);
    return ;
}

void print_Mat(Mat* img)
{
	int i = 0, j = 0, k = 0;
    float* ptr = NULL;

    for (i = 0; i < img->rows; i++) {
        printf("[");
		ptr = img->ptr<float>(i);
        for (j = 0; j < img->cols; j++) {
            printf("%.8f ", *ptr);
			ptr++;
        }
		printf("]\n");
    }
    return ;
}

