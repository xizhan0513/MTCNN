#include "mtcnn.h"

using namespace std;
using namespace cv;

void print_1D(short* img, int len)
{
	int i = 0;

	printf("len = %d\n", len);
	printf("[");
	for (i = 0; i < len; i++) {
		printf("%hd ", img[i]);
	}
	printf("]\n");

	return ;
}

void print_1D(int* img, int len)
{
	int i = 0;

	printf("len = %d\n", len);
	printf("[");
	for (i = 0; i < len; i++) {
		printf("%d ", img[i]);
	}
	printf("]\n");

	return ;
}

void print_1D(float* img, int len)
{
	int i = 0;

	printf("len = %d\n", len);
	printf("[");
	for (i = 0; i < len; i++) {
		printf("%.8f ", img[i]);
	}
	printf("]\n");

	return ;
}

void print_2D(Mat* img, int type)
{
	int i = 0, j = 0;
    int* ptr = NULL;

	printf("rows = %d, cols = %d, channels = %d\n", img->rows, img->cols, img->channels());
    for (i = 0; i < img->rows; i++) {
        printf("[");
		ptr = img->ptr<int>(i);
        for (j = 0; j < img->cols; j++) {
            printf("%d ", *ptr);
			ptr++;
        }
		printf("]\n");
    }

    return ;
}

void print_2D(Mat* img, float type)
{
	int i = 0, j = 0;
    float* ptr = NULL;

	printf("rows = %d, cols = %d, channels = %d\n", img->rows, img->cols, img->channels());
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

void print_3D(Mat* img, unsigned char type)
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

	return ;
}

void print_3D(Mat* img, float type)
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

	return ;
}

void print_4D(Mat* img, int len, float type)
{
    int i = 0, j = 0, k = 0, v = 0;
    float* ptr = NULL;

	printf("height = %d, width = %d, lenth = %d, channels = %d\n", img->size().height, img->size().width, len, img->channels());
    printf("[");
    for (i = 0; i < img->size().height; i++) {
        printf("[");
        for (j = 0; j < img->size().width; j++) {
            printf("[");
            for(k = 0; k < len; k++) {
				ptr = (float*)(img->data + img->step[0] * i + img->step[1] * j + img->step[2] * k);
				printf("[");
				for (v = 0; v < img->channels(); v++) {
					printf("%.8f ", *ptr);
					ptr++;
				}
				printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
    }
    printf("]\n");

	return ;
}

void save_diff_file_2D(Mat* img, float type)
{
	int i = 0, j = 0;
    float* ptr = NULL;

	FILE* f = fopen("2D_float.bin", "wb+");

    for (i = 0; i < img->rows; i++) {
        ptr = img->ptr<float>(i);
        for (j = 0; j < img->cols; j++) {
			fwrite(ptr, 4, 1, f);
            ptr++;
        }
    }

    fclose(f);
    return ;
}

void save_diff_file_3D(Mat* img, unsigned char type)
{
	int i = 0, j = 0, k = 0;
    unsigned char* ptr = NULL;

	FILE* f = fopen("3D_uchar.bin", "wb+");

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

void save_diff_file_3D(Mat* img, float type)
{
	int i = 0, j = 0, k = 0;
    float* ptr = NULL;

	FILE* f = fopen("3D_float.bin", "wb+");

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

void save_diff_file_4D(Mat* img, int len, float type)
{
	int i = 0, j = 0, k = 0, v = 0;
    float* ptr = NULL;

	FILE* f = fopen("4D_float.bin", "wb+");

    for (i = 0; i < img->size().height; i++) {
        for (j = 0; j < img->size().width; j++) {
            for (k = 0; k < len; k++) {
				ptr = (float*)(img->data + img->step[0] * i + img->step[1] * j + img->step[2] * k);
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
