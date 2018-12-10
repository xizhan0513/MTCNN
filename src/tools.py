import sys
import os

def save_diff_file(img, file_name):
    img_size = img.shape

    f = open(file_name, 'wb+')

    for i in range(0, img_size[0]):
        for j in range(0, img_size[1]):
            if len(img_size) == 2:
                tmp = img[i][j]
                f.write(tmp)
            else:
                for k in range(0, img_size[2]):
                    if len(img_size) == 3:
                        tmp = img[i][j][k]
                        f.write(tmp)
                    else:
                        if len(img_size) == 4:
                            for v in range(0, img_size[3]):
                                tmp = img[i][j][k][v]
                                f.write(tmp)
    f.close()
