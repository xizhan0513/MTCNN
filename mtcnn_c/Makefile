CC=arm-linux-gnueabihf-g++
TARGET=mtcnn.elf
INCLUDE=-I /opt/opencv_for_arm_3.4.1/opencv_arm/include
LIBS=-L. -L /opt/opencv_for_arm_3.4.1/opencv_arm/lib/ -lopencv_video -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lgxdnn -ldl -lm -lpthread -lrt
DEPEND=align.cpp detect_face_2.cpp face_align_crop.cpp function.cpp mtcnn_npu.c tools.cpp
CPPFLAGS=-Wall -std=c++11

$(TARGET): $(DEPEND)
	$(CC) $^ -o $@ $(INCLUDE) $(LIBS) $(CPPFLAGS)

clean:
	rm $(TARGET)
