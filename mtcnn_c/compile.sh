g++ face_align_crop.cpp detect_face_2.cpp function.cpp tools.cpp align.cpp -o test `pkg-config opencv --cflags --libs` -std=c++11
mv test net_output_file
