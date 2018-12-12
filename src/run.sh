python face_align_crop.py --img-path test.jpg --model-path pb/mtcnn.pb --output-path ./
mv *.0 *.1 ../mtcnn_c
mv rout0.bin rout1.bin ../mtcnn_c
mv oout0.bin oout1.bin oout2.bin ../mtcnn_c
