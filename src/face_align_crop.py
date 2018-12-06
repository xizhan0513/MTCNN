import sys
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os
#from scipy import misc
from detect_face_2 import detect_face
import cv2
from skimage import transform as trans
from tools import *

def get_points(img, model_path, threshold, scales):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    def create_net(sess):
        pnet_fun = lambda img: sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'),feed_dict={'pnet/input:0': img})
        rnet_fun = lambda img: sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'),feed_dict={'rnet/input:0': img})
        onet_fun = lambda img: sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'),feed_dict={'onet/input:0': img})
        return pnet_fun, rnet_fun, onet_fun

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            pnet,rnet,onet = create_net(sess)
            bounding_boxes, points = detect_face(img, pnet, rnet, onet, threshold, scales)
    return bounding_boxes, points

def face_preprocess(img, landmark=None):
    image_size = [112,112]
    src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ], dtype=np.float32)
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
    return warped

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # parameters
    minsize = 20
    threshold = [0.8,0.85,0.9]
    factor = 0.85

#    img = misc.imread(args.img_path)
    img = cv2.imread(args.img_path)
    img = img[...,::-1]
    if img.ndim != 3:
        print('Unable to align "%s", img dim error' % args.img_path)
    else:
        factor_count = 0
        h = img.shape[0]
        w = img.shape[1]
        minl = np.amin([h, w])
        m = 12.0 / minsize
        minl = minl * m
        scales = []
        while minl >=12:
            scales += [m*np.power(factor, factor_count)]
            minl = minl * factor
            factor_count += 1


        bounding_boxes, points = get_points(img, args.model_path, threshold, scales)
        print(bounding_boxes)
        print(points)
        _landmark = points[:,0].reshape((2,5)).T
        warped = face_preprocess(img, landmark=_landmark)
        bgr = warped[...,::-1]
        cv2.imwrite(args.output_path + '1.jpg', bgr)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-path', type=str)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--output-path', type=str)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
