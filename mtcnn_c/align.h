#ifndef ALIGN_H_
#define ALIGN_H_

#include "mtcnn.h"

int Align(cv::Mat& image, cv::Mat& image_crop, std::vector<cv::Point2d> source_pts);
cv::Mat findSimilarityTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, cv::Mat& Tinv);

#endif
