#include "mtcnn.h"

using namespace cv;

int Align(cv::Mat& image, cv::Mat& image_crop, std::vector<cv::Point2d> source_pts)
{
	vector<Point2d> target_pts(5);
	target_pts[0]=Point2d(38.2946, 51.6963);
	target_pts[1]=Point2d(73.5318, 51.5014);
	target_pts[2]=Point2d(56.0252, 71.7366);
	target_pts[3]=Point2d(41.5493, 92.3655);
	target_pts[4]=Point2d(70.7299, 92.2041);
	cv::Mat rotmat;
	cv::Mat Tinv;
	rotmat = findSimilarityTransform(source_pts, target_pts, Tinv);

	cv::warpAffine(image,image_crop,rotmat,cv::Mat::zeros(112,112,image.type()).size());

	return 0;
}

cv::Point3f transform(cv::Point3f pt, cv::Mat rot, cv::Point3f trans) {
	cv::Point3f res;
    	res.x = rot.at<float>(0, 0)*pt.x + rot.at<float>(0, 1)*pt.y + rot.at<float>(0, 2)*pt.z + trans.x;
    	res.y = rot.at<float>(1, 0)*pt.x + rot.at<float>(1, 1)*pt.y + rot.at<float>(1, 2)*pt.z + trans.y;
    	res.z = rot.at<float>(2, 0)*pt.x + rot.at<float>(2, 1)*pt.y + rot.at<float>(2, 2)*pt.z + trans.z;
    	return res;
}

cv::Mat findNonReflectiveTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, Mat& Tinv) {
    assert(source_points.size() == target_points.size());
    assert(source_points.size() >= 2);
    Mat U = Mat::zeros(target_points.size() * 2, 1, CV_64F);
    Mat X = Mat::zeros(source_points.size() * 2, 4, CV_64F);
    for (unsigned int i = 0; i < target_points.size(); i++) {
      U.at<double>(i * 2, 0) = source_points[i].x;
      U.at<double>(i * 2 + 1, 0) = source_points[i].y;
      X.at<double>(i * 2, 0) = target_points[i].x;
      X.at<double>(i * 2, 1) = target_points[i].y;
      X.at<double>(i * 2, 2) = 1;
      X.at<double>(i * 2, 3) = 0;
      X.at<double>(i * 2 + 1, 0) = target_points[i].y;
      X.at<double>(i * 2 + 1, 1) = -target_points[i].x;
      X.at<double>(i * 2 + 1, 2) = 0;
      X.at<double>(i * 2 + 1, 3) = 1;
    }
    Mat r = X.inv(DECOMP_SVD)*U;
    Tinv = (Mat_<double>(3, 3) << r.at<double>(0), -r.at<double>(1), 0,
                         r.at<double>(1), r.at<double>(0), 0,
                         r.at<double>(2), r.at<double>(3), 1);

    Mat T = Tinv.inv(DECOMP_SVD);
    Tinv = Tinv(Rect(0, 0, 2, 3)).t();
    return T(Rect(0,0,2,3)).t();
}

cv::Mat findSimilarityTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, Mat& Tinv) {
    Mat Tinv1, Tinv2;
    Mat trans1 = findNonReflectiveTransform(source_points, target_points, Tinv1);
    std::vector<Point2d> source_point_reflect;
    for (auto sp : source_points) {
      source_point_reflect.push_back(Point2d(-sp.x, sp.y));
    }
    Mat trans2 = findNonReflectiveTransform(source_point_reflect, target_points, Tinv2);
    trans2.colRange(0,1) *= -1;
    std::vector<Point2d> trans_points1, trans_points2;
    transform(source_points, trans_points1, trans1);
    transform(source_points, trans_points2, trans2);
    double norm1 = norm(Mat(trans_points1), Mat(target_points), NORM_L2);
    double norm2 = norm(Mat(trans_points2), Mat(target_points), NORM_L2);
    Tinv = norm1 < norm2 ? Tinv1 : Tinv2;
    return norm1 < norm2 ? trans1 : trans2;
}
