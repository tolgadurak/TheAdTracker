#ifndef HELPERS_H
#define HELPERS_H

#include "opencv2/imgproc/imgproc.hpp" //Error without it 

bool rectInImage(cv::Rect rect, cv::Mat image);
bool inMat(cv::Point p, int rows, int cols);
cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);

#endif