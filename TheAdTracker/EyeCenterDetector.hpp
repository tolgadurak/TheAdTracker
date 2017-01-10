#pragma once
#include "opencv2/imgproc/imgproc.hpp"
class EyeCenterDetector
{
public:
    EyeCenterDetector();
    ~EyeCenterDetector();
    
    cv::Point   detect(cv::Mat face, cv::Rect eye, std::string debugWindow);
    
private:
    
    cv::Point   unscalePoint(cv::Point p, cv::Rect origSize);
    void        scaleToFastSize(const cv::Mat &src, cv::Mat &dst);
    cv::Mat     computeMatXGradient(const cv::Mat &mat);
    void        testPossibleCentersFormula(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out);
    bool        floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat);
    cv::Mat     floodKillEdges(cv::Mat &mat);

};

