//
//  GazeEstimator.hpp
//  TheAdTracker
//
//  Created by Tolga Durak on 27/11/2016.
//  Copyright © 2016 Tolga Durak. All rights reserved.
//

#ifndef GazeEstimator_h
#define GazeEstimator_h

#include "EyeCenterDetector.hpp"
#include "constants.hpp"
#include <dlib/opencv.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>


class GazeEstimator
{
public:
    GazeEstimator(cv::Rect face, cv::Mat frame, const dlib::shape_predictor* pose_model, EyeCenterDetector *detector);
    ~GazeEstimator();
    
    cv::Point getLeftPupil();
    cv::Point getRightPupil();
    cv::Point2d getNoseEndPoint();
    cv::Point2d getImagePoint();
    std::vector<cv::Point> getEyePoints();
    
    
private:
    const int LEFT_EYE_START = 36, LEFT_EYE_END = 41,
    RIGHT_EYE_START = 42, RIGHT_EYE_END = 47;
    
    cv::Mat frame;
    cv::Mat frame_gray;
    cv::Mat faceROI;
    cv::Rect face;
    cv::Point leftPupil;
    cv::Point rightPupil;
    cv::Rect leftEyeRegion;
    cv::Rect rightEyeRegion;
    std::vector<cv::Point> eyePoints;
    cv::Point2d imagePoint;
    cv::Point2d noseEndPoint;
    dlib::full_object_detection shape;

    
    
    void estimatePupils(EyeCenterDetector *detector);
    std::vector<cv::Point3d> get_3d_model_points();
    std::vector<cv::Point2d> get_2d_image_points(dlib::full_object_detection &d);
    cv::Mat get_camera_matrix(float focal_length, cv::Point2d center);
    void estimateHeadPose();
    void estimateEyeRegions();
    void estimateEyePoints();
    cv::Mat toGrayImage(cv::Mat source);
    void pushLandmarkPoints(const dlib::full_object_detection& d, const int start, const int end);
};

#endif /* GazeEstimator_h */
