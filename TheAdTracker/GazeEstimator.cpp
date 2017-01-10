//
//  GazeEstimator.cpp
//  TheAdTracker
//
//  Created by Tolga Durak on 27/11/2016.
//  Copyright © 2016 Tolga Durak. All rights reserved.
//

#include "GazeEstimator.hpp"
#include "EyeCenterDetector.hpp"
#include <opencv2/opencv.hpp>

GazeEstimator::GazeEstimator(cv::Rect face, cv::Mat frame, const dlib::shape_predictor* pose_model, EyeCenterDetector *detector )
{
    cv::Mat frame_gray = toGrayImage(frame);
    this->frame = frame;
    this->frame_gray = frame_gray;
    this->face = face;
    
    dlib::cv_image<dlib::bgr_pixel> cimg(this->frame);
    dlib::rectangle r(
                      (long)(face.x),
                      (long)(face.y),
                      (long)(face.x + face.width),
                      (long)(face.y + face.height)
                      );
    this->shape = (*pose_model)(cimg, r);
    this->estimateEyePoints();
    this->estimateHeadPose();
    this->estimateEyeRegions();
    this->estimatePupils(detector);

}

GazeEstimator::~GazeEstimator()
{
    
}

cv::Mat GazeEstimator::toGrayImage(cv::Mat source)
{
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(source, rgbChannels);
    return rgbChannels[2];
}

cv::Point GazeEstimator::getLeftPupil()
{
    return this->leftPupil;
}

cv::Point GazeEstimator::getRightPupil()
{
    return this->rightPupil;
}

std::vector<cv::Point> GazeEstimator::getEyePoints()
{
    return this->eyePoints;
}

cv::Point2d GazeEstimator::getImagePoint()
{
    return this->imagePoint;
}

cv::Point2d GazeEstimator::getNoseEndPoint()
{
    return this->noseEndPoint;
}

void GazeEstimator::estimateEyePoints()
{
    if(!frame.empty() && face.area() > 0)
    {
        this->pushLandmarkPoints(shape, LEFT_EYE_START, LEFT_EYE_END);
        this->pushLandmarkPoints(shape, RIGHT_EYE_START, RIGHT_EYE_END);
    }
    
}

void GazeEstimator::pushLandmarkPoints(const dlib::full_object_detection& d, const int start, const int end)
{
    for(int i = start ; i <= end; ++i)
    {
        this->eyePoints.push_back(cv::Point(static_cast<int>(d.part(i).x()), static_cast<int>(d.part(i).y())));
    }
}

void GazeEstimator::estimateEyeRegions()
{
    if(this->face.area() > 0)
    {
        this->faceROI = this->frame_gray(this->face);
        if(!this->faceROI.empty()) {
            cv::Mat debugFace = this->faceROI;
            //-- Find eye regions and draw them
            int eye_region_width = static_cast<int>(this->face.width * (kEyePercentWidth / 100.0));
            int eye_region_height = static_cast<int>(this->face.width * (kEyePercentHeight / 100.0));
            int eye_region_top = static_cast<int>(this->face.height * (kEyePercentTop / 100.0));
            
            cv::Rect left(static_cast<int>(this->face.width*(kEyePercentSide / 100.0)),
                          eye_region_top, eye_region_width, eye_region_height);
            cv::Rect right(static_cast<int>(this->face.width - eye_region_width - this->face.width*(kEyePercentSide / 100.0)),
                           eye_region_top, eye_region_width, eye_region_height);
            
            this->leftEyeRegion = left;
            this->rightEyeRegion = right;
        }        
    }

}

void GazeEstimator::estimatePupils(EyeCenterDetector *detector)
{
    if(!this->faceROI.empty() && this->leftEyeRegion.area() > 0 && this->rightEyeRegion.area() > 0 && detector != NULL)
    {
        //-- Find Eye Centers
        try
        {
            this->leftPupil = detector->detect(this->faceROI, this->leftEyeRegion, "Left Eye");
            this->rightPupil = detector->detect(this->faceROI, this->rightEyeRegion, "Right Eye");
        }
        catch (const std::exception&)
        {
            
        }
        this->rightPupil.x += this->rightEyeRegion.x;
        this->rightPupil.y += this->rightEyeRegion.y;
        this->leftPupil.x += this->leftEyeRegion.x;
        this->leftPupil.y += this->leftEyeRegion.y;
        
        this->leftPupil += cv::Point(face.x, face.y);
        this->rightPupil += cv::Point(face.x, face.y);
    }
}

std::vector<cv::Point3d> GazeEstimator::get_3d_model_points()
{
    std::vector<cv::Point3d> modelPoints;
    
    modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));
    modelPoints.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));
    modelPoints.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));
    modelPoints.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));
    modelPoints.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));
    modelPoints.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));
    
    return modelPoints;
    
}

std::vector<cv::Point2d> GazeEstimator::get_2d_image_points(dlib::full_object_detection &d)
{
    std::vector<cv::Point2d> image_points;
    image_points.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );    // Nose tip
    image_points.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );      // Chin
    image_points.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );    // Left eye left corner
    image_points.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );    // Right eye right corner
    image_points.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) );    // Left Mouth corner
    image_points.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) );    // Right mouth corner
    return image_points;
    
}

cv::Mat GazeEstimator::get_camera_matrix(float focal_length, cv::Point2d center)
{
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    return camera_matrix;
}
void GazeEstimator::estimateHeadPose()
{
    // Pose estimation
    std::vector<cv::Point3d> model_points = get_3d_model_points();

    std::vector<cv::Point2d> image_points = get_2d_image_points(shape);
    double focal_length = this->frame.cols;
    cv::Mat camera_matrix = get_camera_matrix(focal_length, cv::Point2d(this->frame.cols/2,this->frame.rows/2));
    cv::Mat rotation_vector;
    cv::Mat rotation_matrix;
    cv::Mat translation_vector;
    
    
    cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);
    
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
    
    std::vector<cv::Point3d> nose_end_point3D;
    std::vector<cv::Point2d> nose_end_point2D;
    nose_end_point3D.push_back(cv::Point3d(0,0,1000.0));
    
    cv::projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
    this->imagePoint = image_points[0];
    this->noseEndPoint = nose_end_point2D[0];
}
