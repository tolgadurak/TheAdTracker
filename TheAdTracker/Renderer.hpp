//
//  Renderer.hpp
//  TheAdTracker
//
//  Created by Tolga Durak on 29/11/2016.
//  Copyright © 2016 Tolga Durak. All rights reserved.
//

#ifndef Renderer_hpp
#define Renderer_hpp

#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
class Renderer
{
public:
        static void draw_eye_points(cv::Mat img, std::vector <cv::Point> points)
        {
            for (cv::Point point : points)
            {
                cv::circle(img, point, 1, cv::Scalar(255, 255, 255), -1);
            }
        }
    
        static void draw_pupils(cv::Mat img, cv::Point p1, cv::Point p2, int radius = 1, cv::Scalar color = cv::Scalar(255, 255, 255), int thickness = 2)
        {
            cv::circle(img, p1, radius, color, thickness);
            cv::circle(img, p2, radius, color, thickness);
        }
    
        static void draw_head_pose(cv::Mat img, cv::Point2d imagePoint, cv::Point2d noseEndPoint, cv::Scalar color = cv::Scalar(255, 255, 255), int thickness = 2)
        {
            cv::line(img,imagePoint, noseEndPoint, color, thickness);
        }
    
    
private:
};
#endif /* Renderer_hpp */
