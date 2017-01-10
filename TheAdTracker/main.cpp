#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "render_face.hpp"

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.hpp"
#include "EyeCenterDetector.hpp"
#include "VideoFaceDetector.hpp"
#include "GazeEstimator.hpp"
#include "Renderer.hpp"

/** Function Headers */
cv::Point * estimateGazePoint(VideoFaceDetector * detector, cv::Mat frame);
cv::Point * estimateGazePoint(cv::Mat frame);

/** Global variables */

const cv::String face_cascade_name = "haarcascade_frontalface_default.xml";
const std::string main_window_name = "Capture - Face / Eyes";
const std::string left_eye_window_name = "Left Eye";
const std::string right_eye_window_name = "Right Eye";
const std::string pose_model_name = "shape_predictor_68_face_landmarks.dat";
dlib::shape_predictor pose_model;
cv::Mat debugImage;
EyeCenterDetector eyeCenterDetector;
/**
 * @function main
 */
int main(int argc, const char** argv) {
    cv::Mat frame;
   
    dlib::deserialize(pose_model_name) >> pose_model;
    cv::namedWindow(main_window_name, CV_WINDOW_NORMAL);
    cv::setWindowProperty(main_window_name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    cv::setWindowProperty(main_window_name, CV_WND_PROP_ASPECTRATIO, CV_WINDOW_FREERATIO);
    if(kEyeWindowsEnabled) {
    // Left Eye Window
    cv::namedWindow(left_eye_window_name,CV_WINDOW_NORMAL);
    cv::moveWindow(left_eye_window_name, 0, 780);

    // Right Eye Window
    cv::namedWindow(right_eye_window_name,CV_WINDOW_NORMAL);
    cv::moveWindow(right_eye_window_name, 1320, 780);
    }

    cv::VideoCapture camera(0);
    if (camera.isOpened()) {
        camera.set(CV_CAP_PROP_FRAME_WIDTH,640);
        camera.set(CV_CAP_PROP_FRAME_HEIGHT,480);
        VideoFaceDetector detector(face_cascade_name, camera);
        while (true) {

            detector >> frame;
            frame.copyTo(debugImage);

            // Apply the classifier to the frame
            if (!frame.empty()) {
                estimateGazePoint(&detector, frame);
            }
            else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }
            cv::flip(debugImage, debugImage, 1);
            imshow(main_window_name, debugImage);

            int c = cv::waitKey(1);
            if ((char)c == 27) { break; }
        }
    }

    return 0;
}

/**
 * @function estimateGazePoint
 */
cv::Point * estimateGazePoint(VideoFaceDetector *detector, cv::Mat frame) {
    cv::Rect face = detector->face();
    GazeEstimator estimator(face, frame, &pose_model, &eyeCenterDetector);
    std::vector<cv::Point> points = estimator.getEyePoints();
    cv::Point2d imagePoint = estimator.getImagePoint();
    cv::Point2d noseEndPoint = estimator.getNoseEndPoint();
    Renderer::draw_eye_points(debugImage, points);
    Renderer::draw_pupils(debugImage, estimator.getLeftPupil(), estimator.getRightPupil());
    Renderer::draw_head_pose(debugImage, imagePoint, noseEndPoint);
    return NULL;
}
