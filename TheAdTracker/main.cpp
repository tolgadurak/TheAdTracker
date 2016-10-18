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
#include "findEyeCenter.hpp"
#include "VideoFaceDetector.h"




/** Function Headers */
void detectAndDisplay(VideoFaceDetector & detector, cv::Mat frame);

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations

cv::CascadeClassifier face_cascade;
cv::String face_cascade_name = "haarcascade_frontalface_default.xml";


std::string main_window_name = "Capture - Face / Eyes";
std::string face_window_name = "Debug - Face";

cv::Mat debugImage;

dlib::shape_predictor pose_model;
std::string pose_model_name = "shape_predictor_68_face_landmarks.dat";

/**
 * @function main
 */
int main(int argc, const char** argv) {
	cv::Mat frame;
	dlib::deserialize(pose_model_name) >> pose_model;
	cv::namedWindow(main_window_name, CV_WINDOW_NORMAL);
	cv::moveWindow(main_window_name, 400, 100);
	if (kShowFaceWindow) {
		cv::namedWindow(face_window_name, CV_WINDOW_NORMAL);
		cv::moveWindow(face_window_name, 10, 100);
	}
	cv::VideoCapture camera(0);

	if (camera.isOpened()) {
		VideoFaceDetector detector(face_cascade_name, camera);
		while (true) {
			detector >> frame;
			frame.copyTo(debugImage);

			// Apply the classifier to the frame
			if (!frame.empty()) {
				detectAndDisplay(detector, frame);
			}
			else {
				printf(" --(!) No captured frame -- Break!");
				break;
			}
			cv::flip(debugImage, debugImage, 1);
			imshow(main_window_name, debugImage);

			int c = cv::waitKey(10);
			if ((char)c == 'c') { break; }
		}
	}

	return 0;
}

void findEyes(cv::Mat frame_gray, cv::Rect face) {

	cv::Mat faceROI = frame_gray(face);
	cv::Mat debugFace = faceROI;


	if (kSmoothFaceImage) {
		double sigma = kSmoothFaceFactor * face.width;
		GaussianBlur(faceROI, faceROI, cv::Size(0, 0), sigma);
	}
	//-- Find eye regions and draw them
	int eye_region_width = static_cast<int>(face.width * (kEyePercentWidth / 100.0));
	int eye_region_height = static_cast<int>(face.width * (kEyePercentHeight / 100.0));
	int eye_region_top = static_cast<int>(face.height * (kEyePercentTop / 100.0));
	cv::Rect leftEyeRegion(static_cast<int>(face.width*(kEyePercentSide / 100.0)),
		eye_region_top, eye_region_width, eye_region_height);

	cv::Rect rightEyeRegion(static_cast<int>(face.width - eye_region_width - face.width*(kEyePercentSide / 100.0)),
		eye_region_top, eye_region_width, eye_region_height);

	cv::Point leftPupil;
	cv::Point rightPupil;
	//-- Find Eye Centers
	try
	{
		leftPupil = findEyeCenter(faceROI, leftEyeRegion, "Left Eye");
		rightPupil = findEyeCenter(faceROI, rightEyeRegion, "Right Eye");
	}
	catch (const std::exception&)
	{

	}


	if (kShowFaceWindow) {
		// get corner regions
		cv::Rect leftRightCornerRegion(leftEyeRegion);
		leftRightCornerRegion.width -= leftPupil.x;
		leftRightCornerRegion.x += leftPupil.x;
		leftRightCornerRegion.height /= 2;
		leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
		cv::Rect leftLeftCornerRegion(leftEyeRegion);
		leftLeftCornerRegion.width = leftPupil.x;
		leftLeftCornerRegion.height /= 2;
		leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
		cv::Rect rightLeftCornerRegion(rightEyeRegion);
		rightLeftCornerRegion.width = rightPupil.x;
		rightLeftCornerRegion.height /= 2;
		rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
		cv::Rect rightRightCornerRegion(rightEyeRegion);
		rightRightCornerRegion.width -= rightPupil.x;
		rightRightCornerRegion.x += rightPupil.x;
		rightRightCornerRegion.height /= 2;
		rightRightCornerRegion.y += rightRightCornerRegion.height / 2;

		rectangle(debugFace, leftRightCornerRegion, 200);
		rectangle(debugFace, leftLeftCornerRegion, 200);
		rectangle(debugFace, rightLeftCornerRegion, 200);
		rectangle(debugFace, rightRightCornerRegion, 200);
	}

	// change eye centers to face coordinates
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;

	if (kShowFaceWindow) {
		// draw eye centers
		circle(debugFace, rightPupil, 1, 1234, -1);
		circle(debugFace, leftPupil, 1, 1234, -1);
		imshow(face_window_name, faceROI);
	}

	circle(debugImage, rightPupil + cv::Point(face.x, face.y), 1, cv::Scalar(255, 255, 255), -1);
	circle(debugImage, leftPupil + cv::Point(face.x, face.y), 1, cv::Scalar(255, 255, 255), -1);
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay(VideoFaceDetector & detector, cv::Mat frame) {
	cv::Rect face = detector.face();
	cv::Point facePosition = detector.facePosition();
	std::vector<cv::Mat> rgbChannels(3);
	cv::split(frame, rgbChannels);
	cv::Mat frame_gray = rgbChannels[2];

	// Change to dlib's image format. No memory is copied.
	dlib::cv_image<dlib::bgr_pixel> cimg(frame);

	// Find the pose of each face.

	dlib::rectangle r(
		(long)(face.x),
		(long)(face.y),
		(long)(face.x + face.width),
		(long)(face.y + face.height)
	);
	dlib::full_object_detection shape = pose_model(cimg, r);
	render_face(debugImage, shape);
	rectangle(debugImage, face, 1234, 1);
	findEyes(frame_gray, face);
}
