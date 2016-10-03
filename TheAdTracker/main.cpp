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





/** Function Headers */
void detectAndDisplay(cv::Mat frame);

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations

cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face / Eyes";
std::string face_window_name = "Debug - Face";
cv::String face_cascade_name = "haarcascade_frontalface_alt.xml";
std::string pose_model_name = "shape_predictor_68_face_landmarks.dat";
std::string error_loading_face_cascade = "--(!)Error loading face cascade, please change face_cascade_name in source code.\n";
cv::RNG rng(12345);
cv::Mat debugImage;


dlib::shape_predictor pose_model;


/**
 * @function main
 */
int main(int argc, const char** argv) {
	cv::Mat frame;
	// Load the cascades
	if (!face_cascade.load(face_cascade_name)) { std::cout << error_loading_face_cascade << std::endl; return -1; };
	dlib::deserialize(pose_model_name) >> pose_model;
	cv::namedWindow(main_window_name, CV_WINDOW_NORMAL);
	cv::moveWindow(main_window_name, 400, 100);
	if (kShowFaceWindow) {
		cv::namedWindow(face_window_name, CV_WINDOW_NORMAL);
		cv::moveWindow(face_window_name, 10, 100);
	}

	// I make an attempt at supporting both 2.x and 3.x OpenCV
#if CV_MAJOR_VERSION < 3
	CvCapture* capture = cvCaptureFromCAM(0);
	if (capture) {
		while (true) {
			frame = cvQueryFrame(capture);
#else
	cv::VideoCapture capture(0);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 800);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 600);
	if (capture.isOpened()) {

		while (true) {
			capture.read(frame);
#endif
			// mirror it
			cv::flip(frame, frame, 1);
			frame.copyTo(debugImage);

			// Apply the classifier to the frame
			if (!frame.empty()) {
				detectAndDisplay(frame);
			}
			else {
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			imshow(main_window_name, debugImage);

			int c = cv::waitKey(10);
			if ((char)c == 'c') { break; }
			if ((char)c == 'f') {
				imwrite("frame.png", frame);
			}

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
	
	//-- Find Eye Centers
	cv::Point leftPupil = findEyeCenter(faceROI, leftEyeRegion, "Left Eye");
	cv::Point rightPupil = findEyeCenter(faceROI, rightEyeRegion, "Right Eye");



	/*
	 To draw eye location in debugImage
	 */
	if (kDrawEyeRegionsDebugImage) {
		cv::Rect leftEyeRegionDebugImage(leftEyeRegion);
		cv::Rect rightEyeRegionDebugImage(rightEyeRegion);
		leftEyeRegionDebugImage.x += face.x;
		rightEyeRegionDebugImage.x += face.x;
		leftEyeRegionDebugImage.y += face.y;
		rightEyeRegionDebugImage.y += face.y;
		cv::Rect leftLeftCornerRegionDebugImage(leftEyeRegionDebugImage);
		cv::Rect rightRightCornerRegionDebugImage(rightEyeRegionDebugImage);
		cv::Rect leftRightCornerRegionDebugImage(leftEyeRegionDebugImage);
		cv::Rect rightLeftCornerRegionDebugImage(rightEyeRegionDebugImage);

		leftRightCornerRegionDebugImage.width -= leftPupil.x;
		leftRightCornerRegionDebugImage.x += leftPupil.x;
		leftRightCornerRegionDebugImage.height /= 2;
		leftRightCornerRegionDebugImage.y += leftRightCornerRegionDebugImage.height / 2;

		leftLeftCornerRegionDebugImage.width = leftPupil.x;
		leftLeftCornerRegionDebugImage.height /= 2;
		leftLeftCornerRegionDebugImage.y += leftLeftCornerRegionDebugImage.height / 2;

		rightLeftCornerRegionDebugImage.width = rightPupil.x;
		rightLeftCornerRegionDebugImage.height /= 2;
		rightLeftCornerRegionDebugImage.y += rightLeftCornerRegionDebugImage.height / 2;

		rightRightCornerRegionDebugImage.width -= rightPupil.x;
		rightRightCornerRegionDebugImage.x += rightPupil.x;
		rightRightCornerRegionDebugImage.height /= 2;
		rightRightCornerRegionDebugImage.y += rightRightCornerRegionDebugImage.height / 2;

		rectangle(debugImage, leftEyeRegionDebugImage, cv::Scalar(255, 255, 255));
		rectangle(debugImage, rightEyeRegionDebugImage, cv::Scalar(255, 255, 255));
		/*rectangle(debugImage, leftRightCornerRegionDebugImage, 200);
		rectangle(debugImage, leftLeftCornerRegionDebugImage, 200);
		rectangle(debugImage, rightLeftCornerRegionDebugImage, 200);
		rectangle(debugImage, rightRightCornerRegionDebugImage, 200);*/
		
	}

	/*
	 To show in debugImage
	 */
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
		circle(debugFace, rightPupil, 3, 1234, -1);
		circle(debugFace, leftPupil, 3, 1234, -1);
		imshow(face_window_name, faceROI);
	}

	circle(debugImage, rightPupil + cv::Point(face.x, face.y), 3, cv::Scalar(255, 255, 255), -1);
	circle(debugImage, leftPupil + cv::Point(face.x, face.y), 3, cv::Scalar(255, 255, 255), -1);
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay(cv::Mat frame) {
	std::vector<cv::Rect> faces;


	std::vector<cv::Mat> rgbChannels(3);
	cv::split(frame, rgbChannels);
	cv::Mat frame_gray = rgbChannels[2];

	// Change to dlib's image format. No memory is copied.

	dlib::cv_image<dlib::bgr_pixel> cimg(frame);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150));

	// Find the pose of each face.

	for (unsigned long i = 0; i < faces.size(); ++i)
	{

		dlib::rectangle r(
			(long)(faces[i].x),
			(long)(faces[i].y),
			(long)(faces[i].x + faces[i].width),
			(long)(faces[i].y + faces[i].height)
		);
		dlib::full_object_detection shape = pose_model(cimg, r);


		render_face(debugImage, shape);

	}

	//-- Show what you got
	if (faces.size() > 0) {
		findEyes(frame_gray, faces[0]);
	}
}
