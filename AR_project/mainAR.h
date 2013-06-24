#ifndef MAINAR_H
#define MAINAR_H
// just include all of the openCV headers for now
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/legacy/compat.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;

int main();

VideoCapture setupWebcam();

bool displayFrame(Mat image);

Mat trackObject(Mat myImage);

Mat computeCentroidAndOrientation(Mat inputImage);

std::vector< std::vector< Point> > computeContours(Mat myImage);

std::vector< Vec2f> lineDetection(Mat inputImage, int cannyThresh1, int cannyThresh2);

std::vector<Vec2f> clusterLines(std::vector<Vec2f> lines, Mat myImage);

Vec4f rhoTheta2XY(double rho, double theta);

#endif // MAINAR_H
