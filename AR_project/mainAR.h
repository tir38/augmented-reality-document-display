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
//# include <Magick++.h>   // for zbar: QR code scanning
# include <zbar.h>       // for zbar


using namespace cv;
using namespace zbar;

int main();

VideoCapture setupWebcam();

bool displayFrame(Mat image);

Mat trackObject(Mat myImage);

Mat computeCentroidAndOrientation(Mat inputImage);

Mat createMask(Mat inputImage, Mat trackedImage);

std::vector< std::vector< Point> > computeContours(Mat myImage);

std::vector< Vec2f> lineDetection(Mat inputImage, int cannyThresh1, int cannyThresh2, int houghThresh);

std::vector<Vec2f> clusterLines(std::vector<Vec2f> lines, Mat myImage);

Mat doTransformation(std::vector<Vec2f> inputPoints, Mat inputImage, Mat& warpMatrix);

void zBarTest();

std::vector<Vec2f> computeCorners(std::vector<Vec2f> clusteredLines, Mat inputImage);

std::string readQRCode(Mat inputImage, ImageScanner& myScanner);

void doReverseTransformation(Mat overlayImage, Mat warpMatrix, Mat& perspectiveOverlay);


// global variables
extern bool centroidButtonState_;
extern bool maskButtonState_;
extern bool cannyButtonState_;
extern bool houghButtonState_;
extern bool clusterButtonState_;
extern bool cornersButtonState_;
extern bool perspectiveButtonState_;
extern bool inverseButtonState_;

extern int cannyThres1_;
extern int cannyThresh2_;
extern int houghThresh_;

// helper methods
Vec4f rhoTheta2XY(float rho, float theta);

Vec2f rhoTheta2SlopeIntercept(float rho, float theta);

std::vector<Vec2f> putPointsInOrder(std::vector<Vec2f> intersectionPoints);

Mat loadDisplayImage(std::string filename);

Mat doOverlay(Mat backgroundImage, Mat foregroundImage);


// GUI methods
void callBackCentroidButton(int state, void* pointer);
void callBackMaskButton(int state, void* pointer);
void callBackCannyButton(int state, void* pointer);
void callBackHoughButton(int state, void* pointer);
void callBackClusterButton(int state, void* pointer);
void callBackCornersButton(int state, void* pointer);
void callBackPerspectiveButton(int state, void* pointer);
void callBackInverseButton(int state, void* pointer);


#endif // MAINAR_H
