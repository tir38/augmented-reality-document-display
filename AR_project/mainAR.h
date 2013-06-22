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

int main();
bool setupWebcam(CvCapture*& myCapture);
bool displayFrame(CvCapture*& myCapture);

#endif // MAINAR_H
