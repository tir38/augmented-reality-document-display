
# include "mainAR.h"
# include <vector>
# include <stdio.h>
# include "math.h" // for arctan
using namespace cv;

int main (){
    // ====================== SET UP ==========================
    int displayPeriod   = 3;                           // number of 10 msec cycles
    int mainPeriod      = 1;                            // number of 10 msec cycles
    int trackPeriod     = 3;                           // number of 10 msec cycles
    int resetCounter    = displayPeriod * mainPeriod * trackPeriod;
    int counter = 0;

    CvCapture* myCapture;                               // declare capture

    if (!setupWebcam(myCapture)){                       // setup webcam
        std::cout << "===== ERROR: failed to setup webcam";
    }

    // setup windows
    cvNamedWindow("videoWindow", CV_WINDOW_AUTOSIZE);   // create a window
    cvMoveWindow("videoWindow", 10, 10);                // move window to top left corner
    cvNamedWindow("testWindow1", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("testWindow1", 800, 10);


    // ====================== MAIN LOOP ==========================
    while(true){

        // get a frame
        IplImage* myImage;                      // create image
        myImage = cvQueryFrame(myCapture);      // "query" an image (i.e. get and decode)

        // do frame processing
        if(counter % trackPeriod == 0){
            if(!trackObject(myImage)){
                std::cout << "===== ERROR: trackObject failed\n";
            }
        }

        // update display
        if(counter % displayPeriod == 0){
            if(!displayFrame(myImage)){
                std::cout << "===== ERROR: can't display frame\n";
            }
        }

        // handle main( ) timing and break out
        int key = cvWaitKey(10 * mainPeriod); // convert frequency to period
        if (key == 27){ // ESC key
            break;
        }

        // handle counter
        counter++; // increment counter
        if (counter > resetCounter){
            counter = 0;
        }

    }

    // ====================== CLEANUP ==========================
    cvReleaseCapture(&myCapture);       // release the capture source
    cvDestroyWindow("videoWindow");     // destroy the video window

    int dummy;
    return dummy;
};


/**
    description: sets up webcam, video capture, and display window
    input: reference to pointer to CvCapture
    returns: true if successful
**/
bool setupWebcam(CvCapture*& myCapture){
    int myDeviceID = 1;                             // device 1 is my USB webcam
    myCapture = cvCreateCameraCapture(myDeviceID);  // initialize CvCapture* on device
    if (myCapture == NULL){                         // if fails will return Null pointer
        return false;
    }
    return true;
};

/**
    description: displays single frame
    input: reference to pointer to IplImage
    returns: true if successful
**/
bool displayFrame(IplImage*& myImage){
    cvShowImage("videoWindow", myImage );   // show the image
    return true;
}

/**
    description: tracks white blob in image and updates display image
    input: reference to pointer to IplImage
    returns: true if successful
**/
bool trackObject(IplImage*& myImage){
    // create copy of myImage and convert to HSV space
    IplImage* hsvImage = cvCreateImage(cvGetSize(myImage), myImage->depth, 3);  // create empty image of correct format; assume 3 channels (RGB)
    cvCvtColor(myImage, hsvImage, CV_BGR2HSV);                                  // convert from BGR to HSV

    // create image to hold intensity threshold
    IplImage* thresholdImage = cvCreateImage(cvGetSize(myImage), myImage->depth, 1); // single channel: threshold intensity

    // do thresholding
    CvScalar lowerBound = cvScalar(0,   0,      200); // hue, saturation, intensity
    CvScalar upperBound = cvScalar(255, 255,    255);
    cvInRangeS(hsvImage, lowerBound, upperBound, thresholdImage);

    // compute centroid and blob orientation and draw
    IplImage* centroidImage;                                        // declare image to hold centroid drawing
    centroidImage = computeCentroidAndOrientation(thresholdImage);  // compute centroid and orientation
    cvAdd(myImage, centroidImage, myImage);                         // display results

    // compute contours and draw them
        // When computing contours input image will be overwritten;
        // So to preserve threshold, create copy called contourImage
    IplImage* contourImage = cvCreateImage(cvGetSize(myImage), myImage->depth, 1);
    cvCopy(thresholdImage, contourImage);

    // generate structures for finding contours
    CvMemStorage* myStorage = cvCreateMemStorage();
    CvSeq* pointerToContours = NULL; // sequence

    // do find contours
    cvFindContours(contourImage,
                   myStorage,
                   &pointerToContours,
                   sizeof(CvContour),
                   CV_RETR_LIST);

    // draw the contours on myImage
    cvDrawContours(myImage, pointerToContours, cvScalar(0,0,255), cvScalar(0,0,255), 2, 2, 8);
//    cvAdd(thresholdImage, overlayL1Image, thresholdImage);     // combine myImage with overlay image


    // temp display results
    cvShowImage("testWindow1", thresholdImage );   // show the image
    return true;
}

/**
    description: computes the centroid and orientation of the threshold image
    input: reference to pointer to IplImage (1 channel) threshold image
    returns: pointer to IplImage (3 channel) for overlay onto myImage

**/
IplImage* computeCentroidAndOrientation(IplImage*& inputImage){
    // calculate the moments of the thresholded image
    CvMoments* momentsOfThreshold = (CvMoments*)malloc(sizeof(CvMoments));  // allocate space to store moments
    cvMoments(inputImage, momentsOfThreshold, 1);                           // compute moments

    // pull pertiment moments from all moments
    double moment10 = cvGetSpatialMoment(momentsOfThreshold, 1, 0); // first moment (X)
    double moment01 = cvGetSpatialMoment(momentsOfThreshold, 0, 1); // second moment (Y)
    double moment00 = cvGetCentralMoment(momentsOfThreshold, 0, 0); // zero moment (i.e. area)
    double moment11 = cvGetCentralMoment(momentsOfThreshold, 1, 1); // first cross moment (XY)
    double moment20 = cvGetCentralMoment(momentsOfThreshold, 2, 0); // second moment (X^2)
    double moment02 = cvGetCentralMoment(momentsOfThreshold, 0, 2); // second moment (Y^2)

    // think about adding conditional such that area > someArea or else break

    // compute centroid (xBar, yBar) from moments
    int xBar = (int)moment10 / moment00; // typecast to int
    int yBar = (int)moment01 / moment00; // typecast to int

    // compute orientation from moments (see reference paper)
    double mu20 = (moment20 / moment00) - ((moment10 * moment10) / (moment00));
    double mu02 = (moment02 / moment00) - ((moment01 * moment01) / (moment00));
    double mu11 = (moment11 / moment00) - ((moment10 * moment01) / (moment00));
    double theta = 0.5 * atan((2 * mu11) / (mu20 - mu02)); // radians

    std::cout << "theta = " << theta << "\n";

    // create image to hold overlay drawing
    IplImage* overlayImage = cvCreateImage(cvGetSize(inputImage), inputImage->depth, 3);

    // draw + shape at centroid
    // compute points; "top", "bottom", "left", and "right" are just semantic names given to four points
    // do geometric calc and typecast to int all inline (yeah its ugly, i know)
    CvPoint topPoint    = cvPoint(((int)(xBar+120*sin(theta))),         ((int)(yBar+120*cos(theta))));
    CvPoint bottomPoint = cvPoint(((int)(xBar-120*sin(theta))),         ((int)(yBar-120*cos(theta))));
    CvPoint leftPoint   = cvPoint(((int)(xBar- 80*sin(theta-1.57))),    ((int)(yBar- 80*cos(theta-1.57))));
    CvPoint rightPoint  = cvPoint(((int)(xBar+ 80*sin(theta-1.57))),    ((int)(yBar+ 80*cos(theta-1.57))));

    // basically check that every point is within 640 x 480 window
    if ((topPoint.x > 0)    && (topPoint.x < 600)       && (topPoint.y > 0)     && (topPoint.y<480)
    && (bottomPoint.x > 0)  && (bottomPoint.x < 600)    && (bottomPoint.y > 0)  && (bottomPoint.y<480)
    && (leftPoint.x > 0)    && (leftPoint.x < 600)      && (leftPoint.y > 0)    && (leftPoint.y<480)
    && (rightPoint.x > 0)   && (rightPoint.x < 600)     && (rightPoint.y > 0)   && (rightPoint.y<480)){

        // draw two lines
        cvLine(overlayImage, topPoint, bottomPoint, cvScalar(0,0,255), 5); // horizontal line
        cvLine(overlayImage, leftPoint, rightPoint, cvScalar(0,0,255), 5); // vertical line*/

    }
    return overlayImage;

}
