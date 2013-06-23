
# include "mainAR.h"
# include <vector>
# include <stdio.h>
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

    // calculate the moments of the thresholded image
    CvMoments* momentsOfThreshold = (CvMoments*)malloc(sizeof(CvMoments));
    cvMoments(thresholdImage, momentsOfThreshold, 1);

    // pull pertiment moments from all moments
    double moment10 = cvGetSpatialMoment(momentsOfThreshold, 1, 0); //
    double moment01 = cvGetSpatialMoment(momentsOfThreshold, 0, 1); //
    double moment00 = cvGetCentralMoment(momentsOfThreshold, 0, 0); // area

    // compute centroid (xBar, yBar) from moments
    double xBar = moment10 / moment00;
    double yBar = moment01 / moment00;
    int xBari = (int)xBar;
    int yBari = (int)yBar;

    // compute orientation from moments


    // create image to hold overlay drawing
    IplImage* overlayImage = cvCreateImage(cvGetSize(myImage), myImage->depth, 3);

    // draw + shape at centroid
    if ((xBari > 50) && (xBari < 550) && (yBari > 50) && (yBari<430)){
        cvLine(overlayImage, cvPoint(xBari-50, yBari), cvPoint(xBari+50, yBari), cvScalar(0,0,255), 5); // horizontal line
        cvLine(overlayImage, cvPoint(xBari, yBari-50), cvPoint(xBari, yBari+50), cvScalar(0,0,255), 5); // vertical line*/

        // combine my image with overlay image
        cvAdd(myImage, overlayImage, myImage);
    }

    // temp display results
    cvShowImage("testWindow1", thresholdImage );   // show the image
    return true;
}
