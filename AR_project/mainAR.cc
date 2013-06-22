
# include "mainAR.h"
# include <vector>
# include <stdio.h>
using namespace cv;

int main (){
    // ====================== SET UP ==========================
    int displayPeriod   = 10;                           // number of 10 msec cycles
    int mainPeriod      = 1;                            // number of 10 msec cycles
    int resetCounter    = displayPeriod * mainPeriod;
    int counter = 0;

    CvCapture* myCapture;                               // declare capture

    if (!setupWebcam(myCapture)){                       // setup webcam
        std::cout << "===== ERROR: failed to setup webcam";
    }

    cvNamedWindow("videoWindow", CV_WINDOW_AUTOSIZE);   // create a window
    cvMoveWindow("videoWindow", 10, 10);                // move window to top left corner

    // ====================== MAIN LOOP ==========================
    while( 1 != 0){

        // do video display
        if(counter % displayPeriod == 0){
            if(!displayFrame(myCapture)){
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
    std::cout  << "===== DEBUG: inside setupWebcam()\n";
    myCapture = cvCreateCameraCapture(0); // initialize CvCapture*
    return true;
};

/**
    description: displays single frame
    input: reference to pointer to CvCapture
    returns: true if successful
**/
bool displayFrame(CvCapture*& myCapture){

    IplImage* myImage;                      // create image
    myImage = cvQueryFrame(myCapture);      // "query" an image (i.e. get and decode)
    cvShowImage("videoWindow", myImage );   // show the image

    return true;
}
