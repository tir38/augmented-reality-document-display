
# include "openCVtest.h"

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

# include "opencv2/opencv.hpp"
# include <vector>
# include <stdio.h>

using namespace cv;

int main (){
    int dummy;
    testCamera();
    return dummy;

};


/*
    description: tests camera functionailty by capturing a single frame and displaying it for a short time
*/
void testCamera(){

    // setup
    CvCapture* myCapture = cvCreateCameraCapture(0); // declare and initialize capture
    IplImage* myImg; // = 0;                        // declare and initialize an image

    // send feed to a window
    // create a window
    cvNamedWindow("videoWindow", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("videoWindow", 10, 10); // move window to top left corner

    int i = 0; // break handler
    int key; // integer to store which key was pressed

    while (i != 1){
        // grab a frame
        if(!cvGrabFrame(myCapture)){   // returns boolean, frame is stored in buffer
          printf("ERROR: Could not grab a frame\n");
          return;
        }

        // do the processing and decoding and return the image
        myImg = cvRetrieveFrame(myCapture);

        // show the image
        cvShowImage("videoWindow", myImg );

        // wait for a key press or 10 msec pause
        key = cvWaitKey(10);

        // if space key pressed, grab single frame
        if (key == 32){ // if space key pressed
            // create a new window to display single frame
            cvNamedWindow("stillFrameWindow", CV_WINDOW_AUTOSIZE);
            cvMoveWindow("stillFrameWindow", 700, 10); // move window to top center of screen

            // show the image
            cvShowImage("stillFrameWindow", myImg );
        }

        // if ESC key pressed, exit
        else if (key == 27){  // if ESC key pressed
           break;
        }
    }

    // release and cleanup ...
    cvReleaseImage(&myImg);             // ... the image
    cvReleaseCapture(&myCapture);       // ... the capture source
    cvDestroyWindow("videoWindow");     // ... the two video windows
    cvDestroyWindow("stillFrameWindow");
};
