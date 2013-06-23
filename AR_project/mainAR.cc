
# include "mainAR.h"
# include <vector>
# include <stdio.h>
# include "math.h" // for arctan
using namespace cv;
/** =============================================================================
    description: main program
    input: none
    returns: int
**/
int main (){
    // ====================== SET UP ==========================
    int displayPeriod   = 3;                           // number of 10 msec cycles
    int mainPeriod      = 1;                            // number of 10 msec cycles
    int trackPeriod     = 3;                           // number of 10 msec cycles
    int resetCounter    = displayPeriod * mainPeriod * trackPeriod;
    int counter = 0;

    VideoCapture myVideoCapture = setupWebcam();            // setup webcam
    if (!myVideoCapture.isOpened()){                        // confirm video capture is open
        std::cout << "===== ERROR: failed to setup webcam\n";
        return 0;
    }

    // setup windows
    namedWindow("videoWindow", CV_WINDOW_AUTOSIZE);   // create a window
    moveWindow("videoWindow", 10, 10);                // move window to top left corner
    namedWindow("testWindow1", CV_WINDOW_AUTOSIZE);
    moveWindow("testWindow1", 800, 10);


    // ====================== MAIN LOOP ==========================
    while(true){

        // get a frame
        Mat myImage;
        myVideoCapture >> myImage;

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
        int key = waitKey(10 * mainPeriod); // convert frequency to period
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
    myVideoCapture.release();       // release the capture source
    destroyWindow("videoWindow");   // destroy the video window

    int dummy;
    return dummy;
}


/** =============================================================================
    description: sets up webcam, video capture, and display window
    input: none
    returns: VideoCapture object
**/
VideoCapture setupWebcam(){
    int myDeviceID = 1;                                 // device 1 is my USB webcam
    VideoCapture myVideoCapture(myDeviceID);            // declare and initialize video capture
    return myVideoCapture;
}

/** =============================================================================
    description: displays single frame to window
    input: Mat
    returns: true if successful
**/
bool displayFrame(Mat image){
    imshow("videoWindow", image); // show image
    return true;
}

/** =============================================================================
    description: tracks white blob in image and updates display image
    input: reference to pointer to IplImage
    returns: true if successful
**/
bool trackObject(Mat myImage){
    // create copy of myImage and convert to HSV space
    Mat hsvImage;
    hsvImage.create(myImage.rows, myImage.cols, myImage.type());
    cvtColor(myImage, hsvImage, CV_BGR2HSV, 0); // convert from BGR to HSV, keep same number of channels (0)

    // create image to hold threshold
    Mat thresholdImage;
    thresholdImage.create(myImage.rows, myImage.cols, CV_8U); // force to be 8bit unsigned, single channel

    // do thresholding
    Scalar lowerBound(0,   0,      200); // hue, saturation, intensity
    Scalar upperBound(255, 255,    255);
    inRange(hsvImage, lowerBound, upperBound, thresholdImage);

    // do closing
    Mat closedImage;
    closedImage.create(myImage.rows, myImage.cols, CV_8U);                      // force to be 8bit unsigned, single channel
    Size size(3,3);                                                             // create kernel
    Mat closingKernel = getStructuringElement(MORPH_RECT, size, Point(-1,-1));
    int closingIterations = 2;
    morphologyEx(thresholdImage, closedImage, MORPH_CLOSE, closingKernel, Point(-1,-1), closingIterations, BORDER_CONSTANT, morphologyDefaultBorderValue()); // basically use default parameters

    // compute centroid and blob orientation and draw
    Mat centroidImage = computeCentroidAndOrientation(closedImage);     // compute centroid and orientation
    myImage = centroidImage + myImage;                                  // merge images

    // compute contours
    std::vector< std::vector< Point> > contours = computeContours(closedImage); // update myImage
    drawContours(myImage, contours, -1, Scalar(0,0,255), 2, 8); // draw all contours (-1), red, linethickness =2, 8bit

////    std::cout << pointerToContours->total << " number of contours\n";
////    // find Hough lines
////    void* linesPointer;
////    double rho      = 0.5;
////    double theta    = 0.5; // may need to rename this variable
////    int threshold   = 1;
////    cvHoughLines2(closedImage, linesPointer, CV_HOUGH_STANDARD, rho, theta, threshold);

////    // fit lines to contours
////    Vec4f lines;
////    std::cout << "about to do cvFitLine()\n";
////    fitLine(Mat(pointerToContours[0]),lines,CV_DIST_L2, 0, 0.01, 0.01,); // first contour

    // temp display results
    imshow("testWindow1", closedImage );   // show the image
    return true;
}

/** =============================================================================
    description: computes the centroid and orientation of the single-channel image
    input: Mat (1 channel) image
    returns: Mat (3 channel) for overlay onto myImage
**/
Mat computeCentroidAndOrientation(Mat inputImage){

    std::cout << "===================\n";
    // calculate the moments of the thresholded image
    Moments momentsOfThreshold = moments(inputImage, false);

    // pull pertiment moments from all moments
    double m10 = momentsOfThreshold.m10; // first moment (X)
    double m01 = momentsOfThreshold.m01; // first moment (Y)
    double m00 = momentsOfThreshold.m00; // zero moment (i.e. area)
    double m11 = momentsOfThreshold.m11; // first cross moment (XY)
    double m20 = momentsOfThreshold.m20; // second moment (X^2)
    double m02 = momentsOfThreshold.m02; // second moment (Y^2)

    // think about adding conditional such that area > someArea or else break

    // compute centroid (xBar, yBar) from moments
    int xBar = (int)(m10 / m00); // typecast to int
    int yBar = (int)(m01 / m00); // typecast to int

    std::cout << "centroid :" << xBar << ", " << yBar << "\n";

    // compute orientation from moments (see reference paper)
    double mu20 = (m20 / m00) - ((m10 * m10) / (m00));
    double mu02 = (m02 / m00) - ((m01 * m01) / (m00));
    double mu11 = (m11 / m00) - ((m10 * m01) / (m00));
    double theta = 0.5 * atan((2 * mu11) / (mu20 - mu02)); // radians

    std::cout << "theta = " << theta << "\n";

    // create image to hold overlay drawing
    Mat overlayImage;
    overlayImage.create(inputImage.rows, inputImage.cols, CV_8UC3); // force to be 8bit unsigned, 3 channel (to match original image

    // draw + shape at centroid
    // compute points; "top", "bottom", "left", and "right" are just semantic names given to four points
    // do geometric calc and typecast to int all inline (yeah its ugly, i know)
    Point topPoint = Point (((int)(xBar+120*sin(theta))),         ((int)(yBar+120*cos(theta))));
    Point bottomPoint = Point(((int)(xBar-120*sin(theta))),         ((int)(yBar-120*cos(theta))));
    Point leftPoint   = Point(((int)(xBar- 80*sin(theta-1.57))),    ((int)(yBar- 80*cos(theta-1.57))));
    Point rightPoint  = Point(((int)(xBar+ 80*sin(theta-1.57))),    ((int)(yBar+ 80*cos(theta-1.57))));

    // basically check that every point is within 640 x 480 window
    if ((topPoint.x > 0)    && (topPoint.x < 600)       && (topPoint.y > 0)     && (topPoint.y<480)
    && (bottomPoint.x > 0)  && (bottomPoint.x < 600)    && (bottomPoint.y > 0)  && (bottomPoint.y<480)
    && (leftPoint.x > 0)    && (leftPoint.x < 600)      && (leftPoint.y > 0)    && (leftPoint.y<480)
    && (rightPoint.x > 0)   && (rightPoint.x < 600)     && (rightPoint.y > 0)   && (rightPoint.y<480)){

        // draw two lines
        line(overlayImage, topPoint, bottomPoint, Scalar(0,0,255), 5); // horizontal line
        line(overlayImage, leftPoint, rightPoint, Scalar(0,0,255), 5); // vertical line*/
    }
    return overlayImage;
}

/** =============================================================================
    description: computes the contours
    input: Mat, image 8bit unsigned, single channel
    returns: vector of vector of Points
**/
std::vector< std::vector< Point> > computeContours(Mat inputImage){
    // When computing contours input image will be overwritten;
    // So to preserve threshold, create copy called contourImage
    Mat contourImage = inputImage.clone();

    // generate structures for finding contours
    std::vector< std::vector< Point> > contours; // declare vector of vector of points

    // find contours
    findContours(contourImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE); // consider playing with CV_CHAIN_APPROX_NONE (see documentation)
    return contours;
}
