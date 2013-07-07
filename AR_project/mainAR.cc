# include "mainAR.h"
# include <vector>      // for vectors
# include <stdio.h>     // for basic cout
# include "math.h"      // for arctan
# include <algorithm>   // for sort
//# include <Magick++.h>   // for zbar: QR code scanning
# include <zbar.h>       // for zbar

using namespace cv;
using namespace zbar;

// starting button states
bool centroidButtonState_   = false;
bool maskButtonState_       = false;
bool cannyButtonState_      = false;
bool houghButtonState_      = false;
bool clusterButtonState_    = false;
bool cornersButtonState_    = false;
bool perspectiveButtonState_= false;
bool inverseButtonState_    = false;

// starting threshold values
int cannyThres1_    = 0;
int cannyThresh2_   = 95;
int houghThresh_    = 80;


/** =============================================================================
    description: main program
    input: none
    returns: int
**/
int main (){
    // ====================== SET UP ==========================
    // setup timing parameters
    int displayPeriod   = 3;                           // number of 10 msec cycles
    int mainPeriod      = 1;                            // number of 10 msec cycles
    int trackPeriod     = 3;                           // number of 10 msec cycles
    int overlayPeriod   = 20;
    int resetCounter    = displayPeriod * mainPeriod * trackPeriod;
    int counter = 0;

    // setup wecam
    VideoCapture myVideoCapture = setupWebcam();            // setup webcam
    if (!myVideoCapture.isOpened()){                        // confirm video capture is open
        std::cout << "===== ERROR: failed to setup webcam\n";
        return 0;
    }

    // setup main window
    namedWindow("videoWindow", CV_WINDOW_AUTOSIZE);   // create a window
    moveWindow("videoWindow", 10, 10);                // move window to top left corner


    // setup buttons, initial state of all buttons is off
    createButton("show centroid and orientation",   callBackCentroidButton, NULL, CV_CHECKBOX, 0);
    createButton("show masked image",               callBackMaskButton,     NULL, CV_CHECKBOX, 0);
    createButton("show Canny edges",                callBackCannyButton,    NULL, CV_CHECKBOX, 0);
    createButton("show Hough lines",                callBackHoughButton,    NULL, CV_CHECKBOX, 0);
    createButton("show clustered lines",            callBackClusterButton,  NULL, CV_CHECKBOX, 0);
    createButton("show corners",                    callBackCornersButton,  NULL, CV_CHECKBOX, 0);
    createButton("show perspective transformation", callBackPerspectiveButton, NULL, CV_CHECKBOX, 0);
    createButton("show inverse perspective image", callBackInverseButton,   NULL, CV_CHECKBOX, 0);

    // setup zBar reader
    ImageScanner myScanner;     //setup reader
    myScanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);     // configure the reader

    Mat overlayImage;                                           // to store loaded overlay image


    // ====================== MAIN LOOP ==========================
    while(true){

        // ------- get a frame -------
        Mat myImage;
        myVideoCapture >> myImage;

        // ------- track object -------
        bool trackSuccess = false;
        Mat maskedImage; // setup masked image
        if(counter % trackPeriod == 0){
            std::cout << "===================\n";
            std::cout << "\ndo tracking:\n"; // do tracking
            Mat trackedImage = trackObject(myImage);            // track image
            maskedImage = createMask(myImage.clone(), trackedImage);    // create mask from tracked image and copy of myImage

            // confirm successful tracking and masking
            if (maskedImage.data){
                trackSuccess = true;
            }
        }

        // ------- detect lines -------
        bool detectSuccess = false;
        std::vector< Vec2f> lines;
        if(trackSuccess){
            lines = lineDetection(maskedImage, cannyThres1_, cannyThresh2_, houghThresh_);

            // confirm successful line detection
            if (lines.size() > 4){
                detectSuccess = true;
            }
        }

        // ------- do clustering on lines -------
        bool clusterSuccess = false;
        std::vector<Vec2f> clusteredLines;
        if (lines.size()>0){
            clusteredLines = clusterLines(lines, myImage);

            // confirm clustering worked
            if (clusteredLines.size() > 2){ // next step will be find corners. no reason to find corners if less than two lines
                clusterSuccess = true;
            }
        }

        // ------- do corner estimation from lines -------
        bool intersectionSuccess = false;
        std::vector<Vec2f> orderedPoints;
        if (clusterSuccess){
            orderedPoints = computeCorners(clusteredLines, myImage); // find all intersection points

            if (orderedPoints.size() == 4){
                intersectionSuccess = true; // update success handle
            }
        }

        // ------- do perspective transformation -------
        // right now only do perspective if I have exactly 4 points
        bool transformationSuccess = false;
        Mat correctedImage;
        Mat warpMatrix (3, 4, CV_32FC1);
        if (intersectionSuccess){
             correctedImage = doTransformation(orderedPoints, myImage, warpMatrix);

             // confirm successful transformation
             if (correctedImage.cols > 0 && correctedImage.rows > 0){     // if corrected inage
                transformationSuccess = true;
             }
        }

        // ------- read QR code from corrected image -------
        std::string filename;                                           // to store filename from QR code
        bool readQRcodeSuccess = false;
        if (transformationSuccess){                                     // if transform succeeded
            std::cout << "\ndoing readQRCode:\n";
            filename = readQRCode(correctedImage, myScanner);

            // confirm succesful QR read
            if (filename.size() > 0) {
                readQRcodeSuccess = true;
            }
        }


        // ------- load file to display -------
        bool loadFileSuccess = false;
        if (readQRcodeSuccess && (overlayPeriod %10 == 0)){
            std::cout << "\nloading overlay image:\n";
            overlayImage =  loadDisplayImage(filename);

            if(overlayImage.cols > 0 && overlayImage.rows > 0){
                 loadFileSuccess = true;
            }
        }

        // ------- transform overlay down to myImage perspective and merge -------
        Mat perspectiveOverlay(myImage.rows, myImage.cols, CV_8UC4, Scalar(0,0,0,0)); // to store inverse warped overlay
        if (loadFileSuccess){                                                       // if overlay was loaded orrectly
            std::cout << "\ndoing transformation:\n";
            doReverseTransformation(overlayImage, warpMatrix, perspectiveOverlay);  // do reverse transform
            myImage = myImage + perspectiveOverlay;                                 // merge warped overlay with myImage
        }

        /** ----- general format -----------:
            declare any variables
            bool currentTaskSuccess = false;

            if (previousTaskSuccess){
                std::cout << "\ndoing current task:\n";
                output = doTask(inputs);

                // confirm successful task
                if (whatever) {
                    currentTaskSuccess = true;
                }
            }
          **/


        // ------- update display -------
        if(counter % displayPeriod == 0){
            if(!displayFrame(myImage)){
                std::cout << "===== ERROR: can't display frame\n";
            }
        }

        // ------- handle main( ) timing and break out -------
        int key = waitKey(10 * mainPeriod); // convert frequency to period
        if (key == 27){ // ESC key
            break;
        }

        // ------- handle counter -------
        counter++; // increment counter
        if (counter > resetCounter){
            counter = 0;
        }
    }

    // ====================== CLEANUP ==========================
    myVideoCapture.release();       // release the capture source
    destroyWindow("videoWindow");   // destroy the video window(s)
//    destroyWindow("inverse perspective window");

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
    input: Mat myImage (in RGB space)
    returns: Mat image with centroid (in binary space)
**/
Mat trackObject(Mat myImage){
    // create copy of myImage and convert to HSV space
    Mat hsvImage;
    hsvImage.create(myImage.rows, myImage.cols, myImage.type());
    cvtColor(myImage, hsvImage, CV_BGR2HSV, 0); // convert from BGR to HSV, keep same number of channels (0)

    // do thresholding
    Mat thresholdImage;
    thresholdImage.create(myImage.rows, myImage.cols, CV_8U); // force to be 8bit unsigned, single channel
    Scalar lowerBound(0,   0,      180); // hue, saturation, intensity
    Scalar upperBound(255, 255,    255);
    inRange(hsvImage, lowerBound, upperBound, thresholdImage);

    // do closing; I need to do closing to remove any text on page
    Mat closedImage;
    closedImage.create(myImage.rows, myImage.cols, CV_8U);                      // force to be 8bit unsigned, single channel
    Size size(4,4);                                                             // set kernel size, may need to increase for large QR codes
    Mat closingKernel = getStructuringElement(MORPH_RECT, size, Point(-1,-1));  // create kernel
    int closingIterations = 3;
    morphologyEx(thresholdImage, closedImage, MORPH_CLOSE, closingKernel, Point(-1,-1), closingIterations, BORDER_CONSTANT, morphologyDefaultBorderValue()); // basically use default parameters

    // compute centroid and blob orientation and draw
    Mat centroidImage = computeCentroidAndOrientation(closedImage);     // compute centroid and orientation

    return closedImage;
}


/** =============================================================================
    description: computes the centroid and orientation of the single-channel image
    input: Mat (1 channel) image
    returns: Mat (3 channel) for overlay onto myImage
**/
Mat computeCentroidAndOrientation(Mat inputImage){

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

    // std::cout << "centroid :" << xBar << ", " << yBar << "\n";

    // compute orientation from moments (see reference paper)
    double mu20 = (m20 / m00) - ((m10 * m10) / (m00));
    double mu02 = (m02 / m00) - ((m01 * m01) / (m00));
    double mu11 = (m11 / m00) - ((m10 * m01) / (m00));
    double theta = 0.5 * atan((2 * mu11) / (mu20 - mu02)); // radians

    // std::cout << "theta = " << theta << "\n";

    // create image to hold  centroid and orientation overlay drawing
    Mat centroidOverlayImage(inputImage.rows, inputImage.cols, CV_8UC3, Scalar(0,0,0)); // force to be 8bit unsigned, 3 channel (to match original image

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
        line(centroidOverlayImage, topPoint, bottomPoint, Scalar(0,0,255), 5); // horizontal line
        line(centroidOverlayImage, leftPoint, rightPoint, Scalar(0,0,255), 5); // vertical line*/
    }

    if(centroidButtonState_){    // if button state is on, merge images for display
        imshow("centroid image", centroidOverlayImage);
    }

    return centroidOverlayImage;
}

/** =============================================================================
    description: creates mask from
    intput: inputImage 3 channel
    output: masked image 3 channel
**/
Mat createMask(Mat inputImage, Mat trackedImage){

    Mat maskedImage; // setup masked image
    maskedImage.create(inputImage.rows, inputImage.cols, CV_8UC3);

    // create 3channel image where trackedImage is in each channel
    Mat trackedImage_C3(inputImage.rows, inputImage.cols, inputImage.type());

    // creat temp vector of Mat to store replicated matrix
    std::vector<Mat> temp;
    temp.push_back(trackedImage);
    temp.push_back(trackedImage);
    temp.push_back(trackedImage);
    merge(temp, trackedImage_C3);

    maskedImage = inputImage.mul(trackedImage_C3); // elementwise multiplication; since tracked image is binary, multiplying input image by either 1 or 0, it preserves the ROI

    if (maskButtonState_){imshow("masked image", maskedImage);} // if button pressed show image
    return maskedImage;
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


/** =============================================================================
    description: line detection using Canny edge detection and Hough Lines
    input: Mat, image 8bit unsigned, single channel, two threshold parameters
    returns: vector of Vec2f; lines in [rho, theta] format
**/
std::vector< Vec2f> lineDetection(Mat inputImage, int cannyThresh1, int cannyThresh2, int houghThresh){
    std::cout << "\ndo canny with threshold [" << cannyThresh1 << " , " << cannyThresh2 << "]\n";

    // make sure thresholds are positive values
    if (cannyThresh1 < 0){cannyThresh1 = 0;}
    if (cannyThresh2 < 0){cannyThresh2 = 0;}
    if (houghThresh < 1){houghThresh = 1;}

    // ------------- get Canny edges
    // convert myImage to greyscale
    Mat greyscale;
    cvtColor(inputImage, greyscale, CV_BGR2GRAY);

    // do gaussian blur to get rid of noise
    Mat blurImage;
    blur(greyscale, blurImage, Size(3,3) );

    // do Canny edge detection
    Mat edgesImage;
    int kernelSize = 3; // I'm looking for crisp thin edges, so use small kernel

    Canny(blurImage, edgesImage, cannyThresh1, cannyThresh2, kernelSize);

    // visualize Canny edges
    if(cannyButtonState_){imshow("canny edge detection", edgesImage);} // if button pressed, show canny edges
    // -------------- done with Canny edges


    std::cout << "\ndo hough lines with threshold [" << houghThresh << "]\n";

    // find Hough lines; because the edge detection is really crisp, we can jack up the Hough threshold
    double rho      = 2;
    double theta    = CV_PI/90;         // may need to rename this variable
    std::vector<Vec2f> lines;         // pre allocate space
    HoughLines(edgesImage, lines, rho, theta, houghThresh, 0, 0);

    std::cout << "\t computed (" << lines.size() << ") hough lines\n";

    // display Hough lines
    if (houghButtonState_){
        for (int i = 0; i < lines.size(); i++){
            float rho = lines[i][0];
            float theta = lines[i][1];
            Vec4f lineEq = rhoTheta2XY(rho, theta); // convert rho, theta to x1, y1, x2, y2; just for plotting
            line(inputImage, Point(round(lineEq(0)), round(lineEq(1))), Point(round(lineEq(2)), round(lineEq(3))), Scalar(0,255,0), 2, 8);
        }

        imshow("hough lines", inputImage);

        // print Hough lines to command line
        for (int i = 0; i < lines.size(); i++){
            std::cout << "\t[" << lines[i][0] << ",\t " << lines[i][1] << "]\n";
        }
    }
    return lines;
}


/** =============================================================================
    description: does clustering of all Hough lines in rho, theta space
    input: lines and image to overlay
    returns: new equations of lines for centers clustering
**/
std::vector<Vec2f> clusterLines(std::vector<Vec2f> lines, Mat myImage){
    std::cout << "\ndo k-means clustering:\n";

    // convert vector<Vec2f> to Mat
    Mat dataPoints(lines.size(), 2, CV_32F);
    for (int i = 0; i < lines.size(); i++){ // iterate through lines
        dataPoints.at<float>(i,0) = lines[i][0];
        dataPoints.at<float>(i,1) = lines[i][1];
    }

    // tuning parameters; move to global parameters???
    int K = 4;                  // cluster into K bins
    if (lines.size() < K){      // if number of lines is less than number of attempted clusters...
        K = lines.size();       // ... reduce number of clusters
    }
    int attempts = 6;           // do k means 5 times and pick best one
    int maxIterations = 100;    // each attemp iterate 100 times
    int epsilon = 2;            // AND with epsilon < 2 pixels

    // output Mat's
    Mat bestLabels;
    Mat centers;

    // do kMeans
    kmeans(dataPoints, K, bestLabels, TermCriteria(CV_TERMCRIT_ITER, maxIterations, epsilon), attempts , KMEANS_RANDOM_CENTERS, centers );

    // if button pressed, print Hough lines and labels
    if(clusterButtonState_){
        std::cout << "\t hough lines with cluster labels:\n";
        for (int i = 0; i < lines.size(); i++){
            std::cout << "\t[" << lines[i][0] << ",\t " << lines[i][1] << "]\t with label " << bestLabels.at<int>(i, 1) <<  "\n";
        }
    }

    // convert Mat back to vector<Vec2f>
    std::vector<Vec2f> clusteredLines;
    std::cout << "\t clustered lines(rho, theta):\n";
    Mat plottableImage = myImage.clone(); // create clone just for plotting

    for (int i = 0; i < centers.rows; i++){
        Vec2f center;
        center[0] = centers.at<float>(i,0);
        center[1] = centers.at<float>(i,1);
        clusteredLines.push_back(center);

        // convert rho, theta to x1, y1, x2, y2; just for plotting
        float rho = center[0];
        float theta = center[1];
        std::cout << "\t[" << rho << ",\t " << theta << "]\n";

        Vec4f lineEq = rhoTheta2XY(rho, theta);

        // if button pressed add single line to plot
        if (clusterButtonState_){
            line(plottableImage, Point(round(lineEq(0)), round(lineEq(1))), Point(round(lineEq(2)), round(lineEq(3))), Scalar(0,0,255), 2, 8);
        }
    }

    if (clusterButtonState_){imshow("clustered lines", plottableImage);} // if button pressed show lines

    return clusteredLines;
}


/** =============================================================================
    description: tries to find corners by computing intersections of lines
    input: vector of lines in rho/theta format
    returns: void
**/
std::vector <Vec2f> computeCorners(std::vector<Vec2f> clusteredLines, Mat inputImage){
    std::cout << "\ncomputing corners:\n";
    Mat plotableImage = inputImage.clone();    // strictly for visualizing

    // convert all lines from rho/theta into slope intercept
    std::vector< Vec2f> lines;
    for (int i = 0; i < clusteredLines.size(); i++){
        // convert all lines to slope intercept form (y = m*x + b)
        Vec2f line = rhoTheta2SlopeIntercept(clusteredLines[i][0], clusteredLines[i][1]);
        lines.push_back(line);
    }

    // compute intersection point of each line with each other line
    std::vector <Vec2f> intersectionPoints;
    int k = 0;
    for (int i = 0; i < clusteredLines.size(); i++){
        for (int j = i+1; j < clusteredLines.size(); j++){
            float x = (lines[j][1] - lines[i][1]) / (lines[i][0] - lines[j][0]); // x = (b2 - b1)/(m1 - m2)
            float y = lines[i][0] * x  + lines[i][1]; // y = m1*x + b1

            k++;

            int xInt = round(x); // conver floats to pixel values
            int yInt = round(y);

            if ((xInt >= 0) && (xInt <= 640) && (yInt >= 0) && (yInt <= 480)){ // if intersection is inside frame
                // add to list of intersection points
                Vec2f temp;
                temp[0] = xInt;
                temp[1] = yInt;
                intersectionPoints.push_back(temp);

                // and plot
                if(cornersButtonState_){circle(plotableImage, Point(xInt, yInt), 5, Scalar(255,0,0), 2, 8);}
            }
        }
    }

    // if found 4 corners, put in order clockwise around centroid
    std::vector<Vec2f> orderedPoints;
    if (intersectionPoints.size() == 4) {
        orderedPoints = putPointsInOrder(intersectionPoints);

        // if button turned on, draw polygon
        if(cornersButtonState_){
            std::cout << "\t ordered intersection points:\n";

            for (int i = 0; i < orderedPoints.size()-1; i++){
                std::cout << "\t [" << orderedPoints[i][0] << ",\t " << orderedPoints[i][1] << "\t]\n";
                line(plotableImage, Point(round(orderedPoints[i][0]), round(orderedPoints[i][1])), Point(round(orderedPoints[i+1][0]), round(orderedPoints[i+1][1])), Scalar(255,255,0), 2, 8);
            }
            line(plotableImage, Point(round(orderedPoints[0][0]), round(orderedPoints[0][1])), Point(round(orderedPoints[intersectionPoints.size()-1][0]), round(orderedPoints[intersectionPoints.size()-1][1])), Scalar(255,255,0), 2, 8); // closing line
            std::cout << "\t [" << orderedPoints[3][0] << ",\t " << orderedPoints[3][1] << "\t]\n";

            imshow("corners", plotableImage);
        }
    }
    return orderedPoints;
}


/** =============================================================================
    description: performs perspective transformation of caputured image to rectangle
            I am doing an perspective projection FROM "distorted image" (aka trapezoid shape)
            TO "corrected image" (aka rectangluar 8.5 x 11 image)
    input: vector of Vec2f points in distorted image and distorted image
    returns: Mat of corrected image
  **/
Mat doTransformation(std::vector<Vec2f> inputPoints, Mat inputImage, Mat& warpMatrix){
    std::cout << "\nperforming perspective transformation:\n";

    // declare and initialize variables
    int numPoints = inputPoints.size();

    Point2f distortedPoints[numPoints];
    Point2f correctedPoints[numPoints];

    // put input points from vector<Vec2f> into array of Point2f
    for (int i = 0; i < numPoints; i++){
        distortedPoints[i] = Point2f(inputPoints[i][0], inputPoints[i][1]);
    }

    // set output points ; there is a lot going wrong here.
    // I know the points are in order around the perimeter of the blob but I don't know which one is "top left", "top right" etc.
    // Also, I'm assuming that I'm looking at an 8.5 x 11 sheet of paper so, try to fit that in 640 x 480 w/out cropping
    int shortSide = 479;
    int longSide =round(shortSide * 11 / 8.5);

    std::cout << "\t putting transformed image into image of size [" << longSide << ", " << shortSide << "]\n";
    correctedPoints[0] = Point2f(0,         0);
    correctedPoints[1] = Point2f(shortSide, 0);
    correctedPoints[2] = Point2f(shortSide, longSide);
    correctedPoints[3] = Point2f(0,         longSide);

    Mat outputImage = Mat::zeros(longSide, shortSide, inputImage.type());

    warpMatrix = getPerspectiveTransform(distortedPoints, correctedPoints);

    // print warp matrix to command line
    std::cout << "\t warpMatrix = \n";
    for (int i = 0; i < 3; i++){
        std::cout << "\t| " << warpMatrix.at<float>(i,0) << ",\t " << warpMatrix.at<float>(i,1) << ",\t " << warpMatrix.at<float>(i, 2) << ",\t " << warpMatrix.at<float>(i,3) << "\t|\n";
    }

    warpPerspective(inputImage, outputImage, warpMatrix, outputImage.size());

    if(perspectiveButtonState_){imshow("perspective", outputImage);}

    return outputImage;
}

/** =============================================================================
code for this was pulled from https://github.com/rportugal/opencv-zbar
**/
std::string  readQRCode(Mat inputImage, ImageScanner& myScanner){
    std::cout << "\nread QR codes:\n";

    Mat myGrayscaleImage;
    cvtColor(inputImage, myGrayscaleImage, CV_BGR2GRAY);

    // Obtain image data
    int width = myGrayscaleImage.cols;
    int height = myGrayscaleImage.rows;
    uchar *raw = (uchar *)(myGrayscaleImage.data);

    // Wrap image data
    Image image(width, height, "Y800", raw, width * height);

    // Scan the image for barcodes
    myScanner.scan(image);

    // Extract results
    std::string outputString;
    int counter = 0;
    for (Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol){
        outputString = symbol->get_data();
    }

    std::cout << "\t " << outputString <<"\n";
    return outputString;
}


/** =============================================================================

**/
void doReverseTransformation(Mat overlayImage, Mat warpMatrix, Mat& perspectiveOverlay){
    std::cout << "\ndo reverse transformation:\n";

    warpPerspective(overlayImage,
                    perspectiveOverlay,
                    warpMatrix,
                    Size(perspectiveOverlay.cols, perspectiveOverlay.rows),
                    WARP_INVERSE_MAP);
//                    BORDER_TRANSPARENT);

    if(inverseButtonState_){imshow("inverse perspective window", perspectiveOverlay);} // if button pressed, display
}
