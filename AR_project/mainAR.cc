
# include "mainAR.h"
# include <vector>
# include <stdio.h>
# include "math.h" // for arctan
# include <algorithm> // for sort
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
    namedWindow("tracked area of interest", CV_WINDOW_AUTOSIZE);
    moveWindow("tracked area of interest", 800, 10);
    namedWindow("canny edge detection", CV_WINDOW_AUTOSIZE);
    moveWindow("canny edge detection", 750,20);

    int cannyThres1 = 0;
    int cannyThresh2 = 95;
    int houghThresh = 80;
    createTrackbar("Canny Low", "canny edge detection", &cannyThres1, 100);
    createTrackbar("Canny High", "canny edge detection", &cannyThresh2, 100);
    createTrackbar("Hough threshold", "canny edge detection", &houghThresh, 100);

    // ====================== MAIN LOOP ==========================
    while(true){

        // get a frame
        Mat myImage;
        myVideoCapture >> myImage;

        // setup masked image
        Mat maskedImage;
        maskedImage.create(myImage.rows, myImage.cols, CV_8UC3);

        // track object
        bool trackSuccess = false;
        if(counter % trackPeriod == 0){
            // do tracking
            Mat trackedImage = trackObject(myImage); // get tracked object (8 bit unsigned, single channel)

            // create 3channel image where trackedImage is in each channel
            Mat trackedImage_C3(myImage.rows, myImage.cols, myImage.type());

            // creat temp vector of Mat to store replicated matrix
            std::vector<Mat> temp;
            temp.push_back(trackedImage);
            temp.push_back(trackedImage);
            temp.push_back(trackedImage);

            std::cout<< "about to merge\n";
            merge(temp, trackedImage_C3);

            maskedImage = myImage.mul(trackedImage_C3); // elementwise multiplication; since tracked image is binary, multiplying input image by either 1 or 0, it preserves the ROI
            trackSuccess = true;
        }

        // detect lines
        bool detectSuccess = false;
        std::vector< Vec2f> lines;
        if(trackSuccess){
            lines = lineDetection(maskedImage, cannyThres1, cannyThresh2, houghThresh);

//            // display Hough lines; move this into lineDetection subfunction
//            for (int i = 0; i < lines.size(); i++){
//                float rho = lines[i][0];
//                float theta = lines[i][1];
//                Vec4f lineEq = rhoTheta2XY(rho, theta); // convert rho, theta to x1, y1, x2, y2; just for plotting
//                line(myImage, Point(round(lineEq(0)), round(lineEq(1))), Point(round(lineEq(2)), round(lineEq(3))), Scalar(0,255,0), 2, 8);
//            }

            std::cout << "number of Hough lines = " << lines.size() << "\n";
            detectSuccess = true;
        }

        // do clustering on lines
        bool clusterSuccess = false;
        std::vector<Vec2f> clusteredLines;
        if (lines.size()>0){
            clusteredLines = clusterLines(lines, myImage);
            std::cout << "number of clustered lines = " << clusteredLines.size() << "\n";
            clusterSuccess = true;
        }

        // do corner estimation from lines; corner is defined as "meaningful" intersection of lines
        if ((clusteredLines.size() >= 2) && (clusterSuccess)){ // obviously no reason to find corners on fewer than two lines
            std::vector<Vec2f> intersectionPoints = computeCorners(clusteredLines, myImage);
            std::vector<Vec2f> orderedPoints = putPointsInOrder(intersectionPoints);

            // draw polygon
            for (int i = 0; i < intersectionPoints.size()-1; i++){
                line(myImage, Point(round(orderedPoints[i][0]), round(orderedPoints[i][1])), Point(round(orderedPoints[i+1][0]), round(orderedPoints[i+1][1])), Scalar(255,255,0), 2, 8);
            }
            line(myImage, Point(round(orderedPoints[0][0]), round(orderedPoints[0][1])), Point(round(orderedPoints[intersectionPoints.size()-1][0]), round(orderedPoints[intersectionPoints.size()-1][1])), Scalar(255,255,0), 2, 8); // closing line
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
    destroyWindow("videoWindow");   // destroy the video window(s)
    destroyWindow("tracked area of interest");
    destroyWindow("canny edge detection");

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
    Size size(2,2);                                                             // create kernel
    Mat closingKernel = getStructuringElement(MORPH_RECT, size, Point(-1,-1));
    int closingIterations = 3;
    morphologyEx(thresholdImage, closedImage, MORPH_CLOSE, closingKernel, Point(-1,-1), closingIterations, BORDER_CONSTANT, morphologyDefaultBorderValue()); // basically use default parameters

    // compute centroid and blob orientation and draw
    Mat centroidImage = computeCentroidAndOrientation(closedImage);     // compute centroid and orientation
//    myImage = centroidImage + myImage;                                  // merge images

    //  AT THE MOMENT I DON'T NEED CONTOURS
//    // compute contours
//    std::vector< std::vector< Point> > contours = computeContours(closedImage); // update myImage
//    std::cout << "there are some contours :"  << contours.size() << "\n";

//    // draw contours
//    for (int i = 0; i < contours.size(); i++){
//        std::vector<Point> singleContour = contours[i];                     // get single contour
//        Scalar randomColor = Scalar(rand()&255, rand()&255, rand()&255);    // generate random colro
//        drawContours(myImage, contours, i, randomColor, 2, 8 );             // draw single contour
//        int numPoints = singleContour.size();                               // get number of points
//        std::cout << "\t contour [" << i << "] has " << numPoints << " points.\n";
//    }

//    Mat justContoursImage;
//    justContoursImage.create(myImage.rows, myImage.cols, CV_8U); // also draw contours on new image
//    drawContours(justContoursImage, contours, -1, Scalar(255), 2, 8);

    // display intermediate results
    imshow("tracked area of interest", closedImage);   // show the image
    return closedImage;
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

/** =============================================================================
    description: line detection using Canny edge detection and Hough Lines
    input: Mat, image 8bit unsigned, single channel, two threshold parameters
    returns: vector of Vec2f; lines in [rho, theta] format
**/
std::vector< Vec2f> lineDetection(Mat inputImage, int cannyThresh1, int cannyThresh2, int houghThresh){

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

    // setup GUI
    std::cout << "about to do canny with thresh [" << cannyThresh1 << " , " << cannyThresh2 << "]\n";
    Canny(blurImage, edgesImage, cannyThresh1, cannyThresh2, kernelSize);

    // visualize Canny edges
    imshow("canny edge detection", edgesImage);
    // -------------- done with Canny edges

    // find Hough lines; because the edge detection is really crisp, we can jack up the Hough threshold
    double rho      = 2;
    double theta    = CV_PI/90;         // may need to rename this variable
    std::vector<Vec2f> lines;         // pre allocate space
    HoughLines(edgesImage, lines, rho, theta, houghThresh, 0, 0);

//    // remove Hough lines with non-positive rho; I DON'T THINK I NEED THIS
//    for (int i = 0; i < lines.size();){ // iterate through all and remove some
//        if (lines[i][0] <= 0) {
//            lines.erase(lines.begin()+i     );
//        }
//        else{
//            i++; // put the iterator here so that I don't skip anything
//        }
//    }

    return lines;
}

/** =============================================================================
    description: does clustering of all Hough lines in rho, theta space
    input: lines and image to overlay
    returns: new equations of lines for centers clustering
**/
std::vector<Vec2f> clusterLines(std::vector<Vec2f> lines, Mat myImage){

    // convert vector<Vec2f> to Mat
    Mat dataPoints(lines.size(), 2, CV_32F);
    for (int i = 0; i < lines.size(); i++){ // iterate through lines
        dataPoints.at<float>(i,0) = lines[i][0];
        dataPoints.at<float>(i,1) = lines[i][1];
    }

    // tuning parameters
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
//    kmeans(dataPoints, K, bestLabels, TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, maxIterations, epsilon), attempts , KMEANS_RANDOM_CENTERS, centers );
    kmeans(dataPoints, K, bestLabels, TermCriteria(CV_TERMCRIT_ITER, maxIterations, epsilon), attempts , KMEANS_RANDOM_CENTERS, centers );

    std::cout << "best labels is of size [" << bestLabels.rows << ", " << bestLabels.cols << "]\n";

    // for debugging:
    // print Hough lines
    for (int i = 0; i < lines.size(); i++){
        std::cout << "\t[" << lines[i][0] << ", " << lines[i][1] << "] with label " << bestLabels.at<int>(i, 1) <<  "\n";
    }

    // plot line clusters
    // convert Mat back to vector<Vec2f> and plot
    std::vector<Vec2f> clusteredLines;
    std::cout << "clusteredLines(rho, theta):\n";
    for (int i = 0; i < centers.rows; i++){
        Vec2f center;
        center[0] = centers.at<float>(i,0);
        center[1] = centers.at<float>(i,1);
        clusteredLines.push_back(center);

        // convert rho, theta to x1, y1, x2, y2; just for plotting
        float rho = center[0];
        float theta = center[1];
        std::cout << "\t[" << rho << ", " << theta << "]\n";

//        if (rho < 0.001 && theta < 0.001){  // if rho and theta are both approx zero, its because that cluster doesn't have any element in it...
//            continue;                       // ... so don't save it.
//        }

        Vec4f lineEq = rhoTheta2XY(rho, theta);

        // plot
//        line(myImage, Point(round(lineEq(0)), round(lineEq(1))), Point(round(lineEq(2)), round(lineEq(3))), Scalar(0,0,255), 2, 8);
    }
    return clusteredLines;
}

/** =============================================================================
    description: simple script to convert line in rho/theta format to ([x1, y1], [x2, y2]) format
    input: doubles rho and theta
    returns: vector<float>(4) : {x1, y1, x2, y2}
**/
Vec4f rhoTheta2XY(float rho, float theta){

   float a = cos(theta);
   float b = sin(theta);
   float x0 = a*rho;
   float y0 = b*rho;

   Vec4f xy;
   xy(0) = (x0 + 1000*(-b));
   xy(1) = (y0 + 1000*(a));
   xy(2) = (x0 - 1000*(-b));
   xy(3) = (y0 - 1000*(a));

    return xy;
}


/** =============================================================================
    description: simple script to convert line in rho/theta format to y = mx + b format
    input: doubles rho and theta
    returns: vector<float>(2) : {m, b}
**/
Vec2f rhoTheta2SlopeIntercept(float rho, float theta){
    // call rhoTheta2XY() first;
    // and then compute slope and intercept

    Vec4f xy;
    xy = rhoTheta2XY(rho, theta);

    float m = (xy[3] - xy[1]) / (xy[2] - xy[0]); // rise / run
    float b = xy[3] - (m*xy[2]);

    Vec2f output;
    output[0] = m;
    output[1] = b;
    return output;
}

/** =============================================================================
    description: tries to find corners by computing intersections of lines
    input: vector of lines in rho/theta format
    returns: void
**/
std::vector <Vec2f> computeCorners(std::vector<Vec2f> clusteredLines, Mat inputImage){

    // convert all lines from rho/theta into slope intercept
    std::vector< Vec2f> lines;
    for (int i = 0; i < clusteredLines.size(); i++){
        // convert all lines to slope intercept form (y = m*x + b)
        Vec2f line = rhoTheta2SlopeIntercept(clusteredLines[i][0], clusteredLines[i][1]);
        lines.push_back(line);
    }

    // compute intersection point of each line with each other line
    std::cout << "found intersections:\n";
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
                std::cout << "\t [" << xInt << ", " << yInt << "]\n";

                // and plot
//                circle(inputImage, Point(xInt, yInt), 5, Scalar(255,0,0), 2, 8);
            }
        }
    }

    std::cout << "found " << intersectionPoints.size() << " intersection points\n";
    return intersectionPoints;
}

/** =============================================================================
    description: puts polygon points in order clockwise
    input: vector of Vec2f points
    returns: void
  **/
std::vector<Vec2f> putPointsInOrder(std::vector<Vec2f> intersectionPoints){
    std::cout << "inside putPointsInOrder()\n";
    std::cout << "\t number of intersection points = " << intersectionPoints.size() << "\n";

    std::vector< std::vector< float> > tempPoints;

    // compute centroid of all points
    Vec2f centroid;
    for (int i = 0; i < intersectionPoints.size(); i++){
        centroid[0] = centroid[0] + intersectionPoints[i][0];
        centroid[1] = centroid[1] + intersectionPoints[i][1];
    }
    centroid[0] = centroid[0] / intersectionPoints.size();
    centroid[1] = centroid[1] / intersectionPoints.size();
    std::cout << "\t centroid of intersection points = [" << centroid[0] << ", " << centroid[1] << "]\n";

    // compute the heading from centroid to each point; order points by heading
    for (int i = 0; i < intersectionPoints.size(); i++){
        float heading = atan((intersectionPoints[i][1] - centroid[1]) / (intersectionPoints[i][0] - centroid[0]));
        if (intersectionPoints[i][0] - centroid[0] < 0){
            heading = heading* (-1);
            if (heading > 0){
                heading = heading+(M_PI/2);
            }
            else{
                heading = heading-(M_PI/2);
            }
        }
        std::cout << "\t\t heading = " << heading << "\n";

        // put heading in a new temp vector along with intersection point x,y
        // by putting heading in first column, I can easily sort later
        std::vector<float> tempV;
        float tempA[] = {heading, intersectionPoints[i][0], intersectionPoints[i][1]};
        tempV.assign (tempA,tempA+3);   // assigning from array.
        tempPoints.push_back(tempV); // put
    }

    // now do ordering
    std::sort(tempPoints.begin(), tempPoints.end());

    // overwrite intersectionPoints with ordered points
    intersectionPoints.clear();
    for (int i = 0; i < tempPoints.size(); i++){
        Vec2f orderedPoint;
        orderedPoint[0] = tempPoints[i][1];
        orderedPoint[1] = tempPoints[i][2];
        std::cout << "\t\t[" << tempPoints[i][0] << ", " << tempPoints[i][1] << ", " << tempPoints[i][2] << "]\n";
        intersectionPoints.push_back(orderedPoint);
    }

    return intersectionPoints;
}

