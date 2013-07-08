// source file for non-critical helper methods
# include "mainAR.h"
# include <vector>      // for vectors
# include <stdio.h>     // for basic cout
# include "math.h"      // for arctan
# include <algorithm>   // for sort
# include <zbar.h>       // for zbar

using namespace cv;
using namespace zbar;


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
    description: puts polygon points in order clockwise
    input: inputPoints: vector<Point2f> points in random order
    returns: outputPoints: vector<Point2f> points in order
  **/
std::vector<Point2f> putPointsInOrder(std::vector<Point2f> inputPoints, Mat inputImage){

    std::vector< std::vector< float> > tempPoints;

    // compute centroid of all points
    Point2f centroid;
    for (int i = 0; i < inputPoints.size(); i++){
        centroid.x = centroid.x + inputPoints[i].x;
        centroid.y = centroid.y + inputPoints[i].y;
    }
    centroid.x = centroid.x / inputPoints.size();
    centroid.y = centroid.y / inputPoints.size();

    // compute the heading from centroid to each point; order points by heading
    for (int i = 0; i < inputPoints.size(); i++){
        float heading = atan((inputPoints[i].y - centroid.y) / (inputPoints[i].x - centroid.x));
        if (inputPoints[i].x - centroid.x < 0){
            heading = heading* (-1);
            if (heading > 0){
                heading = heading+(M_PI/2);
            }
            else{
                heading = heading-(M_PI/2);
            }
        }

        // put heading in a new temp vector along with intersection point x,y
        // by putting heading in first column, I can easily sort later
        std::vector<float> tempV;
        float tempA[] = {heading, inputPoints[i].x, inputPoints[i].y};
        tempV.assign (tempA,tempA+3);   // assigning from array.
        tempPoints.push_back(tempV); // put
    }

    // now do ordering
    std::sort(tempPoints.begin(), tempPoints.end());

    // overwrite intersectionPoints with ordered points
    std::vector<Point2f> outputPoints;
    for (int i = 0; i < tempPoints.size(); i++){
        Point2f orderedPoint;
        orderedPoint.x = tempPoints[i][1];
        orderedPoint.y = tempPoints[i][2];
        outputPoints.push_back(orderedPoint);
    }

    // if button turned on, draw polygon
    if(orderButtonState_){
        Mat plotableImage = inputImage.clone();
        for (int i = 0; i < outputPoints.size()-1; i++){
            line(plotableImage, Point(round(outputPoints[i].x), round(outputPoints[i].y)), Point(round(outputPoints[i+1].x), round(outputPoints[i+1].y)), Scalar(255,255,0), 2, 8);
        }
        line(plotableImage, Point(round(outputPoints[0].x), round(outputPoints[0].y)), Point(round(outputPoints[inputPoints.size()-1].x), round(outputPoints[inputPoints.size()-1].y)), Scalar(255,255,0), 2, 8); // closing line

        imshow("order", plotableImage);
    }


    return outputPoints;
}



/** =============================================================================
**/
Mat loadDisplayImage(std::string filename){

    // this is where i will determine what file type i am trying to load
    // and do any neccessary conversions into readable format(s)

    // read in file
    Mat overlayImage = imread(filename, 1); // 1 = force load as 3-channel

    std::cout << "\t overlayImage's type: " << overlayImage.type() << ", " <<  overlayImage.channels() << ", " << overlayImage.depth() << "\n";

    // add in an alpha channel by splitting and rejoining
    Mat output;
    Mat alphaChannel;
    alphaChannel = Mat::ones(overlayImage.rows, overlayImage.cols, CV_8UC1); // create alpha channel with all ones
    std::vector<Mat> arrayOfChannels;
    split(overlayImage, arrayOfChannels); // split into array [B, G, R]
    arrayOfChannels.push_back(alphaChannel); // add alpha to array
    std::cout<< "\t size of arrayOfChannels = " << arrayOfChannels.size() << "\n";
    merge(arrayOfChannels, output); // merge all channels in array

    std::cout << "\t overlay image size = [" << output.cols << ", " << output.rows << "]; channels = " << output.channels() << "\n";

    return output;
}


/** =============================================================================
    description: creates overlay of two images by cutting ROI pixels out of foreground and non-ROI pixels out of background and merging into single image
    input: backgroundImage: Mat 3 channel, BGR
            foregroundImage: Mat 4 channel (BGRA, with alpha defined)
    output: outputImage, Mat 3 channel, BGR
**/
Mat doOverlay(Mat backgroundImage, Mat foregroundImage){
    std::cout << "\ndo overlay\n";
    // THIS OVERLAY TECHNIQUE IS SUPER NASTY; READ CAREFULLY.

    // threshold foreground to create mask of pixels with alpha = 1.0
    Mat mask;                                                           // to store alpha mask
    mask.create(foregroundImage.rows, foregroundImage.cols, CV_8U);     // force to be 8bit unsigned, single channel
    Scalar lowerBound(0,   0,      0, 1.0);                             // set threshold values; all BGR; alpha = 1.0
    Scalar upperBound(255, 255,    255, 1.0);
    inRange(foregroundImage, lowerBound, upperBound, mask);             // get mask

    // use mask to save ROI of foreground
    Mat saveForegroundImage;                                // to store part of foreground to keep
    foregroundImage.copyTo(saveForegroundImage, mask);      // get part to keep

    // convert background to 4 channel (add in an alpha channel by splitting and rejoining)
    Mat background4ChannelImage;
    Mat alphaChannel;
    alphaChannel = Mat::ones(backgroundImage.rows, backgroundImage.cols, CV_8UC1); // create alpha channel with all ones
    std::vector<Mat> arrayOfChannels;                   // declare array to store channels
    split(backgroundImage, arrayOfChannels);            // split into array [B, G, R]
    arrayOfChannels.push_back(alphaChannel);            // add alpha to array
    merge(arrayOfChannels, background4ChannelImage);    // merge all channels in array


    // crop and save image from background
    Mat inverseMask;                                // get pixels NOT in the mask, this will become the mask for background
    bitwise_not(mask, inverseMask);                 // not mask
    Mat saveBackgroundImage;                        // to store part of the background to save
    background4ChannelImage.copyTo(saveBackgroundImage, inverseMask);   // get part of background in inverseMask


    // combine saved background and saved foreground
    Mat output = saveForegroundImage + saveBackgroundImage;
    return output;
}




