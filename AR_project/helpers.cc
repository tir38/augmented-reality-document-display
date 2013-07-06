// source file for non-critical helper methods
# include "mainAR.h"
# include <vector>      // for vectors
# include <stdio.h>     // for basic cout
# include "math.h"      // for arctan
# include <algorithm>   // for sort
//# include <Magick++.h>   // for zbar: QR code scanning
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
    input: vector of Vec2f points
    returns: void
  **/
std::vector<Vec2f> putPointsInOrder(std::vector<Vec2f> intersectionPoints){
    std::cout << "\nputting points in order;\n";

    std::vector< std::vector< float> > tempPoints;

    // compute centroid of all points
    Vec2f centroid;
    for (int i = 0; i < intersectionPoints.size(); i++){
        centroid[0] = centroid[0] + intersectionPoints[i][0];
        centroid[1] = centroid[1] + intersectionPoints[i][1];
    }
    centroid[0] = centroid[0] / intersectionPoints.size();
    centroid[1] = centroid[1] / intersectionPoints.size();
    //std::cout << "\t centroid of intersection points = [" << centroid[0] << ", " << centroid[1] << "]\n";

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
    std::cout << "\t ordered points :\n";
    for (int i = 0; i < tempPoints.size(); i++){
        Vec2f orderedPoint;
        orderedPoint[0] = tempPoints[i][1];
        orderedPoint[1] = tempPoints[i][2];
        std::cout << "\t[" << tempPoints[i][0] << ",\t " << tempPoints[i][1] << ",\t " << tempPoints[i][2] << "\t]\n";
        intersectionPoints.push_back(orderedPoint);
    }

    return intersectionPoints;
}



/** =============================================================================
**/
Mat loadDisplayImage(std::string filename){

    // this is where i will determine what file type i am trying to load
    // and do any neccessary conversions into readable format(s)

    Mat overlayImage = imread(filename);
    std::cout << "\t overlay image size = [" << overlayImage.cols << ", " << overlayImage.rows << "]\n";

    return overlayImage;
}
