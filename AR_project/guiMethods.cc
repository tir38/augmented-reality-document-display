// stores all GUI button callback methods

# include "mainAR.h"

using namespace std;

/**
**/
void callBackCentroidButton(int state, void* pointer){
    if (state == 0){
        centroidButtonState_ = false;
        destroyWindow("centroid image");
    }
    else if (state == 1){
        centroidButtonState_ = true;
        namedWindow("centroid image", CV_WINDOW_AUTOSIZE);
        moveWindow("centroid image", 100, 100);
    }
    std::cout << "centroidButtonState_ = " << centroidButtonState_ << "\n";
}

/**
**/
void callBackMaskButton(int state, void* pointer){
    if (state == 0){
        maskButtonState_ = false;
        destroyWindow("masked image");
    }
    else if (state == 1){
        maskButtonState_ = true;
        namedWindow("masked image", CV_WINDOW_AUTOSIZE);
        moveWindow("masked image", 200, 100);
    }
    std::cout << "maskButtonState_ = " << maskButtonState_ << "\n";
}

/**
**/
void callBackCannyButton(int state, void* pointer){
    if (state == 0){
        cannyButtonState_ = false;
        destroyWindow("canny edge detection");
    }
    else if (state == 1){
        cannyButtonState_ = true;
        namedWindow("canny edge detection", CV_WINDOW_AUTOSIZE);
        moveWindow("canny edge detection", 300,100);

        cannyThres1_ = 0;
        cannyThresh2_ = 95;
        createTrackbar("Canny Low", "canny edge detection", &cannyThres1_, 100);
        createTrackbar("Canny High", "canny edge detection", &cannyThresh2_, 100);
    }
    std::cout << "cannyButtonState_ = " << cannyButtonState_ << "\n";
}


/**
**/
void callBackHoughButton(int state, void* pointer){
    if (state == 0){
        houghButtonState_ = false;
        destroyWindow("hough lines");
    }
    else if (state == 1){
        houghButtonState_ = true;
        namedWindow("hough lines", CV_WINDOW_AUTOSIZE);
        moveWindow("hough lines", 400,100);

        houghThresh_ = 80;
        createTrackbar("Hough threshold", "hough lines", &houghThresh_, 100);
    }
    std::cout << "houghButtonState_ = " << houghButtonState_ << "\n";
}

/**
**/
void callBackClusterButton(int state, void* pointer){
    if (state == 0){
        clusterButtonState_ = false;
        destroyWindow("clustered lines");
    }
    else if (state == 1){
        clusterButtonState_ = true;
        namedWindow("clustered lines", CV_WINDOW_AUTOSIZE);
        moveWindow("clustered lines", 500,100);
    }
    std::cout << "clusterButtonState_ = " << clusterButtonState_ << "\n";
}

/**
**/
void callBackCornersButton(int state, void* pointer){
    if (state == 0){
        cornersButtonState_ = false;
        destroyWindow("corners");
    }
    else if (state == 1){
        cornersButtonState_ = true;
        namedWindow("corners", CV_WINDOW_AUTOSIZE);
        moveWindow("corners", 600,100);
    }
    std::cout << "cornersButtonState_ = " << cornersButtonState_ << "\n";
}

/**
**/
void callBackPerspectiveButton(int state, void* pointer){
    if (state == 0){
        perspectiveButtonState_ = false;
        destroyWindow("perspective");
    }
    else if (state == 1){
        perspectiveButtonState_ = true;
        namedWindow("perspective", CV_WINDOW_AUTOSIZE);
        moveWindow("perspective", 700,100);
    }
    std::cout << "perspectiveButtonState_ = " << perspectiveButtonState_ << "\n";
}


/**
**/
void callBackInverseButton(int state, void* pointer){
    if (state == 0){
        inverseButtonState_ = false;
        destroyWindow("inverse");
    }
    else if (state == 1){
        inverseButtonState_ = true;
        namedWindow("inverse", CV_WINDOW_AUTOSIZE);
        moveWindow("inverse", 600,100);
    }
    std::cout << "inverseButtonState_ = " << inverseButtonState_ << "\n";
}
