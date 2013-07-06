# include "mainAR.h"

using namespace std;

void callBackCentroidButton(int state, void* pointer){
    if (state == 0){
        centroidButtonState_ = false;
        destroyWindow("centroid image");
    }
    else if (state == 1){
        centroidButtonState_ = true;
        namedWindow("centroid image", CV_WINDOW_AUTOSIZE);
        moveWindow("centroid image", 800, 10);

    }

    std::cout << "centroidButtonState_ = " << centroidButtonState_ << "\n";

}


void callBackMaskButton(int state, void* pointer){
    if (state == 0){
        maskButtonState_ = false;
        destroyWindow("masked image");
    }
    else if (state == 1){
        maskButtonState_ = true;
        namedWindow("masked image", CV_WINDOW_AUTOSIZE);
        moveWindow("masked image", 800, 40);
    }

    std::cout << "maskButtonState_ = " << maskButtonState_ << "\n";


}
