// this is a file for me to put code snippets that I wrote but don't need.
// for cleaning up the rest of the code


// ===================================
// GENERATE CONTOURS; was part of trackObject()
    // compute contours
    std::vector< std::vector< Point> > contours = computeContours(closedImage); // update myImage
    std::cout << "there are some contours :"  << contours.size() << "\n";

    // draw contours
    for (int i = 0; i < contours.size(); i++){
        std::vector<Point> singleContour = contours[i];                     // get single contour
        Scalar randomColor = Scalar(rand()&255, rand()&255, rand()&255);    // generate random colro
        drawContours(myImage, contours, i, randomColor, 2, 8 );             // draw single contour
        int numPoints = singleContour.size();                               // get number of points
        std::cout << "\t contour [" << i << "] has " << numPoints << " points.\n";
    }

    Mat justContoursImage;
    justContoursImage.create(myImage.rows, myImage.cols, CV_8U); // also draw contours on new image
    drawContours(justContoursImage, contours, -1, Scalar(255), 2, 8);
