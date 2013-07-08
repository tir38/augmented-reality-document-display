Augmented Reality Document Loader and Display
-----

This is a project I cooked up to learn openCV and play around with augmented reality. The code is a bit chopped together as such the level of refinement is maybe a "B-".

Workflow
---
1. get single frame from webcam => initial image
1. convert image to HSV
2. threshold image by intensity (V) to remove dark background and text on page; 
    this becomes mask for all masking
3. mask initial image
4.  perform "my method" :
    - perform Canny edge detection
    - generate Hough lines from Canny edges; lines are in (rho, theta) space
    - perform k-means clustering on Hough lines in (rho, theta) space into (4) bins
    - find intersection points of (4) lines
9. in tandem with "my method",  use "built-in" cv::goodFeaturesToTrack() to get Harris corners
10. combine (4) Harris corners and (4) intersection points; 
11. select "best" (4) from the set of (8) **
12. put points in clockwise order around centroid
12. compute warp matrix for points => (4) corners of "8.5 x 11"in. size window ++
13. perform "forward" perspective projection on initial image using warp matrix
14. read QR code from warped image
15. decode QR code and load file from folder
16. perform "reverse" perspective projection on loaded image, using inverse warp matrix
17. "overlay" loaded image onto initial image $$

** "best" at this point is just the (4) intersection points; The addition of Harris corners actually makes things jumpy. My goal is/was to do k-means on the (8) points to find (4) "best" points. This isn't really working.

++ 8.5 x 11 in. is relative; its really a window that is 480 x (480*11/8.5) pixels

$$ openCV doesn't do overlay very well. my overlay method is a roundabout method of cutting foreground from loaded image, cutting background from intial image, and then putting them together in an output image.

See More
----
[Vimeo]()
[tir38.com]()

Required Libraries
----
- [openCV 2.x](http://opencv.willowgarage.com/wiki/) (built with 2.4) 
- [ZBar](http://zbar.sourceforge.net/)
- [ImageMagick++](http://www.imagemagick.org/script/index.php) (just for zBar)    

Status
-----
This was intially a project to learn about openCV (mission accomplished) and to build a fun augmented reality tool (mission mostly accomplished). I have a lot that I'd love to add to this project, but at the current state, I'm satisfied if I never have time to come back to this. Star this repo to keep up to date on any changes.
