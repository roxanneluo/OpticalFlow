#include "project.h"
#include "Image.h"
#include "OpticalFlow.h"

using namespace cv;

class CVOpticalFlow {
  public:
    // Move im2 to make it look like im1
    static void findFlowND(Mat &vx, Mat &vy, Mat &warp, vector<Mat> &im1, vector<Mat> &im2, double alpha, double ratio, int minWidth, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations);
    static void findFlow(Mat &vx, Mat &vy, Mat &warp, Mat &im1, Mat &im2, double alpha, double ratio, int minWidth, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations);
    static Mat showFlow(Mat &vx, Mat &vy);
    static void warp(Mat &out, Mat &im, Mat &vx, Mat &vy);
    static void bilinear(double *out, Mat &im, double r, double c, int channels);
  private:
    static inline void clip(int &a, int lo, int hi);

};


