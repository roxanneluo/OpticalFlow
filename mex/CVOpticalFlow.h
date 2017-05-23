#ifndef CVOPTICALFLOW
#define CVOPTICALFLOW
#include "project.h"
#include "Image.h"
#include "OpticalFlow.h"


class CVOpticalFlow {
  public:
    // Move im2 to make it look like im1
    static void findFlowND(cv::Mat &vx, cv::Mat &vy, cv::Mat &warp, vector<cv::Mat> &im1, vector<cv::Mat> &im2, double alpha, double ratio, int minWidth, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations);
    static void findFlow(cv::Mat &vx, cv::Mat &vy, cv::Mat &warp, cv::Mat &im1, cv::Mat &im2, double alpha, double ratio, int minWidth, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations);
    static cv::Mat showFlow(cv::Mat &vx, cv::Mat &vy);
    static void warp(cv::Mat &out, cv::Mat &im, cv::Mat &vx, cv::Mat &vy);
    static void bilinear(double *out, cv::Mat &im, double r, double c, int channels);
    // flow_in = flow_out*disparity_scale + shift
    static cv::Mat writeFlow(const cv::Mat &flow_in, double *disparity_scale, double *shift, bool enable_flip_sign = true);
  private:
    static inline void clip(int &a, int lo, int hi);

};

#endif
