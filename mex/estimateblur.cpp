// Rule: g++ -D_LINUX_MAC -D_OPENCV -I/usr/include/opencv -I/usr/local/include/opencv -L/usr/local/lib estimateblur.cpp CVOpticalFlow.a -fopenmp -lopencv_highgui -lopencv_calib3d -I/usr/local/google/home/supasorn/gco /usr/local/google/home/supasorn/gco/gco.a -lopencv_core -lopencv_imgproc -lopencv_ml -lopencv_features2d -lopencv_flann -o estimateblur -w

#include "CVOpticalFlow.h"
#include <iostream>
#include "GCoptimization.h"

using namespace std;
using namespace cv;
#define e3 at<Vec3b>

string base = "/usr/local/google/home/supasorn/focalstackdata/";
string dset = "balls_a003";
int n;

vector<Mat> imgs;
struct flow {
  Mat x, y;
};
vector<flow> v;
vector<Mat> vVis;
vector<Mat> laps;
vector<Mat> diff;


double computePhotoConsistencyScore(Mat &a, Mat &b) {
  double score = 0;
  for (int i = 0; i < a.rows; i++) {
    for (int j = 0; j < a.cols; j++) {
      for (int k = 0; k < 3; k++) {
        score += fabs(a.at<Vec3d>(i, j)[k] - b.at<Vec3d>(i, j)[k]);
      }
    }
  }
  return score / (3 * a.rows * a.cols);
}

void subImage(Mat &in, Mat &out_, int size, int r, int c) {
  Mat out = Mat::zeros(size * 2 + 1, size * 2 + 1, CV_64FC3);
  int w = in.cols;
  int h = in.rows;
  for (int i = -size; i <= size; ++i) {
    for (int j = -size; j <= size; ++j) {
      int nr = i + r;
      int nc = j + c;
      if (nr >=0 && nr < h && nc >= 0 && nc < w) {
        out.at<Vec3d>(i + size, j + size) = in.at<Vec3d>(nr, nc);
      }
    }
  }
  out_ = out;
}
void estimateBlurPair(int id0, int id1, int r, int c, double &outa, double &outb) {
  int size = 12;
  Mat im0, im1;
  subImage(imgs[id0], im0, size, r, c);
  subImage(imgs[id1], im1, size, r, c);
  int h = imgs[0].rows;
  int w = imgs[0].cols;
  //imshow("im0", im0);
  //imshow("im1", im1);
  //moveWindow("im0", 100, 150 + h);
  //moveWindow("im1", 100 + size * 2 + 10, 150 + h);

  const int step = 7;
  Mat blr[2][step];
  blr[0][0] = im0.clone();
  blr[1][0] = im1.clone();

  int csize = 5;
  for (int i = 1; i < step; i++) {
    int si = i * 2 + 1;
    GaussianBlur(im0, blr[0][i], Size(si, si), 0, 0, BORDER_REFLECT_101);
    GaussianBlur(im1, blr[1][i], Size(si, si), 0, 0, BORDER_REFLECT_101);
  }

  for (int i = 0; i < step; i++) {
    subImage(blr[0][i], blr[0][i], csize, size, size);
    subImage(blr[1][i], blr[1][i], csize, size, size);
  }

  //imshow("im0s", blr[0][0]);
  //imshow("im1s", blr[1][0]);
  //moveWindow("im0s", 100, 200 + h);
  //moveWindow("im1s", 100 + size * 2 + 10, 200 + h);

  // Size of actual comparison, should be smaller than size
  double score[2][step] = {0};
  for (int i = 0; i < step; i++) {
    //imshow("blr[0][0]", blr[0][0]);
    //imshow("blr[1][i]", blr[1][i]);
    //moveWindow("blr[0][0]", 100, 200 + h);
    //moveWindow("blr[1][i]", 100 + size * 2 + 10, 200 + h);
    //waitKey(0);
    score[0][i] = computePhotoConsistencyScore(blr[0][0], blr[1][i]);
    score[1][i] = computePhotoConsistencyScore(blr[1][0], blr[0][i]);
  }
  //for (int i = 0; i < 5; i++) 
    //printf("%f ", score[0][i]);
  //printf("\n");
  //for (int i = 0; i < 5; i++) 
    //printf("%f ", score[1][i]);
  //printf("\n");

  double mn[2] = {1e10, 1e10};
  for (int i = 0; i < step; i++) {
    if (score[0][i] < mn[0])
      mn[0] = score[0][i];
    if (score[1][i] < mn[1])
      mn[1] = score[1][i];
  }
  //printf("%f %f\n", mn[0] - score[0][0], mn[1] - score[1][0]);
  //printf("\n\n");

  outa = mn[0] - score[0][0];
  outb = mn[1] - score[1][0];

}

void mouse(int event, int x, int y, int flags, void *param) {
  if (event == CV_EVENT_LBUTTONDOWN) {
    double table[2][n];
    int prob = 12;
    int pr = y, pc = x;
    for (int i = 0; i < n; i++) {
      estimateBlurPair(i, prob, pr, pc, table[0][i], table[1][i]);
    }
    for (int i = 0; i < n; i++) {
      printf("(%+02.4f, %+02.4f, %c)", table[0][i], table[1][i], table[0][i] > table[1][i] ? '+': '-');
      if (i == prob)
        printf(" **");
      printf("\n");
    }

    for (int i = 0; i < n; i++) {
      if (i == prob) 
        printf("0");
      else if (table[0][i] > table[1][i])
        printf("+");
      else
        printf("-");
    }
    printf("\n");
    Mat a = imgs[prob].clone();
    a.at<Vec3d>(pr, pc) = Vec3d(0, 0, 1);
    imshow("a", a);
    printf("\n");

  }
}

Mat visualizeFlow(flow &in) {
  Mat out(in.x.size(), CV_8UC3);
  for (int i = 0; i < in.x.rows; i++) {
    for (int j = 0; j < in.y.cols; j++) {
      double sz = sqrt((in.y.at<double>(i, j) * in.y.at<double>(i, j)) + (in.x.at<double>(i, j) * in.x.at<double>(i, j)));
      sz *= 255;
      if (sz > 255) sz = 255;
      double ang = atan2(in.y.at<double>(i, j), in.x.at<double>(i, j));
      ang = (ang + M_PI) * 180 / M_PI / 2;
      out.at<Vec3b>(i, j) = Vec3b(ang, 255, sz);
    }
  }
  cvtColor(out, out, CV_HSV2BGR);
  return out;
}

void loadFiles() {
  char tmp[256];
  for (n = 0; 1; n++) {
    sprintf(tmp, (base + dset + "/%02d.jpg").c_str(), n);
    FILE *fi = fopen(tmp, "rb");
    if (!fi) break;
  }

  Mat im;
  FILE *fi;
  imgs.resize(n);
  v.resize(n);
  vVis.resize(n);
  for (int i = 0; i < n; i++) {
    sprintf(tmp, (base + dset + "/%02d.jpg").c_str(), i);
    imgs[i] = imread(tmp);
    imgs[i].convertTo(imgs[i], CV_64FC3, 1.0/255, 0);

    if (i > 0) {
      sprintf(tmp, (base + dset + "/%02dw_vx.bin").c_str(), i);
      fi = fopen(tmp, "rb");
      v[i].x = Mat(imgs[i].size(), CV_64F);
      fread(&v[i].x.at<double>(0, 0), sizeof(double), v[i].x.rows * v[i].x.cols, fi);
      fclose(fi);

      sprintf(tmp, (base + dset + "/%02dw_vy.bin").c_str(), i);
      fi = fopen(tmp, "rb");
      v[i].y = Mat(imgs[i].size(), CV_64F);
      fread(&v[i].y.at<double>(0, 0), sizeof(double), v[i].y.rows * v[i].y.cols, fi);
      fclose(fi);

      vVis[i] = visualizeFlow(v[i]);
    }
  }

}

// Hypothesize in focus plane and blur outward, then compute scores
void method1(Mat &out, int id) {
  out = Mat::zeros(imgs[0].size(), CV_64F);
  const int step = 5;
  Mat blr[step];

  blr[0] = imgs[id].clone();
  for (int i = 1; i < step; i++) {
    int si = i * 2 + 1;
    GaussianBlur(imgs[id], blr[i], Size(si, si), 0, 0, BORDER_REFLECT_101);
  }

  int csize = 5;

  vector<int> pr;
  int sep = 5;
  for (int i = 1; id + sep * i < n; i++) 
    pr.push_back(id + sep * i);
  for (int i = 1; id - sep * i >=0; i++)
    pr.push_back(id - sep * i);

  int rowCount = 0;
  #pragma omp parallel for
  for (int i = 0; i < imgs[0].rows; i++) {
    for (int j = 0; j < imgs[0].cols; j++) {
      double sumScore = 0;
      for (int k = 0; k < pr.size(); k++) {
        double minScore = 1e10;
        Mat patch1;
        subImage(imgs[pr[k]], patch1, csize, i, j);
        for (int l = 0; l < step; l++) {
          Mat patch0;
          subImage(blr[l], patch0, csize, i, j);
          double score = computePhotoConsistencyScore(patch0, patch1);
          if (score < minScore) 
            minScore = score;
        }
        sumScore += minScore;
      }
      out.at<double>(i, j) = sumScore / pr.size();
    }
    //#pragma omp critical(count)
    //{
      //rowCount ++;
      //printf("%f\n", 100.0 * rowCount / imgs[0].rows);
    //}
  }
  //imshow("out", out );
  //waitKey(0);
  //
}


void estimateSlope(int id) {
  //for (int i = 0; i < )
}
int main(int argc, char **argv) {
  if (argc > 1) {
    dset = argv[1];
    printf("Dataset: %s\n", dset.c_str());
  }
  loadFiles();

  double alpha = 0.03;
  double ratio = 0.85;
  int minWidth = 20;
  int nOuterFPIterations = 4;
  int nInnerFPIterations = 1;
  int nSORIterations = 40;

  imshow("a", imgs[0]);
  moveWindow("a", 100, 100);
  
  vector<Mat> out;
  out.resize(n);
  for (int i = 0; i < n; i++) {
    printf("%d\n", i);
    method1(out[i], i);
  }

  Mat depth(imgs[0].size(), CV_64F);

  for (int i = 0; i < imgs[0].rows; i++) {
    for (int j = 0; j < imgs[0].cols; j++) {
      double minScore = 1e10;
      int minId = 0;
      for (int k = 0; k < n; k++) {
        if (out[k].at<double>(i, j) < minScore) {
          minScore = out[k].at<double>(i, j);
          minId = k;
        }
      }
      depth = 1.0 * minId / (n - 1);
    }
  }

  imshow("depth", depth);

  setMouseCallback("a", mouse);
  waitKey(0);
}

