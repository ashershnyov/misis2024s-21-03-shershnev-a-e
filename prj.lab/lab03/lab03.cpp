#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <iostream>

uchar contrast(const uchar col, const uchar col_min_old, const uchar col_max_old,
               const uchar col_min_new, const uchar col_max_new) {
  float res = col_min_new + (col - col_min_old) * (col_max_new - col_min_new) / (col_max_old - col_min_old);
  return cvRound(res);
}

cv::Mat generate_hist(const cv::Mat img) {
  cv::Mat hist;
  int hist_size = 256;
  float range[] = {0, 256};
  const float* hist_range[] = {range};
  cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &hist_size, hist_range, true, false);

  int hist_w = 256, hist_h = 256;
  int bin_w = cvRound(double(hist_w / hist_size));

  cv::Mat hist_image(hist_w, hist_h, 0, 230);

  cv::normalize(hist, hist, 0, 230, cv::NORM_MINMAX, -1, cv::Mat());

  for (int i = 0; i < hist_size; i++) {
      cv::Point b_topleft = cv::Point(bin_w*(i), cvRound(hist.at<float>(i)));
      cv::Point b_bottomright = cv::Point(bin_w*(i + 1), 0);
      cv::rectangle(hist_image, b_topleft, b_bottomright, 0, cv::FILLED);
  }

  // TODO: костыль убрать
  cv::Mat hist_image_tmp;
  cv::flip(hist_image, hist_image_tmp, 0);
  return hist_image_tmp;
}

cv::Mat calc_hist(const cv::Mat img) {
  cv::Mat hist;
  int hist_size = 256;
  float range[] = {0, 256};
  const float* hist_range[] = {range};
  cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &hist_size, hist_range, true, false);
  return hist;
}

int main(int argc, char* argv[]) {
  const cv::String parser_keys = 
  "{qb |0.2 |}"
  "{qw |0.2 |}"
  ;
  cv::CommandLineParser parser(argc, argv, parser_keys);
  double q1 = parser.get<double>("qb");
  double q2 = parser.get<double>("qw");

  cv::Mat swin = cv::imread("..\\prj.lab\\lab03\\swin.jpg");
  // cv::Mat swin(256, 256, 0, cv::Scalar(255, 255, 255));

  int col_cnt = 256;
  uchar b_border = 0, w_border = 255;
  int black_cutoff = cvRound(col_cnt * q1), white_cutoff = cvRound(col_cnt * (1 - q2));


  std::vector<cv::Mat> channels;
  cv::split(swin, channels);
  cv::Mat hist;

  for (cv::Mat& channel: channels) {
    for (int y = 0; y < channel.rows; y++) {
      for(int x = 0; x < channel.cols; x++) {
        if (channel.at<uchar>(x, y) <= black_cutoff){
          channel.at<uchar>(x, y) = b_border;
        }
        else if(channel.at<uchar>(x,y) >= white_cutoff){
          channel.at<uchar>(x, y) = w_border;
        }
        else {
          channel.at<uchar>(x, y) = contrast(channel.at<uchar>(x, y), black_cutoff, white_cutoff, b_border, w_border);
        }
      }
    }
  }

  cv::Mat res_img = swin.clone();
  cv::merge(channels, res_img);

  hist = generate_hist(channels[0]);
  cv::imshow("hist2", hist);

  cv::hconcat(swin, res_img, res_img);
  cv::imwrite("swin.png", res_img);
  cv::waitKey(0);
}
