#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <iostream>

uchar contrast(const uchar col, const uchar col_min_old, const uchar col_max_old,
               const uchar col_min_new, const uchar col_max_new) {
  float res = (col_min_new
               + (col - col_min_old)
               * (col_max_new - col_min_new)
               / (col_max_old - col_min_old));
  return cvRound(res);
}

cv::Mat calc_hist(const cv::Mat img) {
  cv::Mat hist;
  int hist_size = 256;
  float range[] = {0, 256};
  const float* hist_range[] = {range};
  cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &hist_size, hist_range, true, false);
  return hist;
}

std::pair<uchar, uchar> find_min_max(const cv::Mat img, double qb, double qw) {
  uchar min = 0, max = 0;
  cv::Mat hist = calc_hist(img);
  int hist_sum = cv::sum(hist)[0];
  long int acc_sum = 0;
  for (int i = 0; i < hist.size().height; i++) {
    if (hist.at<float>(i) > 0) {
      acc_sum += hist.at<float>(i);
    }
    if (acc_sum >= qb * hist_sum) {
      min = i;
      break;
    }
  }
  acc_sum = 0;
  for (int i = hist.size().height - 1; i >= 0; i--) {
    if (hist.at<float>(i) > 0) {
      acc_sum += hist.at<float>(i);
    }
    if (acc_sum >= qw * hist_sum) {
      max = i;
      break;
    }
  }
  return std::pair<uchar, uchar>(min, max);
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


void channel_contrast(cv::Mat& channel, const uchar black_cutoff, const uchar white_cutoff) {
  uchar b_border = 0;
  uchar w_border = 255;
  for (int x = 0; x < channel.rows; x++) {
    for(int y = 0; y < channel.cols; y++) {
      if (channel.at<uchar>(x, y) <= black_cutoff){
        channel.at<uchar>(x, y) = b_border;
      }
      else if(channel.at<uchar>(x,y) >= white_cutoff){
        channel.at<uchar>(x, y) = w_border;
      }
      else {
        channel.at<uchar>(x, y) = contrast(channel.at<uchar>(x, y), black_cutoff,
                                           white_cutoff, b_border, w_border);
      }
    }
  }
}

int main(int argc, char* argv[]) {
  const cv::String parser_keys = 
  "{qb     |0.2|}"
  "{qw     |0.2|}"
  "{mode m |0  |}"
  ;
  cv::CommandLineParser parser(argc, argv, parser_keys);
  double q1 = parser.get<float>("qb");
  double q2 = parser.get<float>("qw");
  int mode = parser.get<int>("mode");

  cv::Mat swin = cv::imread("./../prj.lab/lab03/swin.jpg");

  std::vector<cv::Mat> channels;
  cv::split(swin, channels);
  cv::Mat hist;
  // hist = generate_hist(channels[0]);
  // cv::imshow("hist3", hist);
  uchar black_cutoff = 255, white_cutoff = 0;

  if (mode == 1) {
    for (cv::Mat& channel: channels) {
      std::pair<uchar, uchar> cutoffs = find_min_max(channel, q1, q2);
      if (cutoffs.first <= black_cutoff) {
        black_cutoff = cutoffs.first;
      } 
      if (cutoffs.second >= white_cutoff) {
        white_cutoff = cutoffs.second;
      }
    }
  }

  for (cv::Mat& channel: channels) {
    if (mode != 1){
      std::pair<uchar, uchar> cutoffs = find_min_max(channel, q1, q2);
      black_cutoff = cutoffs.first; white_cutoff = cutoffs.second;
    }
    channel_contrast(channel, black_cutoff, white_cutoff);
  }

  cv::Mat res_img = cv::Mat(swin.size(), swin.type());
  cv::merge(channels, res_img);

  // hist = generate_hist(channels[0]);
  // cv::imshow("hist2", hist);

  cv::hconcat(swin, res_img, res_img);
  cv::imwrite("swin.png", res_img);
  // cv::waitKey(0);
}