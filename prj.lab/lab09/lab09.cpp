#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include<iostream>

cv::Mat calc_hist(const cv::Mat img) {
  cv::Mat hist;
  int hist_size = 256;
  float range[] = {0, 256};
  const float* hist_range[] = {range};
  cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &hist_size, hist_range, true, false);
  return hist;
}

cv::Mat generate_hist(const cv::Mat img) {
  cv::Mat hist;
  int hist_size = 256;
  float range[] = {0, 256};
  const float* hist_range[] = {range};
  cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &hist_size, hist_range, true, false);

  int hist_w = img.size().width, hist_h = img.size().height;
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

int main() {
    cv::Mat img = cv::imread("./../prj.lab/lab09/swin.jpg");
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    std::vector<float> means;

    cv::Mat hist_1 = generate_hist(channels[0]);
    cv::Mat hist_2 = generate_hist(channels[1]);
    cv::Mat hist_3 = generate_hist(channels[2]);
    cv::Mat hist;
    cv::merge(std::vector<cv::Mat>{hist_1, hist_2, hist_3}, hist);

    cv::vconcat(img, hist, img);

    float gray = 0.0;

    for (cv::Mat channel: channels) {
        means.push_back(cv::mean(channel)[0]);
    }

    for (int i = 0; i < channels.size(); i++) {
        // channels[i] *= means[i] / cv::mean(means)[0];
        channels[i] *= cv::mean(means)[0] / means[i];
    }

    hist_1 = generate_hist(channels[0]);
    hist_2 = generate_hist(channels[1]);
    hist_3 = generate_hist(channels[2]);

    cv::merge(std::vector<cv::Mat>{hist_1, hist_2, hist_3}, hist);
    cv::Mat res_img;
    cv::merge(channels, res_img);
    cv::vconcat(res_img, hist, res_img);
    cv::hconcat(img, res_img, res_img);
    cv::imshow("swin", res_img);
    cv::waitKey(0);
}