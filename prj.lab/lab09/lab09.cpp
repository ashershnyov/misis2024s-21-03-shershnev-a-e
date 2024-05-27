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

uchar srgb_to_linrgb(float color) {
    color /= 255.f;
    if (color <= 0.04045) {
        return uchar(255 * color / 12.92f);
    } else {
        return uchar(255 * cv::pow((color + 0.055) / 1.055f, 2.4));
    }
}

void mat_to_linrgb(cv::Mat& img) {
    for (int x = 0; x < img.cols; x++) {
        for (int y = 0; y < img.rows; y++) {
            img.at<uchar>(x, y) = srgb_to_linrgb(img.at<uchar>(x, y));
        }
    }
}

uchar linrgb_to_srgb(float color) {
    color /= 255.f;
    if (color <= 0.0031308) {
        return uchar (255 * 12.92 * color);
    } else {
        return uchar (255 * 1.055 * (cv::pow(color, 1 / 2.4) - 0.055));
    }
}

void mat_to_srgb(cv::Mat& img) {
    for (int x = 0; x < img.cols; x++) {
        for (int y = 0; y < img.rows; y++) {
            img.at<uchar>(x, y) = linrgb_to_srgb(img.at<uchar>(x, y));
        }
    }
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
        mat_to_linrgb(channel);
        means.push_back(cv::mean(channel)[0]);
    }

    for (int i = 0; i < channels.size(); i++) {
        // channels[i] *= means[i] / cv::mean(means)[0];
        channels[i] *= cv::mean(means)[0] / means[i];
        mat_to_srgb(channels[i]);
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