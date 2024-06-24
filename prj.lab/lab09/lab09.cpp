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
            img.at<uchar>(y, x) = srgb_to_linrgb(img.at<uchar>(y, x));
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
            img.at<uchar>(y, x) = linrgb_to_srgb(img.at<uchar>(y, x));
        }
    }
}

int main() {
    // cv::Mat img = cv::imread("./../prj.lab/lab09/swin.jpg");
    cv::Mat img = cv::imread("test_l09.jpg");
    std::vector<cv::Mat> channels, channels_masked;
    cv::Mat img_masked_table  = img.clone();
    std::vector<cv::Point> table_to_mask = {cv::Point{38, 534}, cv::Point{211, 538},
                                            cv::Point{206, 793}, cv::Point{31, 791}};
    cv::fillPoly(img_masked_table, table_to_mask, 0, 4);
    cv::imwrite("masked_table.png", img_masked_table);
    cv::split(img_masked_table, channels_masked);
    cv::split(img, channels);
    std::vector<float> means;

    float gray = 0.0;

    std::vector<cv::Mat1b> hists;

    for (cv::Mat channel: channels_masked) {
        means.push_back(cv::mean(channel)[0]);
    }

    for (cv::Mat channel: channels) {
        mat_to_linrgb(channel);
    }

    for (int i = 0; i < channels.size(); i++) {
        channels[i] *= cv::mean(means)[0] / means[i];
        mat_to_srgb(channels[i]);
        hists.push_back(generate_hist(channels[i]));
    }

    std::vector<std::pair<cv::Point, cv::Vec3b>> circles {
        std::pair<cv::Point, cv::Vec3b>{cv::Point(183, 768), cv::Vec3b{243, 243, 242}},
        std::pair<cv::Point, cv::Vec3b>{cv::Point(185, 725), cv::Vec3b{200, 200, 200}},
        std::pair<cv::Point, cv::Vec3b>{cv::Point(187, 683), cv::Vec3b{160, 160, 160}},
        std::pair<cv::Point, cv::Vec3b>{cv::Point(187, 642), cv::Vec3b{122, 122, 121}},
        std::pair<cv::Point, cv::Vec3b>{cv::Point(188, 600), cv::Vec3b{85, 85, 85}},
        std::pair<cv::Point, cv::Vec3b>{cv::Point(188, 556), cv::Vec3b{52, 52, 52}}
    };

    std::vector<float> mse;
    for(auto cir: circles) {
        cv::Mat_<float> color;
        cv::Mat mask = cv::Mat(img.size(), CV_8UC3);
        cv::circle(mask, cir.first, 10, cv::Scalar{255, 255, 255}, cv::FILLED, cv::FILLED);

        cv::Mat orig_masked, truth_masked, truth(img.size(), CV_8UC3, cir.second);
    
        cv::bitwise_and(img, mask, orig_masked);
        cv::bitwise_and(truth, mask, truth_masked);

        cv::Mat difference;
        cv::absdiff(orig_masked, truth_masked, difference);
        difference = difference.mul(difference);

        cv::Scalar sum = cv::sum(difference);

        cv::cvtColor(mask, mask, cv::COLOR_RGB2GRAY);

        mse.push_back(cv::sqrt((sum[0] + sum[1] + sum[2]) / cv::countNonZero(mask) / 3));
    }

    for (float diff: mse) {
        std::cout << diff << "\n";
    }

    // cv::imwrite("mask1.png", mask);

    cv::Mat1b hist = hists[0].clone();
    cv::hconcat(hist, hists[1], hist);
    cv::hconcat(hist, hists[2], hist);

    cv::Mat res_img;
    cv::merge(channels, res_img);
    cv::hconcat(img, res_img, res_img);
    // cv::imshow("test_wb", res_img);
    cv::imwrite("test_wb.png", res_img);

    // cv::imwrite("test_wb_hist.png", hist);
    // cv::waitKey(0);
}