#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

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
    cv::Mat img = cv::imread("test.png");
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    std::vector<float> means;

    float gray = 0.0;

    for (cv::Mat channel: channels) {
        means.push_back(cv::mean(channel)[0]);
        mat_to_linrgb(channel);
    }

    for (int i = 0; i < channels.size(); i++) {
        channels[i] *= cv::mean(means)[0] / means[i];
        mat_to_srgb(channels[i]);
    }

    cv::Mat res_img;
    cv::merge(channels, res_img);
    cv::hconcat(img, res_img, res_img);
    // cv::imshow("test_wb", res_img);
    cv::imwrite("test_wb.png", res_img);
    // cv::waitKey(0);
}