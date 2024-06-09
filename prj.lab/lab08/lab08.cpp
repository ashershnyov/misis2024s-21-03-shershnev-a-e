#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>


cv::Mat kProj = (cv::Mat_<float>(2, 3) << 
    0.5 + 0.5 / cv::sqrt(3), -0.5 + 0.5 / cv::sqrt(3), -1 / cv::sqrt(3),
    -0.5 + 0.5 / cv::sqrt(3), 0.5 + 0.5 / cv::sqrt(3), -1 / cv::sqrt(3)
);


cv::Vec3f project_point(cv::Vec3f point) {
    cv::Mat tmp = (cv::Mat_<float>(3, 1) << point[0], point[1], point[2]);
    cv::Mat res;
    cv::gemm(kProj, tmp, 1.0, cv::Mat(), 0.0, res);
    return cv::Vec3f{res.at<float>(0, 0), res.at<float>(0, 1), res.at<float>(0, 2)};
}

uchar srgb_to_linrgb(float color) {
    color /= 255.f;
    if (color <= 0.04045) {
        return uchar(255 * color / 12.92f);
    } else {
        return uchar(255 * cv::pow((color + 0.055) / 1.055f, 2.4));
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

int main() {
    std::string fname = "waltuh2.jpg";
    cv::Mat3f sample = cv::imread(fname);
    cv::Mat1b res(1024, 1024, 255);
    std::vector<cv::Vec3f> projected_points;

    int x_center = res.cols / 2, y_center = res.rows / 2;
    // std::cout << x_center << " " << y_center << "\n";

    cv::Mat density_map = cv::Mat::zeros(res.size(), CV_32F);


    float max_val = 0.0;
    
    float scale = int(res.cols / 3) / (1. + 1. / cv::sqrt(3));
    for (int y = 0; y < sample.rows; y++) {
        for (int x = 0; x < sample.cols; x++) {
            cv::Vec3f point = sample.at<cv::Vec3f>(x, y);
            cv::Vec3f point_tm = point / 255.;
            point_tm = project_point(point_tm);
            int x_c = int(point_tm[0] * scale + x_center);
            int y_c = int(point_tm[1] * scale + y_center);
            if (x_c < density_map.cols && x_c >= 0 && y_c < density_map.rows && y_c >= 0){
                density_map.at<float>(x_c, y_c) += 1.0f;
            }
            if (density_map.at<float>(x_c, y_c) >= max_val) {
                max_val = density_map.at<float>(x_c, y_c);
            }
        }
    }

    // std::cout << max_val << "\n";

    density_map /= max_val;
    density_map *= 255.f;

    std::vector<cv::Point> points;
    for (int i = 0; i < 6; i++) {
        float angle = i * CV_PI / 3.f;
        int x = int(x_center + res.cols / 3 * std::cos(angle + 0.5 * CV_PI));
        int y = int(y_center + res.cols / 3 * std::sin(angle + 0.5 * CV_PI));
        points.push_back(cv::Point(x, y));
    }

    for (int i = 0; i < points.size(); i++) {
        cv::line(density_map, points[i], points[(i + 1) % points.size()], cv::Scalar(255.f, 255.f, 255.f), 3);
    }

    cv::Mat1f dens = density_map.clone();
    cv::imshow("map2", dens);
    // double mina, maxa;
    // cv::minMaxLoc(dens, &mina, &maxa);

    // std::cout << dens << "\n" << mina << " " << maxa << "\n";
    // std::vector<cv::Mat> chan;
    // cv::split(res, chan);
    // cv::cvtColor(density_map, density_map, cv::COLOR_RGB2GRAY);
    // cv::imwrite("map2.jpg", dens);
    // std::cout << density_map.type() << "\n";
    // cv::imshow("swin", res);
    cv::waitKey(0);
    return 0;
}