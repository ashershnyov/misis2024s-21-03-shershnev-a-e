#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

// Проекция точки (цвета каждого канала) на плоскость вида
// ax + by + cz = d
cv::Vec3f project_point(const cv::Vec4f surface, const cv::Vec3f point) {
    float a = surface[0], b = surface[1], c = surface[2], d = -surface[3];
    float t = -(a * point[0] + b * point[1] + c * point[2] + d) / (a * a + b * b + c * c);
    return cv::Vec3f{
        a * t + point[0],
        b * t + point[1],
        c * t + point[2]
    };
}

// (0, 0, 1)
// (0, 1, 0)
// (1, 0, 0)

int main() {
    std::string fname = "swin.jpg";
    cv::Mat3f sample = cv::imread(fname);
    cv::Mat1b res(1024, 1024, 127);
    std::vector<cv::Vec3f> projected_points;
    for (int y = 0; y < sample.rows; y++) {
        for (int x = 0; x < sample.cols; x++) {
            // std::cout << sample.at<cv::Vec3f>(x, y) << "\n";
            cv::Vec3f point = sample.at<cv::Vec3f>(x, y) / 255.;
            std::cout << point << "\n";
            point = project_point(cv::Vec4f{1, 1, 1, 1}, point);
            // if (point[0] < 0 || point[1] < 0)
            //     std::cout << point << "\n";
            point = project_point(cv::Vec4f{0, 0, 1, 0}, point);
            cv::Size img_center = cv::Size{int(point[0] * res.cols), int(point[1] * res.rows)};
            cv::circle(res, img_center, 1, 0, 1, cv::FILLED);
        }
    }
    cv::imshow("swin", res);
    cv::waitKey(0);
    return 0;
}