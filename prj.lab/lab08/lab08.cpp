#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

// cv::Mat kRotZ = (cv::Mat_<cv::Vec3f>(3, 1) << 
//     cv::Vec3f{float(std::cos(45)), 0, float(std::sin(45))},
//     cv::Vec3f{0, 1, 0},
//     cv::Vec3f{float(-std::sin(45)), 0, float(std::cos(45))}
// );

cv::Mat kRotY = (cv::Mat_<float>(3, 3) << 
    float(std::cos(45)), 0, float(std::sin(45)),
    0, 1, 0,
    float(-std::sin(45)), 0, float(std::cos(45))
);

cv::Mat kRotX = (cv::Mat_<float>(3, 3) << 
    1, 0, 0,
    0, float(std::cos(45)), float(-std::sin(45)),
    0, float(std::sin(45)), float(std::cos(45))
);

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

cv::Vec3f rotate_point_y(cv::Vec3f point) {
    cv::Mat tmp = (cv::Mat_<float>(1, 3) << point[0], point[1], point[2]);
    tmp = tmp * kRotY;
    // cv::Vec3f tmp = point.mul(kRotZ);
    return cv::Vec3f{tmp.at<float>(0, 0), tmp.at<float>(0, 1), tmp.at<float>(0, 2)};
    // return cv::Vec3f{0, 0, 0};
    // return tmp;
}

cv::Vec3f rotate_point_x(cv::Vec3f point) {
    cv::Mat tmp = (cv::Mat_<float>(1, 3) << point[0], point[1], point[2]);
    tmp = tmp * kRotX;
    return cv::Vec3f{tmp.at<float>(0, 0), tmp.at<float>(0, 1), tmp.at<float>(0, 2)};
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
    std::string fname = "swin.jpg";
    cv::Mat3f sample = cv::imread(fname);
    cv::Mat1b res(1024, 1024, 127);
    std::vector<cv::Vec3f> projected_points;

    for (float x = 0; x <= 1; x += 0.01) {
        for (float y = 0; y <= 1; y += 0.01) {
            for (float z = 0; z <= 1; z += 0.01) {
                cv::Vec3f point{x, y, z};
                point = project_point(cv::Vec4f{2, 2, 2, 3}, point);
                point = rotate_point_x(point);
                // point = rotate_point_y(point);
                // point = project_point(cv::Vec4f{0, 0, 1, 0}, point);
                cv::Size img_center = cv::Size{int(res.cols - point[0] * res.cols), int(point[1] * res.rows)};
                cv::circle(res, img_center, 1, 255, 1, cv::FILLED);
            }
        }
    }

    for (int y = 0; y < sample.rows; y++) {
        for (int x = 0; x < sample.cols; x++) {
            cv::Vec3f point = sample.at<cv::Vec3f>(x, y) / 255.;
            point = project_point(cv::Vec4f{2, 2, 2, 3}, point);
            point = rotate_point_x(point);
            // point = rotate_point_x(point);
            // point = project_point(cv::Vec4f{0, 0, 1, 0}, point);
            cv::Size img_center = cv::Size{int(point[0] * res.cols), int(point[1] * res.rows)};
            cv::circle(res, img_center, 1, 0, 1, cv::FILLED);
        }
    }
    cv::imshow("swin", res);
    cv::waitKey(0);
    return 0;
}