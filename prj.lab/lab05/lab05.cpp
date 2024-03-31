#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main() {
    int rect_side = 99, rect_cnt = 3, rows = 2, circle_radius = 25;
    cv::Mat1b sample(rect_side * rows, rect_side * rect_cnt);
    std::vector<cv::Vec2b> colors{
        cv::Vec2b{0, 127},
        cv::Vec2b{127, 0},
        cv::Vec2b{255, 0},
        cv::Vec2b{255, 127},
        cv::Vec2b{0, 255},
        cv::Vec2b{127, 255}
    };
    for (int i = 0; i < rect_cnt; i++) {
        for (int j = 0; j < rows; j++) {
            cv::Vec2b color = colors[j * rect_cnt + i];

            cv::Point top_left = cv::Point(i * rect_side, j * rect_side);
            cv::Point bottom_right = cv::Point((i + 1) * rect_side, (j + 1) * rect_side);
            cv::rectangle(sample, top_left, bottom_right, color[0], cv::FILLED);

            cv::Point center = cv::Point(i * rect_side + rect_side / 2., j * rect_side + rect_side / 2.);
            cv::ellipse(sample, center, cv::Size(circle_radius, circle_radius), 0, 0, 360, color[1], cv::FILLED);
        }
    }
    cv::Mat1b sample_k1 = sample.clone(), sample_k2 = sample.clone();
    cv::Mat kernel1 = (cv::Mat1f(2, 2) << 1.0, 0.0, 0.0, -1.0);
    cv::filter2D(sample, sample_k1, -1, kernel1, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    cv::Mat kernel2 = (cv::Mat1f(2, 2) << 0.0, 1.0, -1.0, 0.0);
    cv::filter2D(sample, sample_k2, -1, kernel2, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    cv::Mat1f s1_tmp = sample_k1.clone(), s2_tmp = sample_k2.clone(), s3_tmp = s2_tmp.clone();

    cv::sqrt(s1_tmp.mul(s1_tmp) + s2_tmp.mul(s2_tmp), s3_tmp);

    std::vector<cv::Mat1f> channels{s1_tmp, s2_tmp, s3_tmp};
    cv::Mat3f res;
    cv::merge(channels, res);
    cv::imshow("res", res);

    cv::waitKey(0);
    return 0;
}