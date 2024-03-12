#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/fast_math.hpp>

#include <iostream>

cv::Mat generate_sample(const cv::Scalar colors) {
    // Принимает на вход скаляр с 3 цветами
    // Первый цвет - цвет фона
    // Второй цвет - цвет внутреннего квадрата
    // Третий цвет - цвет круга
    cv::Mat img = cv::Mat(256, 256, 0, colors[0]);
    cv::Size img_dim = img.size();

    uchar rect_side = 209;
    uchar point_coord = (img_dim.width - rect_side) / 2;

    cv::Point top_left = cv::Point(point_coord, point_coord);
    cv::Point bottom_right = cv::Point(img_dim.width - point_coord, img_dim.width - point_coord);
    cv::rectangle(img, top_left, bottom_right, colors[1], cv::FILLED);

    uchar circle_rad = 83;

    cv::Point circle_center = cv::Point(img_dim.width / 2, img_dim.width / 2);
    cv::Size thicc = cv::Size(circle_rad, circle_rad);
    cv::ellipse(img, circle_center, thicc, 0, 0, 360, colors[2], cv::FILLED);
    return img;
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

cv::Mat add_noise(const cv::Mat img, const int stddev) {
    cv::Mat_<int> noise(img.size());
    cv::randn(noise, 0, stddev);
    cv::Mat img_tmp = img.clone();
    img_tmp += noise;
    return img_tmp;
}

void calc_mean_stddev(const cv::Mat img, const cv::Mat mask) {
    // TODO: Переиспользуемый код, надо бы с ним чего-то придумать (вынести в функцию)
    cv::Mat hist;
    int hist_size = 256;
    float range[] = {0, 256};
    const float* hist_range[] = {range};
    cv::calcHist(&img, 1, 0, mask, hist, 1, &hist_size, hist_range, true, false);

    double mean = 0;
    int cnt = cv::sum(hist)[0];
    for (int i = 0; i < hist.size().height; i++) {
        mean += i * hist.at<float>(i);
    }
    mean /= cnt;

    double stddev = 0;
    for (int i = 0; i < hist.size().height; i++) {
        stddev += hist.at<float>(i) * (i - mean) * (i - mean);
    }
    stddev /= cnt;

    printf("%.2f %.2f\n", mean, cv::sqrt(stddev));
}

int main(int argc, char* argv[]) {
    cv::Scalar color_arr[] = {cv::Scalar(0, 127, 255), cv::Scalar(20, 127, 235),
                              cv::Scalar(55, 127, 200), cv::Scalar(90, 127, 165)};
    int stddevs_arr[] = {3, 7, 15};

    const cv::Mat kMaskSquareOuter = generate_sample(cv::Scalar(255, 0, 0)),
                  kMaskSquareInner = generate_sample(cv::Scalar(0, 255, 0)),
                  kMaskCircle = generate_sample(cv::Scalar(0, 0, 255));

    std::vector<std::vector<float>> stats_table(4 * 3, std::vector<float>(2 * 4, 0.0));

    // Вероятно этот весь кусок ниже можно было бы куда элегантнее написать
    std::vector<cv::Mat> res_images, sample_images;
    for (cv::Scalar colors: color_arr) {
        std::vector<cv::Mat> noised_images;
        cv::Mat img_hist;
        cv::Mat img = generate_sample(colors);
        printf("For colors %.0f %.0f %.0f\n", colors[0], colors[1], colors[2]);
        for (int stddev: stddevs_arr){
            cv::Mat noised_img = add_noise(img, stddev);
            printf("Real deviation %d\n", stddev);
            calc_mean_stddev(noised_img, kMaskSquareOuter);
            calc_mean_stddev(noised_img, kMaskSquareInner);
            calc_mean_stddev(noised_img, kMaskCircle);
            printf("\n");

            cv::Mat hist_image = generate_hist(noised_img);
            cv::vconcat(noised_img, hist_image, noised_img);
            noised_images.push_back(noised_img);
        }
        printf("\n");
        cv::vconcat(noised_images, img_hist);
        res_images.push_back(img_hist);
        sample_images.push_back(img);
    }

    cv::Mat samples;
    cv::hconcat(sample_images, samples);

    cv::Mat res_img;
    cv::hconcat(res_images,res_img);
    cv::vconcat(samples, res_img, res_img);

    cv::imshow("lab02", res_img);
    // cv::imwrite("../prj.lab/lab02/res.png", res_img);
    cv::waitKey(0);
    return 0;
}