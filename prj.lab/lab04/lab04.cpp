#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

cv::AdaptiveThresholdTypes gType = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
cv::ThresholdTypes gInverse = cv::THRESH_BINARY;
int gBlockSize = 5;
double gC = 0;
cv::Mat gSample, gBin;
std::string gWindowName = "window";

cv::Mat generate_sample(int circle_cnt, float size_min, float size_max,
                          uchar col_min, uchar col_max, float sigma) {
    cv::Mat sample(512, 512, 0, 10);
    float size_step = (size_max - size_min) / circle_cnt;
    int rows = sample.size().height / (size_max * 2) - 1;
    float col_step = (col_max - col_min) / rows;

    for (int row = 0; row < rows; row++) {
        int center_y = cvRound(sample.size().height / (rows) * row + size_max);
        float size_cur = size_min;

        for (int circle = 0; circle < circle_cnt; circle++) {
            int center_x = cvRound(sample.size().width / circle_cnt * circle + 2 * size_max);
            cv::Point circle_center = cv::Point(center_x,center_y);

            cv::Size circle_size = cv::Size(size_cur, size_cur);

            cv::ellipse(sample, circle_center, circle_size, 0, 0, 360, col_min, cv::FILLED);

            size_cur += size_step;
        }

        col_min += col_step;
    }

    cv::GaussianBlur(sample, sample, cv::Size(11, 11), sigma);
    return sample;
}

cv::Mat treshold(const cv::Mat input, cv::AdaptiveThresholdTypes type, cv::ThresholdTypes inverse,
                 int block_size, double c) {
    cv::Mat bin(input.size(), input.type());
    cv::adaptiveThreshold(input, bin, 255, type, inverse, block_size, c);
    return bin;
}

cv::Mat detect(const cv::Mat bin_img) {
    cv::Mat detected = bin_img.clone();
    // cv::GaussianBlur(detected, detected, cv::Size(5, 5), 0);

    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(bin_img, labels, stats, centroids, 8);

    cv::Mat mask = cv::Mat(detected.size(), 0, 0.0);
    for (int i = 1; i < stats.rows; i++) {
        mask += (labels == i);
    }

    cv::Mat contours;
    mask = (mask > 0) * 255;
    cv::Canny(mask, contours, 0, 255);
    cv::cvtColor(detected, detected, cv::COLOR_GRAY2RGB);
    for (int x = 0; x < contours.rows; x++) {
        for (int y = 0; y < contours.cols; y++) {
            if (contours.at<uchar>(x, y) > 0) {
                detected.at<cv::Vec3b>(x, y) = cv::Vec3b(255, 0, 255);
            }
        }
    }

    return detected;
}

void draw_frame(const std::string window_name, const cv::Mat input, const cv::Mat output) {
    cv::Mat concat_img;
    cv::cvtColor(input, concat_img, cv::COLOR_GRAY2RGB);
    cv::Mat detected_img = detect(output);
    cv::hconcat(concat_img, detected_img, concat_img);
    cv::imshow(window_name, concat_img);
}

void change_type_adaptive(int pos, void*) {
    if (pos == 1) 
        gType = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
    else
        gType = cv::ADAPTIVE_THRESH_MEAN_C;
    gBin = treshold(gSample, gType, gInverse, gBlockSize, gC);
    draw_frame(gWindowName, gSample, gBin);
}

void change_inverse(int pos, void*) {
    if (pos == 0) 
        gInverse = cv::THRESH_BINARY;
    else
        gInverse = cv::THRESH_BINARY_INV;
    gBin = treshold(gSample, gType, gInverse, gBlockSize, gC);
    draw_frame(gWindowName, gSample, gBin);
}

void change_block_size(int pos, void*) {
    if (pos <= 3) {
        gBlockSize = 3;
    }
    else if (pos % 2 == 0) 
        gBlockSize = pos + 1;
    else
        gBlockSize = pos;
    gBin = treshold(gSample, gType, gInverse, gBlockSize, gC);
    draw_frame(gWindowName, gSample, gBin);
}

void change_constant(int pos, void*) {
    // TODO: это костыль, потому что генерится мат unsigned,
    // как будто бы не оч правильно так делать - коэф. в обратную сторону работает
    gC = -float(pos) / 10.;
    gBin = treshold(gSample, gType, gInverse, gBlockSize, gC);
    draw_frame(gWindowName, gSample, gBin);
}

void change_type(int pos, void*) {
    // gBin = treshold();
    // draw_frame(gWindowName, gSample, gBin);
}

void create_window(int type) {
    cv::destroyWindow(gWindowName);
    cv::namedWindow(gWindowName);
    // cv::createTrackbar("General treshold type", gWindowName, &type, 1);
    // Для адаптивной бинаризации
    if (type == 0) {
        int c = gC * 20;
        int inv = int(gInverse), atype = int(gType), bsize = gBlockSize;
        cv::createTrackbar("Treshold type", gWindowName, &atype, 1, change_type_adaptive);
        cv::createTrackbar("Inverse", gWindowName, &inv, 1, change_inverse);
        cv::createTrackbar("Block size", gWindowName, &bsize, 20, change_block_size);
        cv::createTrackbar("Constant", gWindowName, &c, 100, change_constant);
    }
    // Для обычной бинаризации
    else if (type == 1) {
        gInverse = cv::THRESH_BINARY;
        int rtype = int(gInverse);
        cv::createTrackbar("Treshold type", gWindowName, &rtype, 1, change_type);
    }

}

int main() {
    int c = gC * 20;
    int inv = int(gInverse), type = int(gType), bsize = gBlockSize;
    gSample = generate_sample(6, 10, 20, 30, 127, 20);
    gBin = gSample.clone();
    create_window(0);

    gBin = treshold(gSample, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 0);

    draw_frame(gWindowName, gSample, gBin);

    cv::waitKey(0);
    return 0;
}