#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

cv::AdaptiveThresholdTypes gType = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
cv::ThresholdTypes gInverse = cv::THRESH_BINARY;
int gBlockSize = 143;
double gC = -10.9;
cv::Mat gSample, gBin;
std::string gWindowName = "window";

int gMinSize = 10, gMaxSize = 20;
int gDenoise = 7;

std::vector<std::tuple<cv::Point, int>> true_circles;
cv::Mat true_img;

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

            true_circles.push_back(std::tuple<cv::Point, int>(circle_center, size_cur));

            size_cur += size_step;
        }

        col_min += col_step;
    }

    // cv::GaussianBlur(sample, sample, cv::Size(11, 11), sigma);

    cv::Mat_<int> noise(sample.size());
    cv::randn(noise, 0, 5);
    sample += noise;

    return sample;
}

cv::Mat treshold(const cv::Mat input, cv::AdaptiveThresholdTypes type, cv::ThresholdTypes inverse,
                 int block_size, double c) {
    cv::Mat bin(input.size(), input.type());
    cv::adaptiveThreshold(input, bin, 255, type, inverse, block_size, c);
    return bin;
}

bool filter_components(const int size) {
    return (size > 0.9 * CV_PI * gMinSize * gMinSize && size < 1.1 * CV_PI * gMaxSize * gMaxSize);
}

cv::Mat gen_true_img(const cv::Size sample_size) {
    cv::Mat img = cv::Mat(sample_size, 0);
    for (auto& circle: true_circles) {
        int rad = std::get<1>(circle);
        cv::ellipse(img, std::get<0>(circle), cv::Size(rad, rad), 0, 0, 360, 255, cv::FILLED);
    }
    return img;
}

double calc_iou(const cv::Mat mask, const cv::Mat ref_mask) {
    double in, un;
    cv::Mat res;
    cv::bitwise_and(mask, ref_mask, res);
    in = cv::countNonZero(res);
    cv::bitwise_or(mask, ref_mask, res);
    un = cv::countNonZero(res);
    return double(in) / double(un);
}

std::vector<std::vector<double>> ious_fill(const std::vector<cv::Mat> masks) {
    cv::Mat labels;
    true_img = gen_true_img(cv::Size(512, 512));
    cv::connectedComponents(true_img, labels, 8);

    // cv::Mat res_mask = cv::Mat(cv::Size(512, 512), 0, 0.0);

    std::vector<std::vector<double>> iou_matrix(masks.size(), std::vector<double>(true_circles.size(), 0.0));
    for (int i = 0; i < masks.size(); i++) {
        for (int j = 0; j < true_circles.size(); j++) {
            cv::Mat mask = (labels == j + 1);
            // res_mask += (labels == j + 1);
            iou_matrix[i][j] = calc_iou(masks[i], mask);
        }
    }
    // cv::imshow("resmat", res_mask);
    return iou_matrix;
}

void calc_stats(const std::vector<std::vector<double>> iou_matrix,
                const double treshold, int &TP, int &FP, int &FN) {
    for (int i = 0; i < iou_matrix.size(); i++) {
        bool fp = 1;
        for (int j = 0; j < iou_matrix[0].size(); j++) {
            if (iou_matrix[i][j] > treshold) {
                fp = 0;
                break;
            }
        }
        if (fp)
            FP += 1;
    }

    for (int i = 0; i < iou_matrix[0].size(); i++) {
        bool tp = 0;
        for (int j = 0; j < iou_matrix.size(); j++) {
            if (iou_matrix[j][i] > treshold) {
                tp = 1;
                break;
            }
        }
        if (tp)
            TP += 1;
        else
            FN += 1;
    }
}

cv::Mat detect_connected_components(const cv::Mat bin_img) {
    cv::Mat detected = bin_img.clone();
    cv::GaussianBlur(detected, detected, cv::Size(gDenoise, gDenoise), 0);

    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(bin_img, labels, stats, centroids, 8);

    cv::Mat mask = cv::Mat(detected.size(), 0, 0.0);
    cv::Mat detect_mask = mask.clone();
    std::vector<cv::Mat> masks;
    for (int i = 1; i < stats.rows; i++) {
        if (filter_components(stats.at<int>(i, cv::CC_STAT_AREA))){
            mask = (labels == i);
            detect_mask += (labels == i);
            masks.push_back(mask);
        }
    }

    std::vector<std::vector<double>> ious;
    ious = ious_fill(masks);

    int tp = 0, fn = 0, fp = 0;
    calc_stats(ious, 0.5, tp, fp, fn);

    std::cout << tp << " " << fp << " " << fn << "\n";

    cv::Mat contours;
    detect_mask = (detect_mask > 0) * 255;
    cv::Canny(detect_mask, contours, 0, 255);
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

cv::Mat detect_laplacian(const cv::Mat input) {
    cv::Mat detected = input.clone();
    // cv::normalize(detected, detected, 255, cv::NORM_MINMAX);
    cv::GaussianBlur(detected, detected, cv::Size(gDenoise, gDenoise), 0);
    cv::Mat output;
    cv::Laplacian(detected, output, CV_16S, 11);
    return output;
}

void draw_frame(const std::string window_name, const cv::Mat input, const cv::Mat output) {
    cv::Mat concat_img;
    cv::cvtColor(input, concat_img, cv::COLOR_GRAY2RGB);
    cv::Mat detected_c = detect_connected_components(output);
    // cv::Mat detected_l = detect_laplacian(concat_img);
    cv::hconcat(concat_img, detected_c, concat_img);
    // cv::hconcat(concat_img, detected_l, concat_img);
    cv::imshow(window_name, concat_img);
    // cv::imshow(window_name, detected_l);
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

void change_min_size(int pos, void*) {
    gMinSize = pos;
    draw_frame(gWindowName, gSample, gBin);
}

void change_max_size(int pos, void*) {
    gMaxSize = pos;
    draw_frame(gWindowName, gSample, gBin);
}

void change_blur_core(int pos, void*) {
    if (pos <= 3) {
        gDenoise = 3;
    }
    else if (pos % 2 == 0) 
        gDenoise = pos + 1;
    else
        gDenoise = pos;
    gDenoise = fmin(gDenoise, 31);
    draw_frame(gWindowName, gSample, gBin);
}

void create_window(int type) {
    cv::destroyWindow(gWindowName);
    cv::namedWindow(gWindowName);
    // cv::createTrackbar("General treshold type", gWindowName, &type, 1);
    // Для адаптивной бинаризации
    if (type == 0) {
        int c = gC * 20;
        int inv = int(gInverse), atype = int(gType), bsize = gBlockSize;

        cv::createTrackbar("Treshold type", gWindowName, nullptr, 1, change_type_adaptive);
        cv::setTrackbarPos("Treshold type", gWindowName, atype);

        cv::createTrackbar("Inverse", gWindowName, nullptr, 1, change_inverse);
        cv::setTrackbarPos("Inverse", gWindowName, inv);

        cv::createTrackbar("Block size", gWindowName, nullptr, 200, change_block_size);
        cv::setTrackbarPos("Block size", gWindowName, bsize);

        cv::createTrackbar("Constant", gWindowName, nullptr, 300, change_constant);
        cv::setTrackbarPos("Constant", gWindowName, -c * 10.);
    }
    // Для обычной бинаризации
    else if (type == 1) {
        gInverse = cv::THRESH_BINARY;
        int rtype = int(gInverse);
        cv::createTrackbar("Treshold type", gWindowName, nullptr, 1, change_type);
        cv::setTrackbarPos("Treshold type", gWindowName, rtype);
    }
    int min_size = gMinSize, max_size = gMaxSize;
    int core = gDenoise;
    cv::createTrackbar("Denoising blur core", gWindowName, nullptr, 61, change_blur_core);
    cv::setTrackbarPos("Denoising blur core", gWindowName, core);

    cv::createTrackbar("Detect min size", gWindowName, nullptr, 100, change_min_size);
    cv::setTrackbarPos("Detect min size", gWindowName, min_size);

    cv::createTrackbar("Detect max size", gWindowName, nullptr, 200, change_max_size);
    cv::setTrackbarPos("Detect max size", gWindowName, max_size);
}

int main() {
    true_img = gen_true_img(cv::Size(512, 512));
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