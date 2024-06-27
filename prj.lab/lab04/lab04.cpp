#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <sstream>

cv::AdaptiveThresholdTypes gType = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
cv::ThresholdTypes gInverse = cv::THRESH_BINARY;
int gBlockSize = 167;
double gC = -7.1;
cv::Mat gSample, gBin;
std::string gWindowName = "window";

int gMinSize = 5, gMaxSize = 40;
int gDenoise = 7;

std::vector<std::tuple<cv::Point, int, uchar>> true_circles;

std::vector<cv::Mat> true_masks;

void extract_data_from_json(const std::string fname){
    cv::FileStorage json(fname, 0);
    cv::FileNode root = json["data"];
    cv::FileNode objects = root["objects"];
    cv::FileNode bg = root["background"]["size"];

    for(int i = 0; i < objects.size(); i++){
        cv::FileNode circ = objects[i]["p"];
        cv::FileNode col = objects[i]["c"];

        cv::Mat true_mask = cv::Mat(cv::Size(bg[0].real(), bg[1].real()), 0, 0.0);
        cv::Point circle_center = cv::Point{(int)circ[0].real(),
                                            (int)circ[1].real()};
        cv::Size circle_size = cv::Size(circ[2].real(), circ[2].real());
        cv::ellipse(true_mask, circle_center, circle_size, 0, 0, 360, 255, cv::FILLED);
        true_masks.push_back(true_mask);
    }

    json.release();
}

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

            true_circles.push_back(std::tuple<cv::Point, int, uchar>(circle_center, size_cur, col_min));

            size_cur += size_step;
        }

        col_min += col_step;
    }

    cv::FileStorage true_json("lab04_0.json", cv::FileStorage::WRITE);
    true_json << "data" << "{";
    true_json << "objects" << "[";
    for(std::tuple<cv::Point, int, uchar> circle: true_circles) {
        true_json << "{";
        true_json << "p" << std::vector<int>{std::get<0>(circle).x, std::get<0>(circle).y, std::get<1>(circle)};
        // true_json << std::get<1>(circle);
        true_json << "c" << std::get<2>(circle);
        true_json << "}";
    }
    true_json << "]";
    true_json << "background" << "{";
    true_json << "size" << "[" << sample.rows << sample.cols << "]";
    true_json << "}";
    true_json.release();

    // cv::GaussianBlur(sample, sample, cv::Size(5, 5), sigma);

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
    std::vector<std::vector<double>> iou_matrix(masks.size(), std::vector<double>(true_masks.size(), 0.0));
    for (int i = 0; i < masks.size(); i++) {
        for (int j = 0; j < true_masks.size(); j++) {
            iou_matrix[i][j] = calc_iou(masks[i], true_masks[j]);
        }
    }
    return iou_matrix;
}

void calc_stats(const std::vector<std::vector<double>> iou_matrix,
                const double treshold, int &TP, int &FP, int &FN) {

    if (iou_matrix.size() == 0)
        return;
    if (iou_matrix[0].size() == 0)
        return;

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

void calc_per_pixel_stats(const cv::Mat &det, const cv::Mat &truth, int &tp, int &fp, int &fn) {
    for (int i = 0; i < det.rows; i++) {
        for (int j = 0; j < det.cols; j++) {
            if (truth.at<uchar>(j, i) > 0 && det.at<uchar>(j, i) > 0) {
                tp++;
            } else if (truth.at<uchar>(j, i) == 0 && det.at<uchar>(j, i) > 0) {
                fp++;
            } else if (truth.at<uchar>(j, i) > 0 && det.at<uchar>(j, i) == 0) {
                fn++;
            }
        }
    }
}

void draw_detection(cv::Mat& img, const std::vector<cv::Mat> masks) {
    for (cv::Mat mask: masks) {
        std::vector<std::vector<cv::Point>> cont, to_draw;
        cv::findContours(mask, cont, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        int size_max = 0, size_max_idx = 0;
        for(int i = 0; i < cont.size(); i++) {
            if (cont[i].size() > size_max) {
                size_max = cont[i].size();
                size_max_idx = i;
            }
        }
        to_draw.push_back(cont[size_max_idx]);
        cv::Point2f enc_center;
        float enc_radius;
        cv::minEnclosingCircle(to_draw[0], enc_center, enc_radius);
        cv::circle(img, enc_center, enc_radius, cv::Vec3b{255, 0, 255}, 2);
    }
}

cv::Mat detect_connected_components(const cv::Mat bin_img) {
    cv::Mat detected = bin_img.clone();
    cv::GaussianBlur(detected, detected, cv::Size(gDenoise, gDenoise), 0);

    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(bin_img, labels, stats, centroids, 8);

    cv::Mat detect_mask = cv::Mat(detected.size(), 0, 0.0);
    std::vector<cv::Mat> masks;
    for (int i = 1; i < stats.rows; i++) {
        if (filter_components(stats.at<int>(i, cv::CC_STAT_AREA))){
            cv::Mat mask = (labels == i);
            detect_mask += (labels == i);
            masks.push_back(mask);
        }
    }

    std::vector<std::vector<double>> ious;
    ious = ious_fill(masks);

    int tp = 0, fn = 0, fp = 0;
    calc_stats(ious, 0.5, tp, fp, fn);

    std::cout << tp << " " << fp << " " << fn << "\n";
    cv::cvtColor(detected, detected, cv::COLOR_GRAY2RGB);

    tp = 0; fn = 0; fp = 0;
    cv::Mat det_masks = cv::Mat1b(detected.size(), 0);
    for(int i = 0; i < masks.size(); i++) {
        cv::bitwise_or(det_masks, masks[i], det_masks);
    }
    cv::Mat gt_masks = cv::Mat1b(detected.size(), 0);
    for(int i = 0; i < true_masks.size(); i++) {
        cv::bitwise_or(gt_masks, true_masks[i], gt_masks);
    }
    calc_per_pixel_stats(det_masks, gt_masks, tp, fp, fn);
    std::cout << tp << " " << fp << " " << fn << "\n\n";
    // cv::hconcat(gt_masks, det_masks, det_masks);
    cv::imwrite("lab07_04_5.png", det_masks);
    // cv::imwrite("lab06_compare_det.png", gt_masks);


    draw_detection(detected, masks);

    return detected;
}

cv::Mat detect_laplacian(const cv::Mat input) {
    cv::Mat detected = input.clone();
    // cv::normalize(detected, detected, 0, 255, cv::NORM_MINMAX);
    cv::GaussianBlur(detected, detected, cv::Size(gDenoise, gDenoise), 0);
    cv::Mat output;
    cv::Laplacian(detected, output, CV_32F, 11);
    double min, max;
    cv::minMaxLoc(output, &min, &max);
    std::cout << min << " " << max << "\n";
    output = output * 0.5 + 0.5 * max;
    cv::Mat3b cl = output.clone();
    cv::minMaxLoc(cl, &min, &max);
    std::cout << min << " " << max << "\n";
    return cl;
}

void draw_frame(const std::string window_name, const cv::Mat input, const cv::Mat output) {
    cv::Mat concat_img;
    cv::cvtColor(input, concat_img, cv::COLOR_GRAY2RGB);
    cv::Mat detected_c = detect_connected_components(output);
    // cv::Mat detected_l = detect_laplacian(concat_img);
    cv::hconcat(concat_img, detected_c, concat_img);
    // cv::hconcat(concat_img, detected_l, concat_img);
    cv::imshow(window_name, concat_img);
    cv::imwrite("lab04_0.png", concat_img);
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
    gSample = generate_sample(6, 10, 20, 30, 127, 20);
    cv::Mat3b sample = cv::imread("./../assets/lab04/lab04_5_s.png");
    std::vector<cv::Mat1b> chan;
    cv::split(sample, chan);

    cv::Mat1b gSample = chan[0];
    // cv::imwrite("lab04_1_s.png", gSample);
    extract_data_from_json("../prj.lab/lab04/test_data/lab04_5.json");
    gBin = gSample.clone();
    // create_window(0);

    gBin = treshold(gSample, gType, gInverse, gBlockSize, gC);

    draw_frame(gWindowName, gSample, gBin);
    cv::waitKey(0);

    return 0;
}