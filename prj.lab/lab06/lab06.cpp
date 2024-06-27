#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <set>

std::vector<std::tuple<cv::Point, int, uchar>> true_circles;
std::vector<cv::Mat> true_masks;

std::vector<std::pair<float, float>> froc_curve_points;

float gFrocArea = 0.0;

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

    cv::Mat_<int> noise(sample.size());
    cv::randn(noise, 0, 5);
    sample += noise;

    return sample;
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

std::vector<float> ious_fill(const std::vector<cv::Mat> masks) {
    std::vector<float> ious(masks.size());
    for (int i = 0; i < masks.size(); i++) {
        float max_iou = 0;
        for (int j = 0; j < true_masks.size(); j++) {
            float iou = calc_iou(masks[i], true_masks[j]);
            if (iou > max_iou) max_iou = iou;
        }
        ious[i] = max_iou;
    }
    return ious;
}

void draw_detection(cv::Mat& img, const std::vector<cv::Vec3f> circles) {
    for (cv::Vec3f circle: circles) {
        cv::circle(img, cv::Point(circle[0], circle[1]), circle[2], cv::Vec3b{255, 0, 255}, 2);
    }
}

float calc_area() {
    float area = 0.0;
    for (int i = 1; i < froc_curve_points.size(); i++) {
        area += ((froc_curve_points[i].second - froc_curve_points[i - 1].second)
                 * (froc_curve_points[i].first + froc_curve_points[i - 1].first)
                 / 2.f);
    }
    return area;
}

std::vector<float> calc_det_scores(const std::vector<cv::Vec3f> circles) {
    std::vector<float> scores(circles.size());
    for (int i = 0; i < circles.size(); i++) {
        float sc = 1.f;
        std::vector<float> w;
        for (int j = 0; j < circles.size(); j++) {
            if (j == i) continue;
            float we = cv::abs(float(circles[i][2]) - float(circles[j][2])) * (-1.) / cv::max(circles[i][2], circles[j][2]);
            w.push_back(we);
        }
        std::sort(w.begin(), w.end());
        for (int j = 0; j < w.size(); j++) {
            sc += w[j] / cv::pow(3, j + 1);
        }
        scores[i] = sc;
    }
    return scores;
}

void froc(std::vector<float> ious, int det_num, std::vector<float> scores, float iou_threshold) {
    float fp_total = 0, tp_ratio = 0;
    std::set<float> scores_set(scores.begin(), scores.end());
    std::vector<float> sorted_scores(scores_set.begin(), scores_set.end());
    std::vector<float> thresholds(sorted_scores.size());
    thresholds[0] = -0.1;
    for (int i = 1; i < thresholds.size(); i++) {
        thresholds[i] = (sorted_scores[i - 1] + sorted_scores[i]) / 2.f;
    }
    thresholds.push_back(1.1);
    std::sort(thresholds.rbegin(), thresholds.rend());

    for (float threshold: thresholds) {
        int tp = 0, fp = 0;
        for (int i = 0; i < det_num; i++) {
            if (scores[i] < threshold) {
                continue;
            }
            if (ious[i] > iou_threshold) {
                tp += 1;
                continue;
            }
            fp += 1;
        }
        // std::cout << "X: " << fp << " Y: " << float(tp) / det_num << "\n\n";
        froc_curve_points.push_back(std::pair<float, float>{fp, float(tp) / det_num});
    }
}

cv::Mat detect_hough(cv::Mat img, int dist_denominator = 16, int min_radius = 3,
                     int max_radius = 20, int p1 = 35, int p2 = 50) {
    cv::Mat detected = img.clone();
    int denoise = 7;
    cv::GaussianBlur(detected, detected, cv::Size(denoise, denoise), 0);

    std::vector<cv::Vec3f> circles;
    float distance = detected.rows / dist_denominator;
    cv::HoughCircles(detected, circles, cv::HOUGH_GRADIENT, 2, distance, p1, p2, min_radius, max_radius);

    cv::Mat converted;
    cv::cvtColor(img, converted, cv::COLOR_GRAY2RGB);

    std::vector<cv::Mat> masks;
    for (auto circle: circles) {
        cv::Mat mask(img.size(), CV_8U);
        cv::circle(mask, cv::Point(circle[0], circle[1]), circle[2], 255, cv::FILLED);
        masks.push_back(mask);
    }

    cv::imwrite("lab06_mask_one.png", masks[0]);

    std::vector<float> scores = calc_det_scores(circles);
    std::vector<float> ious = ious_fill(masks);
    froc(ious, masks.size(), scores, 0.82);

    return converted;
}

int main() {
    // cv::Mat gSample = generate_sample(6, 10, 20, 30, 127, 20);
    cv::Mat3b sample = cv::imread("./../assets/lab04/lab04_1_s.png");
    std::vector<cv::Mat1b> chan;
    cv::split(sample, chan);

    cv::Mat1b gSample = chan[0];

    extract_data_from_json("./../assets/lab04/lab04_1.json");
    cv::Mat detected = detect_hough(gSample, 12, 3, 100, 35, 50);
    cv::Mat concat_img;
    cv::cvtColor(gSample, concat_img, cv::COLOR_GRAY2RGB);
    cv::hconcat(concat_img, detected, concat_img);
    cv::imwrite("lab06_5.png", concat_img);
    std::cout << "X: [";
    for (int i = 0; i < froc_curve_points.size(); i++) {
        std::cout << froc_curve_points[i].first << ", ";
    }
    std::cout << "]" << std::endl << "Y: [";
    for (int i = 0; i < froc_curve_points.size(); i++) {
        std::cout << froc_curve_points[i].second << ", ";
    }
    std::cout << "]";
    // cv::waitKey(0);
    return 0;
}