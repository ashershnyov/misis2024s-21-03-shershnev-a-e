#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

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

std::vector<std::vector<double>> ious_fill(const std::vector<cv::Mat> masks) {
    std::vector<std::vector<double>> iou_matrix(masks.size(), std::vector<double>(true_circles.size(), 0.0));
    for (int i = 0; i < masks.size(); i++) {
        for (int j = 0; j < true_circles.size(); j++) {
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

void draw_detection(cv::Mat& img, const std::vector<cv::Vec3f> circles) {
    for (cv::Vec3f circle: circles) {
        cv::circle(img, cv::Point(circle[0], circle[1]), circle[2], cv::Vec3b{255, 0, 255}, 2);
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
    draw_detection(converted, circles);

    return converted;
}

int main() {
    cv::Mat gSample = generate_sample(6, 10, 20, 30, 127, 20);
    extract_data_from_json("true.json");
    cv::Mat detected = detect_hough(gSample, 12, 3, 100, 35, 50);
    cv::Mat concat_img;
    cv::cvtColor(gSample, concat_img, cv::COLOR_GRAY2RGB);
    cv::hconcat(concat_img, detected, concat_img);
    cv::imshow("hough", concat_img);
    cv::waitKey(0);
    return 0;
}