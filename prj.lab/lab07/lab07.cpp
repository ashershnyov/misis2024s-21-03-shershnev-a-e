#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

cv::AdaptiveThresholdTypes gType = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
cv::ThresholdTypes gInverse = cv::THRESH_BINARY;
int gBlockSize = 167;
double gC = -7.1;

cv::Mat extract_data_from_json(const std::string fname){
    cv::FileStorage json(fname, 0);
    cv::FileNode root = json["data"];
    cv::FileNode objects = root["objects"];
    cv::FileNode bg = root["background"]["size"];

    cv::Mat true_mask = cv::Mat(cv::Size(bg[0].real(), bg[1].real()), 0, 0.0);

    for(int i = 0; i < objects.size(); i++){
        cv::FileNode circ = objects[i]["p"];
        cv::FileNode col = objects[i]["c"];

        cv::Point circle_center = cv::Point{(int)circ[0].real(),
                                            (int)circ[1].real()};
        cv::Size circle_size = cv::Size(circ[2].real(), circ[2].real());
        cv::ellipse(true_mask, circle_center, circle_size, 0, 0, 360, 255, cv::FILLED);
    }

    json.release();
    return true_mask;
}

void calc_stats(const cv::Mat &seg, const cv::Mat &truth, int &tp, int &fp, int &fn) {
    for (int i = 0; i < seg.rows; i++) {
        for (int j = 0; j < seg.cols; j++) {
            if (truth.at<uchar>(j, i) > 0 && seg.at<uchar>(j, i) > 0) {
                tp++;
            } else if (truth.at<uchar>(j, i) == 0 && seg.at<uchar>(j, i) > 0) {
                fp++;
            } else if (truth.at<uchar>(j, i) > 0 && seg.at<uchar>(j, i) == 0) {
                fn++;
            }
        }
    }
}

int main() {
    cv::Mat sample = cv::imread("./../assets/lab04/lab04_5_s.png");
    cv::Mat truth = extract_data_from_json("./../assets/lab04/lab04_5.json");
    cv::imshow("truth", truth);
    std::vector<cv::Mat1b> chan;
    cv::Mat bin;
    cv::Mat segmented;
    cv::pyrMeanShiftFiltering(sample, segmented, 20, 40);
    cv::imwrite("segment.png", segmented);
    cv::split(segmented, chan);
    cv::adaptiveThreshold(chan[0], bin, 255, gType, gInverse, gBlockSize, gC);
    cv::merge(std::vector<cv::Mat>{bin, bin, bin}, bin);
    int tp = 0, fp = 0, fn = 0;
    calc_stats(bin, truth, tp, fp, fn);
    std::cout << tp << " " << fp << " " << fn << "\n";
    cv::imshow("seg", bin);
    cv::imwrite("lab07_5.png", bin);
    cv::waitKey(0);
}