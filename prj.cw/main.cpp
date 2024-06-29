#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <map>
#include <matplot/matplot.h>
#include <algorithm>
#include <set>
#include <fstream>

#include "tsne/tsne.h"

int gPerClassPoints = 100;
int gClasses = 10;

void write_errors(std::vector<float> errors, std::string filename) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(3);
    file << "K=1:        " << errors[0] << "\n";
    file << "K=3:        " << errors[1] << "\n";
    file << "K=5:        " << errors[2] << "\n";
    file << "K=7:        " << errors[3] << "\n";
    file << "K=9:        " << errors[4] << "\n";
    file.close();
}

cv::Mat draw_poins(const cv::Mat_<float> points, std::vector<cv::Vec4b> colors, int h, int w) {
    if (points.cols != 2) {
        throw("points must be 2-dim");
    }
    cv::Mat img(h, w, CV_8UC3, cv::Scalar{255, 255, 255});
    double min_x = 0, min_y = 0, max_x = 0, max_y;
    cv::minMaxLoc(points.col(0), &min_x, &max_x);
    cv::minMaxLoc(points.col(1), &min_y, &max_y);

    cv::Mat alphas = cv::Mat::zeros(h, w, CV_32F);
    float alpha = 1.0 / gPerClassPoints * 10;
    for(int i = 0; i < points.rows; i++) {
        cv::Vec2f point = points.row(i);
        int point_x = int(w * (point[0] - min_x) / (max_x - min_x));
        int point_y = h - int(h * (point[1] - min_y) / (max_y - min_y));
        alphas.at<float>(point_x, point_y) += alpha;
    }
    double min_alpha, max_alpha;
    cv::minMaxLoc(alphas, &min_alpha, &max_alpha);
    alpha /= max_alpha;
    for (int i = 0; i < points.rows; i++) {
        cv::Vec2f point = points.row(i);
        int point_x = int(w * (point[0] - min_x) / (max_x - min_x));
        int point_y = h - int(h * (point[1] - min_y) / (max_y - min_y));

        cv::Mat tmp_to_add = img.clone();

        cv::circle(tmp_to_add, cv::Point{point_x, point_y}, h / 128, cv::Vec4f(colors[i / gPerClassPoints]), cv::FILLED, cv::FILLED);
        cv::addWeighted(tmp_to_add, alpha, img, 1 - alpha, 0.0, img);
    }
    return img;
}

// считает евклидовы расстояния между точками для knn
cv::Mat_<float> calc_dist_mat(const cv::Mat_<float> points) {
    int points_count = points.rows;
    cv::Mat_<float> dists(points_count, points_count);
    for (int i = 0; i < points_count - 1; i++) {
        for (int j = i; j < points_count; j++) {
            dists[i][j] = norm(points.row(i), points.row(j), cv::NORM_L2);
            dists[j][i] = dists[i][j];
        }
    }
    return dists;
}

cv::Mat_<float> calc_rank_mat(const cv::Mat_<float> dists) {
    int points_count = dists.rows;
    cv::Mat_<int> ranks = cv::Mat_<int>::zeros(dists.size());
    for (int i = 0; i < points_count; i++) {
        cv::Mat dist_row = dists.row(i);
        cv::Mat sorted_dists;
        cv::sortIdx(dist_row, sorted_dists, cv::SortFlags());
        for (int j = 1; j < points_count; j++) {
            int current_rank_idx = sorted_dists.at<int>(j);
            ranks.at<int>(i, current_rank_idx) = j;
            // ranks.at<int>(current_rank_idx, i) = j;
        }
    }
    return ranks;
}

cv::Mat calc_co_rank_matrix(const cv::Mat_<int> hd_ranks, const cv::Mat_<int> ld_ranks) {
    int points_count = hd_ranks.rows;
    cv::Mat co_rank_mat = cv::Mat_<int>::zeros(hd_ranks.size());
    for (int i = 0; i < points_count; i++) {
        cv::Mat ld_row = ld_ranks.row(i);
        cv::Mat hd_row = hd_ranks.row(i);
        for (int j = 0; j < points_count; j++) {
            int ld_rank = ld_ranks.at<int>(i, j), hd_rank = hd_ranks.at<int>(i, j);
            for (int k = 0; k < ld_row.cols; k++) {
                if (ld_row.at<int>(k) == j) {
                    ld_rank = k;
                }
            }
            for (int k = 0; k < hd_row.cols; k++) {
                if (hd_row.at<int>(k) == j) {
                    hd_rank = k;
                }
            }
            co_rank_mat.at<int>(ld_rank, hd_rank) += 1;
        }
    }
    return co_rank_mat;
}

// float calc_kn_metric(const cv::Mat co_ranks, int k) {
//     int points_cout = co_ranks.rows;
//     float kn = 0.0;
//     for (int i = 0; i < points_cout; i++) {
//         float kn_tmp = 0.0;
//         for (int j = 0; j < k; j++) {
//             kn_tmp += co_ranks.at<int>(j, i);
//         }
//         kn += kn_tmp;
//     }
//     return kn / float(k * points_cout);
// }

// float calc_kn_metric(const cv::Mat_<int> ld_ranks, const cv::Mat_<int> hd_ranks, int k) {
//     int points_count = ld_ranks.rows;
//     float kn = 0;
//     for (int i = 0; i < points_count; i++) {
//         cv::Mat_<int> ld_row = ld_ranks.row(i);
//         cv::Mat_<int> hd_row = hd_ranks.row(i);
//         cv::Mat_<int> ld_sorted, hd_sorted;
//         cv::sortIdx(hd_row, hd_sorted, cv::SortFlags());
//         cv::sortIdx(ld_row, ld_sorted, cv::SortFlags());
//         std::set<int> ld_set, hd_set;
//         for (int j = 1; j <= k; j++) {
//             ld_set.insert(ld_row.at<int>(j));
//             hd_set.insert(hd_row.at<int>(j));
//         }
//         std::set<int> intersect;
//         std::set_intersection(ld_set.begin(), ld_set.end(), hd_set.begin(), hd_set.end(), std::inserter(intersect, intersect.begin()));
//         kn += intersect.size();
//     }
//     return kn / points_count;
// }

// std::pair<float, float> calc_lcmc(const cv::Mat hd_data, const cv::Mat ld_data, int k) {
//     cv::Mat dists_hd = calc_dist_mat(hd_data);
//     cv::Mat dists_ld = calc_dist_mat(ld_data);
//     cv::Mat hd_ranks = calc_rank_mat(dists_hd);
//     cv::Mat ld_ranks = calc_rank_mat(dists_ld);
//     cv::Mat co_ranks = calc_co_rank_matrix(hd_ranks, ld_ranks);
//     // float kn = calc_kn_metric(co_ranks, k);
//     float kn = calc_kn_metric(ld_ranks, hd_ranks, k);
//     return std::pair<float, float>{kn - float(k) / (hd_data.rows - 1), kn};
// }

// knn для классификации точек
int determine_class(const cv::Mat_<float> dists, const int k, const int point_idx, const std::vector<int> classes) {
    if (k > dists.rows - 1) {
        throw "K should not be greater than the remaining number of points";
    }
    if (k % 2 == 0) {
        throw "K should be odd";
    }
    if (point_idx > dists.rows) {
        throw "Give point index should not be greater than the number of points";
    }
    cv::Mat point_row = dists.row(point_idx);
    cv::Mat sorted_row;
    cv::sortIdx(point_row, sorted_row, cv::SortFlags());
    std::map<int, int> oc;
    int max_cnt = 0, max_cnt_idx = -1;
    for (int i = 1; i <= k; i++) {
        int cur_idx = sorted_row.at<int>(i);
        if (!(oc.count(classes[cur_idx]) > 0)) {
            oc[classes[cur_idx]] = 1;
            if (max_cnt == 0) {
                max_cnt = 1;
                max_cnt_idx = classes[cur_idx];
            }
        }
        else {
            oc[classes[cur_idx]]++;
            if (oc[classes[cur_idx]] > max_cnt) {
                max_cnt = oc[classes[cur_idx]];
                max_cnt_idx = classes[cur_idx];
            }
        }
    }
    return max_cnt_idx;
}

// сжимает данные методом главных компонент до num_components измерений
cv::Mat pca(cv::Mat& data, int num_components) {
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, num_components);
    cv::Mat reduced_data;
    pca.project(data, reduced_data);
    return reduced_data;
}

// Вычисляет процент корректных предсказаний на основе 1-соседного knn
float calc_prediction_q(const cv::Mat &data, std::vector<int> classes, int k) {
    int points_count = data.rows;
    if (classes.size() != points_count) {
        throw("Classes were not given for all of the points");
    }
    int correct_guesses = 0;
    cv::Mat_<float> dists_mat = calc_dist_mat(data);
    for (int i = 0; i < data.rows; i++) {
        int cls = determine_class(dists_mat, k, i, classes);
        if (cls == classes[i]) {
            correct_guesses++;
        }
    }
    return float(correct_guesses) / points_count;
}

int main(int argc, char *argv[]) {
    std::vector<int> col;
    std::vector<float> class_col;
    std::vector<cv::Vec4b> colors = {
        cv::Vec4b{230, 159, 0},
        cv::Vec4b{86, 180, 233},
        cv::Vec4b{0, 158, 115},
        cv::Vec4b{240, 228, 66},
        cv::Vec4b{0, 114, 178}, 
        cv::Vec4b{204, 121, 167},
        cv::Vec4b{178, 34, 34},
        cv::Vec4b{34, 139, 34},
        cv::Vec4b{53, 80, 93},
        cv::Vec4b{0, 0, 0}
    };

    cv::CommandLineParser parser(argc, argv,
        "{n |2  |}"
        "{lr|5  |}"
        "{px|2.5|}"
        "{g |0.4|}"
    );
    int n = parser.get<int>("n");
    double learning_rate = parser.get<double>("lr");
    double perplexity = parser.get<double>("px");
    double gamma = parser.get<double>("g");

    std::string test_root = "../prj.cw/train_set/";

    int dim = 28;

    cv::Mat data;
    std::vector<cv::Mat> imgs;
    for (int folder_ = 0; folder_ < gClasses; folder_++) {
        char current_path[255];
        int n = 0;
        class_col.push_back(folder_);
        n = sprintf(current_path, "%s%i/", test_root.c_str(), folder_);
        for (int file_ = 1; file_ < gPerClassPoints; file_++) {
            col.push_back(folder_);
            char file_path[255];
            n = sprintf(file_path, "%s%i.png", current_path, file_);
            cv::Mat sample;
            sample = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
            imgs.push_back(sample.reshape(1, 1));
        }
    }
    cv::vconcat(imgs, data);
    data /= 255.;

    for (int k = 1; k < 11; k += 2) {
        float orig_error = calc_prediction_q(data, col, k);
        std::cout << orig_error << " ";
    }
    std::cout << "\n";

    cv::Mat reduced_data = pca(data, 30);

    for (int k = 1; k < 11; k += 2) {
        float reduced_error = calc_prediction_q(reduced_data, col, k);
        std::cout << reduced_error << " ";
    }

    std::cout << "\n";

    tsne TSNE(reduced_data, n, perplexity, learning_rate, gamma);

    std::vector<float> errors;
    for (int k = 1; k < 11; k += 2) {
        float tsne_error = calc_prediction_q(TSNE.res, col, k);
        errors.push_back(tsne_error);
    }
    write_errors(errors, "tsne_error.txt");

    // std::cout << TSNE.res << "\n";
    std::vector<std::pair<float, float>> points;
    std::vector<double> points_x, points_y;
    class_col = std::vector<float>();
    matplot::hold(matplot::on);
    for (int i = 0; i < gClasses; i++) {
        class_col = std::vector<float>();
        points_x = std::vector<double>();
        points_y = std::vector<double>();
        for (int point = i * gPerClassPoints; point < (i + 1) * gPerClassPoints; point++) {
            class_col.push_back(i);
            cv::Vec2f cur_point = TSNE.res.at<cv::Vec2f>(point);
            points_x.push_back(cur_point[0]);
            points_y.push_back(cur_point[1]);
        }
        // matplot::scatter(points_x, points_y)->marker_face(true).use_y2(true);
        auto sc = matplot::scatter(points_x, points_y, std::vector<double>{});
        sc->marker_face(true);
        sc->marker_color({float(colors[i][0]), float(colors[i][1]), float(colors[i][2])});
        sc->marker_face_color({float(colors[i][0]), float(colors[i][1]), float(colors[i][2])});
    }
    matplot::hold(matplot::off);
    
    // for (int point = 0; point < TSNE.res.rows; point++) {
    //     cv::Vec2f cur_point = TSNE.res.at<cv::Vec2f>(point);
    //     points_x.push_back(cur_point[0]);
    //     points_y.push_back(cur_point[1]);
    // }

    cv::Mat img = draw_poins(TSNE.res, colors, 1024, 1024);
    // cv::Mat_<float> po = (cv::Mat_<float>(6, 2) << (-2, -2, 2, 2, 2, -2, -2, 2, 0, 0, 1, 1));
    // cv::Mat img = draw_poins(po, colors, 1024, 1024);
    cv::imshow("img", img);
    cv::imwrite("tsne.png", img);

    matplot::legend({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})
        ->location(matplot::legend::general_alignment::topleft);

    // auto sc = matplot::scatter(points_x, points_y, std::vector<double>{}, col);
    // sc->marker_face(true);

    // std::pair<float, float> g = calc_lcmc(data, TSNE.res, 50);
    // std::cout << g.first << " " << g.second << "\n\n";
    matplot::save("plot1.png");
    for (int k = 1; k < 11; k += 2) {
        float tsne_error = calc_prediction_q(TSNE.res, col, k);
        std::cout << tsne_error << " ";
    }
    cv::waitKey(0);
    return 0;
}