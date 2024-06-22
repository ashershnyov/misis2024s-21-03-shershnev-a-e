#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <map>
#include <matplot/matplot.h>
#include <algorithm>
#include <set>

#include "tsne/tsne.h"


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
    int folders = 10, files = 100;

    cv::Mat data;
    std::vector<cv::Mat> imgs;
    for (int folder_ = 0; folder_ < folders; folder_++) {
        char current_path[255];
        int n = 0;
        n = sprintf(current_path, "%s%i/", test_root.c_str(), folder_);
        for (int file_ = 1; file_ < files; file_++) {
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

    for (int k = 1; k < 11; k += 2) {
        float tsne_error = calc_prediction_q(TSNE.res, col, k);
        std::cout << tsne_error << " ";
    }

    // std::cout << TSNE.res << "\n";
    std::vector<std::pair<float, float>> points;
    std::vector<float> points_x, points_y;
    for (int point = 0; point < TSNE.res.rows; point++) {
        cv::Vec2f cur_point = TSNE.res.at<cv::Vec2f>(point);
        points_x.push_back(cur_point[0]);
        points_y.push_back(cur_point[1]);
    }


    auto sc = matplot::scatter(points_x, points_y, 6, col);
    sc->marker_face(true);
    // matplot::show();
    // std::pair<float, float> g = calc_lcmc(data, TSNE.res, 50);
    // std::cout << g.first << " " << g.second << "\n\n";
    matplot::save("plot1.png");
    return 0;
}