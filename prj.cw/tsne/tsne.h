#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <random>
#include <queue>
 
static float sigmaMin = 0, sigmaMax = 100;
static float kEps = 0.01, kTol = 1.0e-7;
static float alpha = 0.66;
static cv::Mat learning_rate;
 
class tsne {

    private:

        cv::Mat calc_softmax(const cv::Mat& mat, int idx) {

            cv::Mat exp_mat(mat.size(), CV_64F);
            cv::exp(-mat, exp_mat);
            exp_mat.at<float>(idx) = 0;
            float sum_exp = cv::sum(exp_mat)[0];
            cv::Mat softmax_mat;
            cv::divide(exp_mat, sum_exp, softmax_mat);

            return softmax_mat.clone();
        }

        float calc_entropy(const cv::Mat& mat) {
            cv::Mat log_mat;

            cv::log(mat, log_mat);
            log_mat *= cv::log(2);
            cv::Mat entropy_mat;
            cv::multiply(mat, log_mat, entropy_mat);
            float entropy = -cv::sum(entropy_mat)[0];

            return entropy;
        }

        // Расчитывает вероятностные распределения для каждой точки
        cv::Mat prob_distributions(cv::Mat_<float> dist_mat) {
            double target_entropy = log2(this->perplexity_);
            cv::Mat_<float> p_mat = cv::Mat::zeros(dist_mat.rows, dist_mat.cols, CV_64F);
 
            for (int i = 0; i < dist_mat.rows; i++) {
                cv::Mat dist = dist_mat.row(i);
                double sigma = 1.0;
                double sigma_min = sigmaMin;
                double sigma_max = sigmaMax;

                for (int iter = 0; iter < 1000; iter++) {
                    cv::Mat prob = calc_softmax(dist / (2 * sigma * sigma), i);
                    double entropy = calc_entropy(prob);
 
                    if (abs(entropy - target_entropy) < kEps) break;
                    if (entropy > target_entropy) sigma_max = sigma;
                    else sigma_min = sigma;
                    sigma = (sigma_max + sigma_min) / 2;
                }
                cv::Mat prob = calc_softmax(dist / (2 * sigma * sigma), i);
                prob.copyTo(p_mat.row(i));
            }

            return p_mat.clone();
        }

        // Расчитывает квадраты расстояний по норме L2 между точками многомерного пространства
        cv::Mat_<float> euc_dists(const cv::Mat a) {
            int points_cnt = a.rows;
            cv::Mat_<float> dist(points_cnt, points_cnt);
            for (int i = 0; i < points_cnt - 1; i++) {
                for (int j = i; j < points_cnt; j++) {
                    dist[i][j] = norm(a.row(i), a.row(j), cv::NORM_L2SQR);
                    dist[j][i] = dist[i][j];
                }
            }
            return dist.clone();
        }

    public:

        tsne(cv::Mat_<float> data, int n, double perplexity, double lr, double gamma) : 
            out_dims_ {n}, perplexity_ {perplexity}, learning_rate_ {lr}, gamma_{gamma}
        {
            this->n_ = data.rows;
            this->res = this->run(data);
        }

        float getSquaredNorm(cv::Mat& mat, int i, int j) {

            cv::Mat y_i = mat.row(i);
            cv::Mat y_j = mat.row(j);
            float squaredNorm = cv::norm(y_i - y_j, cv::NORM_L2SQR);

            return squaredNorm;
        }

        void calcQMatrix(cv::Mat& res) {
            
            float denominator = 0.0;
            for (int i = 0; i < res.rows; i++)
                for (int j = 0; j < res.cols; j++) {
                    // if (i == j) continue;
                    float squaredNorm = this->getSquaredNorm(res, i, j);
                    denominator += 1 / (1 + squaredNorm);
                }

            for (int i = 0; i < this->ld_aff_.rows; i++) {
                for (int j = 0; j < this->ld_aff_.cols; j++) {
                    float squaredNorm = this->getSquaredNorm(res, i, j);
                    float z = 1 / (1 + squaredNorm);
                    this->ld_aff_.at<float>(i, j) = z / denominator;
                }
            }
        }

        cv::Mat run(cv::Mat_<float> data) {
            int n = this->n_;

            cv::Mat_<float> dists = euc_dists(data);
            double a, b;
            cv::minMaxLoc(dists, &a, &b);
            this->hd_aff_ = prob_distributions(dists);
            this->ld_aff_ = cv::Mat::zeros(n, n, CV_32F);
            cv::Mat res = cv::Mat::ones(data.rows, this->out_dims_, CV_32F);
            learning_rate = this->learning_rate_ * cv::Mat::ones(res.size(), res.type());

            // std::random_device rd;
            // std::mt19937 gen(rd());
            // std::normal_distribution<float> dist(0, 0.0001);

            // for (int i = 0; i < res.rows; i++)
            //     for (int j = 0; j < res.cols; j++)
            //         res.at<float>(i, j) = dist(gen);
            cv::RNG rng(43);
            cv::randn(res, 0, 0.0001);

            this->calcQMatrix(res);

            // cv::Mat res_prev = res.clone();
            std::queue<cv::Mat> res_prevs;
            cv::Mat grad_prev = cv::Mat::ones(res.rows, res.cols, CV_32F);
            res_prevs.push(res);
            res_prevs.push(res);
            for(int t = 0; t < this->max_steps_; t++) {
                cv::Mat grad = cv::Mat::zeros(res.rows, res.cols, CV_32F);
                if (t < 0.1 * this->max_steps_) 
                    alpha = 0.9;
                else alpha = 0.5;

                for (int i = 0; i < res.rows; i++) {
                    cv::Mat grad_row = cv::Mat::zeros(1, res.cols, CV_32F);
                    for (int j = 0; j < res.rows; j++) {
                        if (i == j) continue;

                        float squaredNorm = this->getSquaredNorm(res, i, j);
                        float p_ij = this->hd_aff_.at<float>(i, j);
                        // if (t < 50) 
                        //     p_ij *= 4;
                        float q_ij = this->ld_aff_.at<float>(i, j);
                        cv::Mat y_i = res.row(i);
                        cv::Mat y_j = res.row(j);
                        float z = (1 + squaredNorm);

                        grad_row += (p_ij - q_ij) * (y_i - y_j) / z;
                    }
                    grad_row.copyTo(grad.row(i));
                }
                grad *= 4;
                if (t < 0.1 * this->max_steps_)
                    grad += 0.001 * res;
                learning_rate += this->gamma_ * grad.mul(grad_prev);
                grad_prev = grad.clone();
                // grad_i = alpha * grad + (1 - alpha) * grad_i;


                res_prevs.push(res);
                cv::Mat res_prev = res_prevs.front();
                res_prevs.pop();
                res -= learning_rate.mul(grad) + alpha * (res - res_prev);
                this->calcQMatrix(res);
            }

            return res;
        }

    public:
        cv::Mat_<float> res;

    private:
        int n_; // Число входных точек
        int out_dims_; // Число выходных измерений
        // Используется бин. поиск для поиска оптимальной sigma для достижения заданной перплексии
        double perplexity_;
        double learning_rate_;
        double gamma_;

        cv::Mat_<float> hd_aff_;  // Попарные вероятности между точками в исходном пространстве
        cv::Mat_<float> ld_aff_; // Попарные вероятности между точками в целевом пространстве

        int max_steps_ = 100;
};