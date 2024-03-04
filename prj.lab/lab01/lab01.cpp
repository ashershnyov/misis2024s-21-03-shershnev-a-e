#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>

cv::Mat1b gamma_correct(cv::Mat1b img, float gamma) {
  cv::Size img_dim = img.size();
  int tile_width = img_dim.width / 256.;
  for (int grad = 0; grad <= 255; grad++) {
    uchar color = pow(float(grad) / 255., gamma) * 255;
    for (int intile_coord = 0; intile_coord < tile_width; intile_coord++) {
      img.col(tile_width * grad + intile_coord).setTo(color);
    }
  }
  return img;
}

int main(int argc, char* argv[]) {
  const cv::String parser_keys = 
  "{height h |30 |}"
  "{width  s |3  |}"
  "{gamma  g |2.4|}"
  "{@name    |   |}"
  ;
  cv::CommandLineParser parser(argc, argv, parser_keys);

  int width = parser.get<int>("width");
  int height = parser.get<int>("height");
  double gamma = parser.get<double>("gamma");

  bool write_to_file = false;
  std::string filename = parser.get<cv::String>("@name");
  if (filename != "") {
    write_to_file = true;
  }

  cv::Mat1b img1(height, width * 256);
  cv::Mat1b img2(height, width * 256);

  img1 = gamma_correct(img1, 1.0);
  img2 = gamma_correct(img2, gamma);

  cv::vconcat(img1, img2, img1);

  if (!write_to_file) {
    cv::imshow("lab01", img1);
    cv::waitKey(0);
  }
  else {
    cv::imwrite(filename, img1);
  }
}