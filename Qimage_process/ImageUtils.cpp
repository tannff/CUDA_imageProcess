#include "ImageUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

cv::Mat ImageUtils::read_image(const std::string& file_name) {
	cv::Mat ret = cv::imread(file_name);
	if (ret.data != nullptr) {
		std::cout << "read image: " << file_name << " success\n";
		std::cout << "image width: " << ret.cols << ", image height: " << ret.rows << std::endl;
	} else {
		std::cerr << "read image: " << file_name << " failed\n";
	}
	return ret;
}

int ImageUtils::write_image(const cv::Mat& img, const std::string& file_name) {
	bool success = cv::imwrite(file_name, img);
	if (success) {
		std::cout << "write image: " << file_name << " success\n";
		return 0;
	} else {
		std::cerr << "write image : " << file_name << " failed\n";
		return -1;
	}
}
