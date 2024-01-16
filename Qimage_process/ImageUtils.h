#pragma once

#include <string>
#include <opencv2/core/core.hpp>

class ImageUtils {

public:
	static cv::Mat read_image(const std::string& file_name);
	static int write_image(const cv::Mat& img, const std::string& file_name);
};

