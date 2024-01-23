#pragma once

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp" 

void rgb2grayincpu(unsigned char* const d_in, unsigned char* const d_out, uint img_width, uint img_height);
void connected_domains_cpu(cv::Mat d_in, cv::Mat d_out);
void distancetransform_cpu(cv::Mat d_in, cv::Mat d_men, float* maxValue/*, cv::Point Pt*/);
void elipse_up(cv::Mat img_in, cv::Mat img_men, std::vector<std::vector<cv::Point>>& contour_vec, double* interpolation);
