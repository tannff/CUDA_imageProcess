#include "imageProcess.h"

using namespace std;
using namespace cv;

//串行转换灰度图像
void rgb2grayincpu(unsigned char* const d_in, unsigned char* const d_out, uint img_width, uint img_height)
{
	//使用两重循环嵌套实现x方向 y方向的变换
	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{
			d_out[i * img_width + j] = 0.299f * d_in[(i * img_width + j) * 3]
				+ 0.587f * d_in[(i * img_width + j) * 3 + 1]
				+ 0.114f * d_in[(i * img_width + j) * 3 + 2];
		}
	}
}

//只保留最大连通域
void connected_domains_cpu(cv::Mat d_in, cv::Mat d_out)
{
	////标记连通域
	int n_comps = connectedComponents(d_in, d_out, 4, CV_16U);
	vector<int> histogram_of_labels;
	for (int i = 0; i < n_comps; i++)//初始化labels的个数为0
	{
		histogram_of_labels.push_back(0);
	}
	for (int row = 0; row < d_out.rows; row++) //计算每个labels的个数--即连通域的面积
	{
		for (int col = 0; col < d_out.cols; col++)
		{
			int label = d_out.at<unsigned short>(row, col);
			histogram_of_labels[label] += 1;
		}
	}
	histogram_of_labels.at(0) = 0; //将背景的labels个数设置为0

	//计算最大的连通域labels索引
	int maximum = 0;
	int max_idx = 0;
	for (int i = 0; i < n_comps; i++)
	{
		if (histogram_of_labels.at(i) > maximum)
		{
			maximum = histogram_of_labels.at(i);
			max_idx = i;
		}
	}

	//将最大连通域标记为255，并将其他连通域置0
	for (int row = 0; row < d_out.rows; row++)
	{
		for (int col = 0; col < d_out.cols; col++)
		{
			if (d_out.at<unsigned short>(row, col) == max_idx)
			{
				d_out.at<unsigned short>(row, col) = 255;
			}
			else
			{
				d_out.at<unsigned short>(row, col) = 0;
			}
		}
	}
	//将图像更改为CV_8U格式
	d_out.convertTo(d_out, CV_8U);
}

//距离变换算法
void distancetransform_cpu(cv::Mat d_in, cv::Mat d_men, float* maxValue/*, cv::Point Pt*/)
{
	cv::Point Pt = Point(0, 0);
	cv::Mat img_Thin(d_men.size(), CV_32FC1); //定义保存距离变换结果的Mat矩阵  
	distanceTransform(d_men, img_Thin, DIST_L2, 3);  //距离变换  

	//距离变换后的最大值我们认为是图形的几何中心
	Mat distShow;
	distShow = Mat::zeros(img_Thin.size(), CV_8UC1);
	for (int i = 0; i < img_Thin.rows; i++)
	{
		for (int j = 0; j < img_Thin.cols; j++)
		{
			distShow.at<uchar>(i, j) = img_Thin.at<float>(i, j);
			if (img_Thin.at<float>(i, j) > *maxValue)
			{
				*maxValue = img_Thin.at<float>(i, j);  //获取距离变换的最大值  
				Pt = Point(j, i);  //记录最大值的坐标  
			}
		}
	}
	normalize(distShow, distShow, 0, 255, NORM_MINMAX); //为了显示清晰，做0~255归一化  
	circle(d_in, Pt, *maxValue, Scalar(255, 0, 0), 2);//在原图中标示出最大距离半径
	circle(d_in, Pt, 2, Scalar(255, 0, 0), 3);//在原图中标示出几何中心

}

//进阶椭圆算法
void elipse_up(cv::Mat img_in, cv::Mat img_men, std::vector<std::vector<cv::Point>>& contour_vec, double* interpolation) {
	//椭圆拟合
	for (size_t i = 0; i < contour_vec.size(); i++)
	{
		//拟合的点至少为6
		size_t count = contour_vec[i].size();
		if (count < 6)
			continue;

		RotatedRect box = fitEllipse(contour_vec[i]);

		//如果长宽比大于30，则排除，不做拟合
		if (MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height) * 30)
			continue;

		// 获取椭圆的中心点、长轴和短轴信息
		Point2f center = box.center;
		Size2f axes = box.size;
		float angle = box.angle;

		if (axes.width < 100 || axes.height < 100)
			continue;

		//画出拟合的椭圆
		ellipse(img_in, box, Scalar(255, 0, 0), 2, CV_AA);

		// 计算椭圆的长短轴端点坐标
		double radian = angle * CV_PI / 180.0;
		double cosine = cos(radian);
		double sine = sin(radian);

		Point2f longAxisEndPoint(center.x + (axes.width / 2) * cosine,
			center.y + (axes.width / 2) * sine);
		Point2f shortAxisEndPoint(center.x - (axes.height / 2) * sine,
			center.y + (axes.height / 2) * cosine);

		// 在图像上绘制椭圆的长短轴
		line(img_in, center, longAxisEndPoint, Scalar(0, 255, 255), 2);
		line(img_in, center, shortAxisEndPoint, Scalar(0, 0, 255), 2);

		// 根据椭圆的角度计算旋转矩阵
		Mat rotationMatrix = getRotationMatrix2D(center, angle - 90, 1.0);

		// 对原始图像进行旋转
		Mat rotatedImage;
		warpAffine(img_men, rotatedImage, rotationMatrix, img_men.size(), INTER_CUBIC);

		// 显示旋转后的图像
		circle(rotatedImage, Point(120, 164), 2, Scalar(255, 0, 0), 3);

		Mat rowProjection;
		reduce(rotatedImage, rowProjection, 1, REDUCE_MAX);  // 沿着行方向计算最大值

		int top = -1;  // 上边界
		int bottom = -1;  // 下边界

		std::cout << "总共行：" << rowProjection.rows << std::endl;

		for (int row_up = 0; row_up < rowProjection.rows; row_up++) {            //自上到下扫描
			uchar pixel_t_Value = rowProjection.at<uchar>(row_up, rowProjection.cols);
			if (pixel_t_Value == 255) {                      //rowProjection的第row行的像素值大于0时，跳出循环
				top = row_up;
				break;
			}
		}

		for (int row_down = rowProjection.rows - 1; row_down >= 0; row_down--) {      //自下而上扫描  row=行
			uchar pixel_d_Value = rowProjection.at<uchar>(row_down, rowProjection.cols);
			if (pixel_d_Value == 255) {
				bottom = row_down;
				break;
			}
		}

		*interpolation = (bottom - top) / 63.00;


	}
}
	
	