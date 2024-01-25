#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <QObject>
#include "time.h"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ImageUtils.h"
#include "imageProcess.h"
#include "Qimage_process.h"
#include "camera_on.h"
#include <QtWidgets/QApplication>
#include <QObject>
#include <QElapsedTimer>

#define MAX_THREADS 1024

using namespace std;
using namespace cv;


//用于检查cuda返回是否有问题，定义成宏，没有时间消耗
#define CHECK_CUDA_ERROR(ret)									\
	if (ret != cudaSuccess) {									\
		std::cerr << cudaGetErrorString(ret) << std::endl;		\
		return -1;												\
	}

//这些变量定义在最上面，方便修改。
const std::string input_file_path = "F:\\about_lesson\\Qimage_process\\CUDA_imageProcess\\Qimage_process\\6.jpg";
const std::string output1_file_path = "F:\\about_lesson\\Qimage_process\\CUDA_imageProcess\\Qimage_process\\gray_1_cuda.jpg";
const std::string output2_file_path = "F:\\about_lesson\\Qimage_process\\CUDA_imageProcess\\Qimage_process\\gray_1_cpu.jpg";

extern cudaError_t rgb_to_gray(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height, int* hist);
extern cudaError_t thresh_cal(const int* hist, float* sum, float* s, float* n, float* val, int img_width, int img_height, int* OtsuThresh);
extern cudaError_t gray_to_otsu_binary(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height, int* hThresh);
extern cudaError_t dilation(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height);
extern cudaError_t erosion(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height);
extern cudaError_t gaussian_filter(unsigned char* img_in, unsigned char* img_gauss, int img_width, int img_height, int filterWidth, float* filter);
extern cudaError_t sobel_intensity_gradient(unsigned char* img_in, unsigned char* img_sobel, int* Gx, int* Gy,int img_width, int img_height);
extern cudaError_t non_max(unsigned char* img_in, unsigned char* img_nms, int* Gx, int* Gy, int img_width, int img_height);
extern cudaError_t hysteresis(unsigned char* img_in, unsigned char* img_high, unsigned char* img_trace, unsigned* strong_edge_mask, int t_high, int t_low, int img_width, int img_height);
extern cudaError_t distance_transform(unsigned char* img_in, unsigned char* img_out, const int img_width, const int img_height);

int main(int argc, char* argv[]) {

	QApplication a(argc, argv);
	Qimage_process w;

	
	//0.初始化cuda，也可以不加。老版本cuda需要加。
	CHECK_CUDA_ERROR(cudaFree(0));

	//1.read src image
	cv::Mat rgb_image = ImageUtils::read_image(input_file_path);
	if (rgb_image.data == nullptr) {
		return -1;
	}

	//2.图像尺寸参数
	int img_width = rgb_image.cols, img_height = rgb_image.rows;
	int img_depth = rgb_image.channels();

	//3.check channnels
	//[*]opencv的imread默认读取为bgr的格式，如果图像是四通道的，我们就将bgra转成rgb；如果图像是三通道的，就将bgr转rgb
	if (rgb_image.channels() == 4) {
		cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGRA2RGB);
	}
	else if (rgb_image.channels() == 3) {
		cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2RGB);
	}

	
	//4.高斯核（需保证高斯核大小为≥3的奇数）
	float Sigma = 1;
	int filterWidth = 5;
	if (filterWidth < 3) filterWidth = 3;
	else filterWidth = (int)(filterWidth / 2) * 2 + 1;
	float* filter = new float[filterWidth * filterWidth];   //生成高斯核
	int center = filterWidth / 2;
	float sum = 0;

	for (int i = 0; i < filterWidth; i++)
	{
		for (int j = 0; j < filterWidth; j++)
		{
			filter[i * filterWidth + j] = exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * Sigma * Sigma));
			sum += filter[i * filterWidth + j];
		}
	}
	double sum1 = 1 / sum;
	for (int i = 0; i < filterWidth; i++)
	{
		for (int j = 0; j < filterWidth; j++)
		{
			filter[i * filterWidth + j] *= sum1;  //高斯卷积核归一化
		}
	}

	//定义阈值
	int t_high = 150;
	int t_low = 50;

	//定义距离变换类型
	DistanceTypes distance_type = DIST_L1;
	DistanceTransformMasks mask_size = DIST_MASK_3;

	//5.申请device显存，拷贝src数据
		unsigned char* d_rgb;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rgb, img_width * img_height * img_depth * sizeof(unsigned char)));
		CHECK_CUDA_ERROR(cudaMemcpy(d_rgb, rgb_image.data, img_width * img_height * img_depth * sizeof(unsigned char), cudaMemcpyHostToDevice));

		unsigned char* d_gray;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gray, img_width * img_height * sizeof(unsigned char)));

		unsigned char* d_thresh;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_thresh, img_width * img_height * sizeof(unsigned char)));

		unsigned char* d_dil;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dil, img_width * img_height * sizeof(unsigned char)));
		unsigned char* d_closed;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_closed, img_width * img_height * sizeof(unsigned char)));

		unsigned char* d_gauss;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gauss, img_width * img_height * sizeof(unsigned char)));

		unsigned char* d_sobel;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sobel, img_width * img_height * sizeof(unsigned char)));

		unsigned char* d_nms;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nms, img_width * img_height * sizeof(unsigned char)));

		unsigned char* d_trace;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_trace, img_width * img_height * sizeof(unsigned char)));
		unsigned char* d_high;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_high, img_width * img_height * sizeof(unsigned char)));

		unsigned char* d_dist;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dist, img_width * img_height * sizeof(unsigned char)));

		int* d_hist;
		float* d_sum;
		float* d_s;
		float* d_n;
		float* d_val;
		int* d_t;
		int* d_gx;
		int* d_gy;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hist, 256 * sizeof(int)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sum, sizeof(float)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_s, 256 * sizeof(float)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_n, 256 * sizeof(float)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_val, 256 * sizeof(float)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t, 2 * sizeof(int)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gx, img_width * img_height * sizeof(int)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gy, img_width * img_height * sizeof(int)));

		float* d_filter;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_filter, filterWidth * filterWidth * sizeof(float)));
		CHECK_CUDA_ERROR(cudaMemcpy(d_filter, filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
		unsigned* d_map;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_map, img_width * img_height * sizeof(d_map[0])));

		//6.转换颜色，同时记录所用时间
		cudaEvent_t event_begin, event_end ;
		cudaEvent_t event_begin_gray, event_begin_thresh, event_begin_gauss, event_begin_closed, event_begin_canny,
					event_end_gray, event_end_thresh, event_end_gauss, event_end_closed, event_end_canny;

		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin));       //创建 CUDA 事件对象 event_begin 和 event_end
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin, 0));

		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin_gray));
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end_gray));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin_gray, 0));     
		CHECK_CUDA_ERROR(rgb_to_gray(d_rgb, d_gray, img_width, img_height, d_hist));
		CHECK_CUDA_ERROR(cudaEventRecord(event_end_gray, 0));

		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin_gauss));
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end_gauss));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin_gauss, 0));
		CHECK_CUDA_ERROR(gaussian_filter(d_gray, d_gauss, img_width, img_height, filterWidth, d_filter));
		CHECK_CUDA_ERROR(cudaEventRecord(event_end_gauss, 0));

		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin_thresh));
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end_thresh));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin_thresh, 0));
		CHECK_CUDA_ERROR(thresh_cal(d_hist, d_sum, d_s, d_n, d_val, img_width, img_height, d_t));
		CHECK_CUDA_ERROR(gray_to_otsu_binary(d_gauss, d_thresh, img_width, img_height, d_t));
		CHECK_CUDA_ERROR(cudaEventRecord(event_end_thresh, 0));

		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin_closed));
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end_closed));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin_closed, 0));
		CHECK_CUDA_ERROR(dilation(d_thresh, d_dil, img_width, img_height));
		CHECK_CUDA_ERROR(erosion(d_dil, d_closed, img_width, img_height));
		CHECK_CUDA_ERROR(cudaEventRecord(event_end_closed, 0));
		
		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin_canny));
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end_canny));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin_canny, 0));
		CHECK_CUDA_ERROR(sobel_intensity_gradient(d_closed, d_sobel, d_gx, d_gy, img_width, img_height));
		CHECK_CUDA_ERROR(non_max(d_sobel, d_nms, d_gx, d_gy, img_width, img_height));
		CHECK_CUDA_ERROR(hysteresis(d_nms, d_high, d_trace, d_map, t_high, t_low, img_width, img_height));
		CHECK_CUDA_ERROR(cudaEventRecord(event_end_canny, 0));

		CHECK_CUDA_ERROR(distance_transform(d_closed, d_dist, img_width, img_height));
		
		CHECK_CUDA_ERROR(cudaEventRecord(event_end, 0));      //用于记录 CUDA 事件对象 event_begin(开始） 的时间戳,在调用时记录
		CHECK_CUDA_ERROR(cudaStreamSynchronize(0));           //同步

		//7.计算时间
		float cost_time, gtimes1, gtimes2, gtimes3, gtimes4, gtimes5;
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&cost_time, event_begin, event_end));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&gtimes1, event_begin_gray, event_end_gray));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&gtimes2, event_begin_gauss, event_end_gauss));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&gtimes3, event_begin_thresh, event_end_thresh));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&gtimes4, event_begin_closed, event_end_closed));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&gtimes5, event_begin_canny, event_end_canny));
		std::cout << "total(cuda): " << cost_time << "ms" << std::endl;

		//8.保存图像
		cv::Mat binary_image(img_height, img_width, CV_8UC1);
		cudaMemcpy(binary_image.data, d_dist, img_width * img_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);


		ImageUtils::write_image(binary_image, output1_file_path);  //写到指定地址

		//9.释放显存
		CHECK_CUDA_ERROR(cudaFree(d_rgb));
		CHECK_CUDA_ERROR(cudaFree(d_gray));
		CHECK_CUDA_ERROR(cudaFree(d_thresh));
		CHECK_CUDA_ERROR(cudaFree(d_dil));
		CHECK_CUDA_ERROR(cudaFree(d_closed));
		CHECK_CUDA_ERROR(cudaFree(d_hist));
		CHECK_CUDA_ERROR(cudaFree(d_sum));
		CHECK_CUDA_ERROR(cudaFree(d_s));
		CHECK_CUDA_ERROR(cudaFree(d_n));
		CHECK_CUDA_ERROR(cudaFree(d_val));
		CHECK_CUDA_ERROR(cudaFree(d_t));
		CHECK_CUDA_ERROR(cudaFree(d_gauss));
		CHECK_CUDA_ERROR(cudaFree(d_filter));
		CHECK_CUDA_ERROR(cudaFree(d_sobel));
		CHECK_CUDA_ERROR(cudaFree(d_nms));
		CHECK_CUDA_ERROR(cudaFree(d_gx));
		CHECK_CUDA_ERROR(cudaFree(d_gy));
		CHECK_CUDA_ERROR(cudaFree(d_high));
		CHECK_CUDA_ERROR(cudaFree(d_trace));
		CHECK_CUDA_ERROR(cudaFree(d_dist));

		CHECK_CUDA_ERROR(cudaEventDestroy(event_begin));
		CHECK_CUDA_ERROR(cudaEventDestroy(event_end));

		CHECK_CUDA_ERROR(cudaEventDestroy(event_begin_gray));
		CHECK_CUDA_ERROR(cudaEventDestroy(event_end_gray));
		CHECK_CUDA_ERROR(cudaEventDestroy(event_begin_thresh));
		CHECK_CUDA_ERROR(cudaEventDestroy(event_end_thresh));
		CHECK_CUDA_ERROR(cudaEventDestroy(event_begin_gauss));
		CHECK_CUDA_ERROR(cudaEventDestroy(event_end_gauss));
		CHECK_CUDA_ERROR(cudaEventDestroy(event_begin_closed));
		CHECK_CUDA_ERROR(cudaEventDestroy(event_end_closed));
		CHECK_CUDA_ERROR(cudaEventDestroy(event_begin_canny));
		CHECK_CUDA_ERROR(cudaEventDestroy(event_end_canny));


		//CPU图像处理算法
		clock_t start_time, end_time;
		start_time = clock();
		
		QElapsedTimer timer_gray;
		timer_gray.start();

		//1.灰度化
		cv::Mat gray_cpu_image(img_height, img_width, CV_8UC1);
		cvtColor(rgb_image, gray_cpu_image, COLOR_BGR2GRAY);
		rgb2grayincpu(rgb_image.data, gray_cpu_image.data, img_width, img_height);

		qint64 elapsed1 = timer_gray.nsecsElapsed() / 100000.000;
		cout << "rgb to gray cost time(cpu)： " << elapsed1 << " ms" << endl;

		QElapsedTimer timer_gauss;
		timer_gauss.start();

		//2.高斯滤波
		Mat img_gaussian = Mat(gray_cpu_image.size(), CV_8UC1, Scalar(0));
		GaussianBlur(gray_cpu_image, img_gaussian, Size(5, 5), 0, 0);

		qint64 elapsed2 = timer_gauss.nsecsElapsed() / 100000.000;
		cout << "gray to gauss cost time(cpu)： " << elapsed2 << " ms" << endl;

		QElapsedTimer timer_thresh;
		timer_thresh.start();

		//3.二值化
		cv::Mat binary_cpu_image(img_height, img_width, CV_8UC1);
		threshold(img_gaussian, binary_cpu_image, 45, 255, THRESH_BINARY /*| THRESH_OTSU*/);

		qint64 elapsed3 = timer_thresh.nsecsElapsed() / 100000.000;
		cout << "gauss to thresh cost time(cpu)： " << elapsed3 << " ms" << endl;

		QElapsedTimer timer_closed;
		timer_closed.start();

		//4.闭运算，将断续的轮廓连接
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
		Mat blkImg(binary_cpu_image.size(), CV_8UC1, Scalar(0));
		morphologyEx(binary_cpu_image, blkImg, MORPH_CLOSE, element);

		qint64 elapsed4 = timer_closed.nsecsElapsed() / 100000.000;
		cout << "thresh to closed cost time(cpu)： " << elapsed4 << " ms" << endl;

		//4.灰度梯度增强
		//Mat sobel_x, sobel_y, sobel_xy;
		//Sobel(blkImg, sobel_x, CV_8U, 1, 0, 3);
		//Sobel(blkImg, sobel_y, CV_8U, 0, 1, 3);
		//addWeighted(sobel_x, 0.5, sobel_y, 0.5, 1, sobel_xy);                    //沿x,y轴方向叠加梯度

		QElapsedTimer timer_canny;
		timer_canny.start();

		//5.canny边缘检测
		Mat img_canny = Mat(blkImg.size(), CV_8U, Scalar(0));
		Canny(blkImg, img_canny, t_low, t_high);
		vector<vector<Point> > contour_vec;
		vector<Vec4i> hierarchy;
		findContours(img_canny, contour_vec, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		for (int real_contour = 0; real_contour < contour_vec.size(); ++real_contour) {
			drawContours(rgb_image, contour_vec, real_contour, Scalar(255, 255, 255), 2, 8);
		}

		qint64 elapsed5 = timer_canny.nsecsElapsed() / 100000.000;
		cout << "closed to canny cost time(cpu)： " << elapsed5 << " ms" << endl;

		//6.最大连通域
		cv::Mat img_dom;
		blkImg.copyTo(img_dom);
		connected_domains_cpu(img_dom, img_dom);
		
		QElapsedTimer timer_dist;
		timer_dist.start();
		
		bool useDistanceTransform = false;        // 是否使用距离变换算法
		
		float maxValue = 0.0;
		double interpolation = 0.0;
		if (useDistanceTransform) {
			//7.距离变换
			distancetransform_cpu(rgb_image, img_dom, &maxValue);
		}
		else {
			//8.进阶椭圆算法
			elipse_up(rgb_image, img_dom, contour_vec, &interpolation);
		}

		qint64 elapsed6 = timer_dist.nsecsElapsed() / 100000.000;
		cout << "closed to distancetransform cost time(cpu)： " << elapsed6 << " ms" << endl;
	
		end_time = clock();     //结束
		double Times = (double)(end_time - start_time) * 1000 / CLOCKS_PER_SEC;
		cout << "total(cpu)： " << Times << " ms" << endl;
		ImageUtils::write_image(rgb_image, output2_file_path);  //写到指定地址

		emit w.image_proccess_speed(elapsed1, elapsed2, elapsed3, elapsed4, elapsed5, elapsed6,
									cost_time, 
									gtimes1, gtimes2, gtimes3, gtimes4, gtimes5);

		emit w.width_measure(maxValue, interpolation);

		w.show();

		return a.exec();

	}
