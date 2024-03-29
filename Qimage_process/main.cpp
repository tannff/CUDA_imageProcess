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
//extern cudaError_t distance_transform(unsigned char* img_in, unsigned char* img_out, const int img_width, const int img_height);
extern cudaError_t distancetransform(unsigned char* img_in, unsigned char* updown, unsigned char* downup, unsigned char* leftright, unsigned char* rightleft, unsigned char* dtresult, const int img_width, const int img_height);

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

		unsigned char* d_leftright;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_leftright, img_width * img_height * sizeof(unsigned char)));
		unsigned char* d_rightleft;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rightleft, img_width * img_height * sizeof(unsigned char)));
		unsigned char* d_updown;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_updown, img_width * img_height * sizeof(unsigned char)));
		unsigned char* d_downup;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_downup, img_width * img_height * sizeof(unsigned char)));
		unsigned char* d_likedtresult;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_likedtresult, img_width * img_height * sizeof(unsigned char)));

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

		int* d_max_x;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_max_x, 256 * sizeof(int)));
		int* d_max_y;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_max_y, 256 * sizeof(int)));

		float* d_filter;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_filter, filterWidth * filterWidth * sizeof(float)));
		CHECK_CUDA_ERROR(cudaMemcpy(d_filter, filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
		unsigned* d_map;
		CHECK_CUDA_ERROR(cudaMalloc((void**)&d_map, img_width * img_height * sizeof(d_map[0])));

		//6.转换颜色，同时记录所用时间
		cudaEvent_t event_begin, event_end ;
		cudaEvent_t event_begin_gray, event_begin_thresh, event_begin_gauss, event_begin_closed, event_begin_canny, event_begin_dist,
					event_end_gray, event_end_thresh, event_end_gauss, event_end_closed, event_end_canny, event_end_dist;

		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin));       //创建 CUDA 事件对象 event_begin 和 event_end
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin, 0));

		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin_gray));
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end_gray));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin_gray, 0));     
		CHECK_CUDA_ERROR(rgb_to_gray(d_rgb, d_gray, img_width, img_height, d_hist));
		CHECK_CUDA_ERROR(cudaEventRecord(event_end_gray, 0));

		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin_thresh));
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end_thresh));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin_thresh, 0));
		CHECK_CUDA_ERROR(thresh_cal(d_hist, d_sum, d_s, d_n, d_val, img_width, img_height, d_t));
		CHECK_CUDA_ERROR(gray_to_otsu_binary(d_gray, d_thresh, img_width, img_height, d_t));
		CHECK_CUDA_ERROR(cudaEventRecord(event_end_thresh, 0));

		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin_gauss));
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end_gauss));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin_gauss, 0));
		CHECK_CUDA_ERROR(gaussian_filter(d_thresh, d_gauss, img_width, img_height, filterWidth, d_filter));
		CHECK_CUDA_ERROR(cudaEventRecord(event_end_gauss, 0));

		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin_closed));
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end_closed));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin_closed, 0));
		CHECK_CUDA_ERROR(dilation(d_gauss, d_dil, img_width, img_height));
		CHECK_CUDA_ERROR(erosion(d_dil, d_closed, img_width, img_height));
		CHECK_CUDA_ERROR(cudaEventRecord(event_end_closed, 0));
		
		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin_canny));
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end_canny));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin_canny, 0));
		CHECK_CUDA_ERROR(sobel_intensity_gradient(d_closed, d_sobel, d_gx, d_gy, img_width, img_height));
		CHECK_CUDA_ERROR(non_max(d_sobel, d_nms, d_gx, d_gy, img_width, img_height));
		CHECK_CUDA_ERROR(hysteresis(d_nms, d_high, d_trace, d_map, t_high, t_low, img_width, img_height));
		CHECK_CUDA_ERROR(cudaEventRecord(event_end_canny, 0));

		CHECK_CUDA_ERROR(cudaEventCreate(&event_begin_dist));
		CHECK_CUDA_ERROR(cudaEventCreate(&event_end_dist));
		CHECK_CUDA_ERROR(cudaEventRecord(event_begin_dist, 0));
		//CHECK_CUDA_ERROR(distancetransform(d_closed, d_updown, d_downup, d_leftright, d_rightleft, d_dist, img_width, img_height));
		//CHECK_CUDA_ERROR(distance_transform(d_closed, d_dist, img_width, img_height/*, d_max_x, d_max_x*/));
		CHECK_CUDA_ERROR(cudaEventRecord(event_end_dist, 0));

		
		CHECK_CUDA_ERROR(cudaEventRecord(event_end, 0));      //用于记录 CUDA 事件对象 event_begin(开始） 的时间戳,在调用时记录
		CHECK_CUDA_ERROR(cudaStreamSynchronize(0));           //同步

		//7.计算时间
		float cost_time, gtimes1, gtimes2, gtimes3, gtimes4, gtimes5, gtimes6;
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&cost_time, event_begin, event_end));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&gtimes1, event_begin_gray, event_end_gray));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&gtimes2, event_begin_thresh, event_end_thresh));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&gtimes3, event_begin_gauss, event_end_gauss));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&gtimes4, event_begin_closed, event_end_closed));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&gtimes5, event_begin_canny, event_end_canny));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&gtimes6, event_begin_dist, event_end_dist));
		std::cout << "rgb_to_gray cost time(gpu): " << gtimes6 << "ms" << std::endl;

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
		CHECK_CUDA_ERROR(cudaFree(d_leftright))
		CHECK_CUDA_ERROR(cudaFree(d_rightleft))
		CHECK_CUDA_ERROR(cudaFree(d_updown))
		CHECK_CUDA_ERROR(cudaFree(d_downup))
		CHECK_CUDA_ERROR(cudaFree(d_likedtresult))
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
		clock_t start_time_gray, start_time_thresh, start_time_gauss, start_time_closed, start_time_canny, start_time_distancetransform,
			    end_time_gray, end_time_thresh, end_time_gauss, end_time_closed, end_time_canny, end_time_distancetransform;

		start_time = clock();
		start_time_gray = clock();   //开始

		//1.灰度化
		cv::Mat gray_cpu_image(img_height, img_width, CV_8UC1);
		cvtColor(rgb_image, gray_cpu_image, COLOR_BGR2GRAY);
		rgb2grayincpu(rgb_image.data, gray_cpu_image.data, img_width, img_height);

		end_time_gray = clock();     //结束
		double times1 = (double)(end_time_gray - start_time_gray) * 1000 / CLOCKS_PER_SEC;
		cout << "rgb to gray cost time(cpu)： " << times1 << " ms" << endl;

		start_time_thresh = clock();   //开始

		//2.二值化
		cv::Mat binary_cpu_image(img_height, img_width, CV_8UC1);
		threshold(gray_cpu_image, binary_cpu_image, 40, 255, THRESH_BINARY /*| THRESH_OTSU*/);

		end_time_thresh = clock();     //结束
		double times2 = (double)(end_time_thresh - start_time_thresh) * 1000 / CLOCKS_PER_SEC;
		cout << "gray to thresh cost time(cpu)： " << times2 << " ms" << endl;

		start_time_gauss = clock();   //开始

		//3.高斯滤波
		Mat img_gaussian = Mat(gray_cpu_image.size(), CV_8U, Scalar(0));
		GaussianBlur(binary_cpu_image, img_gaussian, Size(5, 5), 0, 0);

		end_time_gauss = clock();     //结束
		double times3 = (double)(end_time_gauss - start_time_gauss) * 1000 / CLOCKS_PER_SEC;
		cout << "thresh to gauss cost time(cpu)： " << times3 << " ms" << endl;

		start_time_closed = clock();   //开始

		//4.闭运算，将断续的轮廓连接
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
		Mat blkImg(img_gaussian.size(), CV_8UC1, Scalar(0));
		morphologyEx(img_gaussian, blkImg, MORPH_CLOSE, element);

		end_time_closed = clock();     //结束
		double times4 = (double)(end_time_closed - start_time_closed) * 1000 / CLOCKS_PER_SEC;
		cout << "gauss to closed cost time(cpu)： " << times4 << " ms" << endl;

		//4.灰度梯度增强
		//Mat sobel_x, sobel_y, sobel_xy;
		//Sobel(blkImg, sobel_x, CV_8U, 1, 0, 3);
		//Sobel(blkImg, sobel_y, CV_8U, 0, 1, 3);
		//addWeighted(sobel_x, 0.5, sobel_y, 0.5, 1, sobel_xy);                    //沿x,y轴方向叠加梯度

		start_time_canny = clock();   //开始

		//5.canny边缘检测
		Mat img_canny = Mat(blkImg.size(), CV_8U, Scalar(0));
		Canny(blkImg, img_canny, t_low, t_high);
		vector<vector<Point> > contour_vec;
		vector<Vec4i> hierarchy;
		findContours(img_canny, contour_vec, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		for (int real_contour = 0; real_contour < contour_vec.size(); ++real_contour) {
			drawContours(rgb_image, contour_vec, real_contour, Scalar(255, 255, 255), 1.8, 8);
		}

		end_time_canny = clock();     //结束
		double times5 = (double)(end_time_canny - start_time_canny) * 1000 / CLOCKS_PER_SEC;
		cout << "closed to canny cost time(cpu)： " << times5 << " ms" << endl;

		//6.最大连通域
		cv::Mat img_dom;
		blkImg.copyTo(img_dom);
		connected_domains_cpu(img_dom, img_dom);

		start_time_distancetransform = clock();   //开始
		
		bool useDistanceTransform = true;        // 是否使用距离变换算法
		
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

		end_time_distancetransform = clock();     //结束
		double times6 = (double)(end_time_distancetransform - start_time_distancetransform) * 1000 / CLOCKS_PER_SEC;
		cout << "canny to distancetransform cost time(cpu)： " << times6 << " ms" << endl;
	
		end_time = clock();     //结束
		double Times = (double)(end_time - start_time) * 1000 / CLOCKS_PER_SEC;
		cout << "rgb to gray cost time(cpu)： " << Times << " ms" << endl;
		ImageUtils::write_image(rgb_image, output2_file_path);  //写到指定地址

		/*ori_image ori_display;
		int reys = 1;
		emit ori_display.test(reys);*/

		//camera_on(qimage, a);

		emit w.image_proccess_speed(times1, times2, times3, times4, times5, times6, 
									cost_time, 
									gtimes1, gtimes2, gtimes3, gtimes4, gtimes5);

		emit w.width_measure(maxValue, interpolation);

		w.show();

		return a.exec();

	}
