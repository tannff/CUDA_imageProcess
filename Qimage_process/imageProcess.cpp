#include "imageProcess.h"

using namespace std;
using namespace cv;

//����ת���Ҷ�ͼ��
void rgb2grayincpu(unsigned char* const d_in, unsigned char* const d_out, uint img_width, uint img_height)
{
	//ʹ������ѭ��Ƕ��ʵ��x���� y����ı任
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

//ֻ���������ͨ��
void connected_domains_cpu(cv::Mat d_in, cv::Mat d_out)
{
	////�����ͨ��
	int n_comps = connectedComponents(d_in, d_out, 4, CV_16U);
	vector<int> histogram_of_labels;
	for (int i = 0; i < n_comps; i++)//��ʼ��labels�ĸ���Ϊ0
	{
		histogram_of_labels.push_back(0);
	}
	for (int row = 0; row < d_out.rows; row++) //����ÿ��labels�ĸ���--����ͨ������
	{
		for (int col = 0; col < d_out.cols; col++)
		{
			int label = d_out.at<unsigned short>(row, col);
			histogram_of_labels[label] += 1;
		}
	}
	histogram_of_labels.at(0) = 0; //��������labels��������Ϊ0

	//����������ͨ��labels����
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

	//�������ͨ����Ϊ255������������ͨ����0
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
	//��ͼ�����ΪCV_8U��ʽ
	d_out.convertTo(d_out, CV_8U);
}

//����任�㷨
void distancetransform_cpu(cv::Mat d_in, cv::Mat d_men, float* maxValue/*, cv::Point Pt*/)
{
	cv::Point Pt = Point(0, 0);
	cv::Mat img_Thin(d_men.size(), CV_32FC1); //���屣�����任�����Mat����  
	distanceTransform(d_men, img_Thin, DIST_L2, 3);  //����任  

	//����任������ֵ������Ϊ��ͼ�εļ�������
	Mat distShow;
	distShow = Mat::zeros(img_Thin.size(), CV_8UC1);
	for (int i = 0; i < img_Thin.rows; i++)
	{
		for (int j = 0; j < img_Thin.cols; j++)
		{
			distShow.at<uchar>(i, j) = img_Thin.at<float>(i, j);
			if (img_Thin.at<float>(i, j) > *maxValue)
			{
				*maxValue = img_Thin.at<float>(i, j);  //��ȡ����任�����ֵ  
				Pt = Point(j, i);  //��¼���ֵ������  
			}
		}
	}
	normalize(distShow, distShow, 0, 255, NORM_MINMAX); //Ϊ����ʾ��������0~255��һ��  
	circle(d_in, Pt, *maxValue, Scalar(255, 0, 0), 2);//��ԭͼ�б�ʾ��������뾶
	circle(d_in, Pt, 2, Scalar(255, 0, 0), 3);//��ԭͼ�б�ʾ����������

}

//������Բ�㷨
void elipse_up(cv::Mat img_in, cv::Mat img_men, std::vector<std::vector<cv::Point>>& contour_vec, double* interpolation) {
	//��Բ���
	for (size_t i = 0; i < contour_vec.size(); i++)
	{
		//��ϵĵ�����Ϊ6
		size_t count = contour_vec[i].size();
		if (count < 6)
			continue;

		RotatedRect box = fitEllipse(contour_vec[i]);

		//�������ȴ���30�����ų����������
		if (MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height) * 30)
			continue;

		// ��ȡ��Բ�����ĵ㡢����Ͷ�����Ϣ
		Point2f center = box.center;
		Size2f axes = box.size;
		float angle = box.angle;

		if (axes.width < 100 || axes.height < 100)
			continue;

		//������ϵ���Բ
		ellipse(img_in, box, Scalar(255, 0, 0), 2, CV_AA);

		// ������Բ�ĳ�����˵�����
		double radian = angle * CV_PI / 180.0;
		double cosine = cos(radian);
		double sine = sin(radian);

		Point2f longAxisEndPoint(center.x + (axes.width / 2) * cosine,
			center.y + (axes.width / 2) * sine);
		Point2f shortAxisEndPoint(center.x - (axes.height / 2) * sine,
			center.y + (axes.height / 2) * cosine);

		// ��ͼ���ϻ�����Բ�ĳ�����
		line(img_in, center, longAxisEndPoint, Scalar(0, 255, 255), 2);
		line(img_in, center, shortAxisEndPoint, Scalar(0, 0, 255), 2);

		// ������Բ�ĽǶȼ�����ת����
		Mat rotationMatrix = getRotationMatrix2D(center, angle - 90, 1.0);

		// ��ԭʼͼ�������ת
		Mat rotatedImage;
		warpAffine(img_men, rotatedImage, rotationMatrix, img_men.size(), INTER_CUBIC);

		// ��ʾ��ת���ͼ��
		circle(rotatedImage, Point(120, 164), 2, Scalar(255, 0, 0), 3);

		Mat rowProjection;
		reduce(rotatedImage, rowProjection, 1, REDUCE_MAX);  // �����з���������ֵ

		int top = -1;  // �ϱ߽�
		int bottom = -1;  // �±߽�

		std::cout << "�ܹ��У�" << rowProjection.rows << std::endl;

		for (int row_up = 0; row_up < rowProjection.rows; row_up++) {            //���ϵ���ɨ��
			uchar pixel_t_Value = rowProjection.at<uchar>(row_up, rowProjection.cols);
			if (pixel_t_Value == 255) {                      //rowProjection�ĵ�row�е�����ֵ����0ʱ������ѭ��
				top = row_up;
				break;
			}
		}

		for (int row_down = rowProjection.rows - 1; row_down >= 0; row_down--) {      //���¶���ɨ��  row=��
			uchar pixel_d_Value = rowProjection.at<uchar>(row_down, rowProjection.cols);
			if (pixel_d_Value == 255) {
				bottom = row_down;
				break;
			}
		}

		*interpolation = (bottom - top) / 63.00;


	}
}
	
	