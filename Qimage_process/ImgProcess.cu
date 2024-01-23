#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>

using namespace std;

/* CUDA�汾�µĻҶȻ����� */
__global__ void kernel_rgb_to_gray(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height, int* hist) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= img_width || idy >= img_height) {
		return;
	}
	int local_index = idy * img_width + idx;
	unsigned char r = img_in[3 * local_index + 0];
	unsigned char g = img_in[3 * local_index + 1];
	unsigned char b = img_in[3 * local_index + 2];
	img_out[local_index] = r * 0.299 + g * 0.587 + b * 0.114;
	//img_out[local_index] = 255;
	atomicAdd(&hist[img_in[local_index]], 1);
	return;
}

extern cudaError_t rgb_to_gray(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height, int* hist) {
	dim3 block_dim(16, 16);   //�����߳̿�
	dim3 grid_dim = dim3((img_width + block_dim.x - 1) / block_dim.x,
		(img_height + block_dim.y - 1) / block_dim.y);

	kernel_rgb_to_gray << <grid_dim, block_dim >> > (img_in, img_out, img_width, img_height, hist);
	return cudaDeviceSynchronize();
}

/* CUDA�汾�µĶ�ֵ������ */
//��򷨼���
__host__ int otsuThresh(int* hist, int imgHeight, int imgWidth)
{
	float sum = 0;
	for (int i = 0; i < 256; i++)
	{
		sum += i * hist[i];
	}
	float w0 = 0, u0 = 0;
	float u = sum / (imgHeight * imgWidth);
	float val = 0, maxval = 0;
	float s = 0, n = 0;
	int thresh = 0;
	for (int i = 0; i < 256; i++)
	{
		s += hist[i] * i;
		n += hist[i];
		w0 = n / (imgHeight * imgWidth);
		u0 = s / n;
		val = (u - u0) * (u - u0) * w0 / (1 - w0);
		if (val > maxval)
		{
			maxval = val;
			thresh = i;
		}
	}
	return thresh;
}

//CUDA�汾���������䷽��ĸı����
__global__ void OTSUthresh(const int* hist, float* sum, float* s, float* n, float* val, int img_width, int img_height, int* OtsuThresh)
{
	if (blockIdx.x == 0)
	{
		int index = threadIdx.x;
		atomicAdd(&sum[0], hist[index] * index);
	}
	else
	{
		int index = threadIdx.x;
		if (index < blockIdx.x)
		{
			atomicAdd(&s[blockIdx.x - 1], hist[index] * index);
			atomicAdd(&n[blockIdx.x - 1], hist[index]);
		}
	}
	__syncthreads(); //�����߳�ͬ��

	if (blockIdx.x > 0)
	{
		int index = blockIdx.x - 1;
		float u = sum[0] / (img_height * img_width);
		float w0 = n[index] / (img_height * img_width);
		float u0 = s[index] / n[index];
		if (w0 == 1)
		{
			val[index] = 0;
		}
		else
		{
			val[index] = (u - u0) * (u - u0) * w0 / (1 - w0);
		}
	}
	__syncthreads(); //�����߳�ͬ��

	//���в��ֳ���
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < 256; i++)
		{
			float maxval = 0;
			if (val[i] > maxval)
			{
				maxval = val[i];
				OtsuThresh[0] = i;
				OtsuThresh[1] = val[i];
			}
		}
	}
}

extern cudaError_t thresh_cal(const int* hist, float* sum, float* s, float* n, float* val, int img_width, int img_height, int* OtsuThresh) {
	
	dim3 block_dim(256, 1);
	dim3 grid_dim(257, 1);
	OTSUthresh << <block_dim, grid_dim >> > (hist, sum, s, n, val, img_width, img_height, OtsuThresh);
	return cudaDeviceSynchronize();
}

//CUDA�汾�Ķ�ֵ��
__global__ void kernel_gray_to_otsu_binary(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height, int* hThresh)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= img_width || idy >= img_height) {
		return;
	}

	int local_index = idy * img_width + idx;
	if (img_in[local_index] > hThresh[0])
	{
		img_out[local_index] = 255;  //����Ϊ��ɫ
	}
	else
	{
		img_out[local_index] = 0;    // ����Ϊ��ɫ
	}
}

extern cudaError_t gray_to_otsu_binary(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height, int* hThresh) {

	dim3 block_dim(16, 16);   //�����߳̿�
	dim3 grid_dim = dim3((img_width + block_dim.x - 1) / block_dim.x,
		(img_height + block_dim.y - 1) / block_dim.y);

	kernel_gray_to_otsu_binary << <grid_dim, block_dim >> > (img_in, img_out, img_width, img_height, hThresh);
	return cudaDeviceSynchronize();
}

/* CUDA�汾�µı����� */
//��ʴ����
__global__ void kernel_erosion(unsigned char* img_in, unsigned char* img_ero, int img_width, int img_height, int len)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= img_width || idy >= img_height) {
		return;
	}

	int local_index = idy * img_width + idx;
	img_ero[local_index] = img_in[local_index];
	int wth = (len - 1) / 2;
	if (idx > wth && idx < img_width - wth && idy > wth && idy < img_height - wth)
	{
		for (int w = -wth; w < wth + 1; w++)
		{
			for (int h = -wth; h < wth + 1; h++)
			{
				if (img_in[(idy + h) * img_width + idx + w] < img_ero[local_index])
					img_ero[local_index] = img_in[(idy + h) * img_width + idx + w];
			}
		}
	}
}
extern cudaError_t erosion(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height) {

	dim3 block_dim(16, 16);   //�����߳̿�
	dim3 grid_dim = dim3((img_width + block_dim.x - 1) / block_dim.x,
		(img_height + block_dim.y - 1) / block_dim.y);

	kernel_erosion << <grid_dim, block_dim, 0 >> > (img_in, img_out, img_width, img_height, 7);
	return cudaDeviceSynchronize();
}

//�����㷨
__global__ void kernel_dilation(unsigned char* img_in, unsigned char* img_dil, int img_width, int img_height, int len)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= img_width || idy >= img_height) {
		return;
	}
	int local_index = idy * img_width + idx;
	img_dil[local_index] = img_in[local_index];
	int wth = (len - 1) / 2;

	if (idx > wth && idx < img_width - wth && idy > wth && idy < img_height - wth)
	{
		for (int w = -wth; w < wth + 1; w++)
		{
			for (int h = -wth; h < wth + 1; h++)
			{
				if (img_in[(idy + h) * img_width + idx + w] > img_dil[local_index])
					img_dil[local_index] = img_in[(idy + h) * img_width + idx + w];
			}
		}
	}
}

extern cudaError_t dilation(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height) {

	dim3 block_dim(16, 16);   //�����߳̿�
	dim3 grid_dim = dim3((img_width + block_dim.x - 1) / block_dim.x,
		(img_height + block_dim.y - 1) / block_dim.y);

	kernel_dilation << <grid_dim, block_dim, 0 >> > (img_in, img_out, img_width, img_height, 7);	
	return cudaDeviceSynchronize();
}

///*CUDA�汾�������ͨ����ȡ*/
////������ͨ���ǩ
//__global__ void kernel_init_labels(unsigned const char* img_in, unsigned char* img_out, int img_width, int img_height)   //rows-width,cols-height
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	int idy = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (idx >= img_width || idy >= img_height) {
//		return;
//	}
//
//	int local_index = idy * img_width + idx;
//	const unsigned char pyx = img_in[local_index];
//
//	//������ͨ��
//	const bool pyx_n = (idy > 0) ? (pyx == img_in[(idy - 1) * img_height + idx]) : false;
//	const bool pyx_w = (idx > 0) ? (pyx == img_in[idy * img_height + idx - 1]) : false;
//	const bool pyx_nw = ((idy > 0) && (idx > 0)) ? (pyx == img_in[(idy - 1) * img_height + idx - 1]) : false;
//	const bool pyx_ne = ((idy > 0) && (idx < img_width - 1)) ? (pyx == img_in[(idy - 1) * img_height + idx + 1]) : false;
//
//	// ��ʼ����ǩ
//	// Label will be chosen in the following order: NW > N > NE > W > current position
//	unsigned int label;
//	label = (pyx_nw) ? (idy - 1) * img_height + idx - 1 : idy * img_height + idx;
//	label = (pyx_n) ? (idy - 1) * img_height + idx : label;
//	label = (pyx_ne) ? (idy - 1) * img_height + idx + 1 : label;
//	label = (pyx_w) ? idy * img_height + idx - 1 : label;
//
//	// ��ǩд��
//	img_out[local_index] = label;
//}
////��ǩ�ж�
//__global__ void kernel_label_reduction(unsigned const char* img_in, unsigned char* img_out, int img_width, int img_height) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	int idy = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (idx >= img_width || idy >= img_height) {
//		return;
//	}
//	int local_index = idy * img_width + idx;
//	const unsigned char pyx = img_in[local_index];
//	const bool pyx_nw = ((idy > 0) && (idx > 0)) ? (pyx == img_in[(idy - 1) * img_height + idx - 1]) : false;
//
//		if (!pyx_nw) {
//			// �ж��������ֵ
//			const bool pyx_n = (idy > 0) ? (img_in[(idy - 1) * img_height + idx]) : false;
//			const bool pyx_ne = ((idy > 0) && (idx < img_width - 1)) ? (pyx == img_in[(idy - 1) * img_height + idx + 1]) : false;
//			const bool pyx_w = (idx > 0) ? (pyx == img_in[idy * img_height + idx - 1]) : false;
//
//			if (pyx_w) {
//				if ((pyx_n && pyx_ne) || (pyx_n && !pyx_ne)) {
//					unsigned int label1 = img_out[idy * img_width + idx];
//					unsigned int label2 = img_out[(idy - 1) * img_width + idx + 1];
//
//					// Reduction
//					reduction(img_out, label1, label2);
//				}
//
//				if (!pyx_n && pyx_ne) {
//					unsigned int label1 = img_out[idy * img_width + idx];
//					unsigned int label2 = img_out[idy * img_width + idx - 1];
//
//					// Reduction
//					reduction(img_out, label1, label2);
//				}
//			}
//		}
//	}