#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

#define STRONG_EDGE 0xFFFF
#define NON_EDGE 0x0

/* CUDA版本下的边缘检测算法 */
//高斯滤波
template<class T>
__device__ T clamp(T value, T min, T max) {
    T result;
    result = value < min ? min : value;
    result = value > max ? max : value;
    return result;
}

__global__ void kernel_gaussian_filter(unsigned char* img_in, unsigned char* img_gauss, int img_width, int img_height, int filterWidth, float* filter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;    
    if (idx >= img_width || idy >= img_height) {
        return;
    }
    int local_index = idy * img_width + idx;
    float pixelcolor = 0.0f;

    for (int i = 0; i < filterWidth; i++)
    {
        for (int j = 0; j < filterWidth; j++)
        {
            // 不超出图像边界
            int clamp_x = __min(__max(idx + j - filterWidth / 2, 0), img_width - 1);
            int clamp_y = __min(__max(idy + i - filterWidth / 2, 0), img_height - 1);

            // 结果计算
            float avg = filter[i * filterWidth + j];
            pixelcolor += avg * static_cast<float>(img_in[clamp_y * img_width + clamp_x]);
        }
    }
    // 写入结果
    img_gauss[local_index] = clamp(pixelcolor, 0.f, 255.f);
}
extern cudaError_t gaussian_filter(unsigned char* img_in, unsigned char* img_gauss, int img_width, int img_height, int filterWidth, float* filter) {

    dim3 block_dim(16, 16);   //定义线程块
    dim3 grid_dim = dim3((img_width + block_dim.x - 1) / block_dim.x,
        (img_height + block_dim.y - 1) / block_dim.y);
  
    kernel_gaussian_filter << <grid_dim, block_dim >> > (img_in, img_gauss, img_width, img_height, filterWidth, filter);
    return cudaDeviceSynchronize();
}

//Sobel算子计算像素梯度
__global__ void kernel_sobel_intensity_gradient(unsigned char* img_in, unsigned char* img_sobel, int* Gx, int* Gy, int img_width, int img_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= img_width || idy >= img_height) {
        return;
    }
    int local_index = idy * img_width + idx;

    if (idx > 0 && idx < img_width - 1 && idy > 0 && idy < img_height - 1)
    { 
        Gx[local_index] = img_in[(idy - 1) * img_width + idx + 1] + 2.0 * img_in[idy * img_width + idx + 1] + img_in[(idy + 1) * img_width + idx + 1]       //idy * img_width + idx为当前像素
            - (img_in[(idy - 1) * img_width + idx - 1] + 2.0 * img_in[idy * img_width + idx - 1] + img_in[(idy + 1) * img_width + idx - 1]);
        Gy[local_index] = img_in[(idy - 1) * img_width + idx - 1] + 2.0 * img_in[(idy - 1) * img_width + idx] + img_in[(idy - 1) * img_width + idx + 1]
            - (img_in[(idy + 1) * img_width + idx - 1] + 2.0 * img_in[(idy + 1) * img_width + idx] + img_in[(idy + 1) * img_width + idx + 1]);
        img_sobel[local_index] = (abs(Gx[local_index]) + abs(Gy[local_index])) / 2.0;
    }
}
extern cudaError_t sobel_intensity_gradient(unsigned char* img_in, unsigned char* img_sobel, int* Gx, int* Gy, int img_width, int img_height) {

    dim3 block_dim(16, 16);   //定义线程块
    dim3 grid_dim = dim3((img_width + block_dim.x - 1) / block_dim.x,
        (img_height + block_dim.y - 1) / block_dim.y);

    kernel_sobel_intensity_gradient << <grid_dim, block_dim >> > (img_in, img_sobel, Gx, Gy, img_width, img_height);
    return cudaDeviceSynchronize();
}

//非极大值抑制
__global__ void kernel_non_max(unsigned char* img_in, unsigned char* img_nms, int* totalGx, int* totalGy, int img_width, int img_height)
{
    const int SUPPRESSED = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= img_width || idy >= img_height) {
        return;
    }
    int local_index = idy * img_width + idx;
    float alpha;
    float mag1, mag2;
    int Gx = totalGx[local_index];
    int Gy = totalGy[local_index];

    if (idx == 0 || idx == (img_width - 1) || idy == 0 || idy == (img_height - 1)) {    //判断是否在图像边界
        img_nms[local_index] = 0;
    }
    else  // 若不在边界上
    {         
      if (img_in == 0)     //若梯度变化为0，则也不属于边缘
          img_nms[local_index] = SUPPRESSED;
      else {
        if (Gx >= 0)  //Gx >= 0, Gy >= 0
          {
            if (Gy >= 0)
            {
                if ((Gx - Gy) >= 0)       // 方向1： SE-E
                {
                    alpha = (float)Gy / Gx;
                    mag1 = (1 - alpha) * img_in[idy * img_width + idx + 1] + alpha * img_in[(idy + 1) * img_width + idx + 1];
                    mag2 = (1 - alpha) * img_in[idy * img_width + idx - 1] + alpha * img_in[(idy - 1) * img_width + idx - 1];
                }
                else                      // 方向2： SE-S
                {
                    alpha = (float)Gx / Gy;
                    mag1 = (1 - alpha) * img_in[(idy + 1) * img_width + idx] + alpha * img_in[(idy + 1) * img_width + idx + 1];
                    mag2 = (1 - alpha) * img_in[(idy - 1) * img_width + idx] + alpha * img_in[(idy - 1) * img_width + idx - 1];
                }
            }
            else  //Gx >= 0, Gy < 0
            {
                if ((Gx + Gy) >= 0)       // 方向8： NE-E
                {
                    alpha = (float)-Gy / Gx;
                    mag1 = (1 - alpha) * img_in[idy * img_width + idx + 1] + alpha * img_in[(idy - 1) * img_width + idx + 1];
                    mag2 = (1 - alpha) * img_in[idy * img_width + idx - 1] + alpha * img_in[(idy + 1) * img_width + idx - 1];
                }
                else                      // 方向7： NE-N
                {
                    alpha = (float)Gx / -Gy;
                    mag1 = (1 - alpha) * img_in[(idy - 1) * img_width + idx] + alpha * img_in[(idy - 1) * img_width + idx + 1];
                    mag2 = (1 - alpha) * img_in[(idy + 1) * img_width + idx] + alpha * img_in[(idy - 1) * img_width + idx - 1];
                }
            }
        }
        else
        {
            if (Gy >= 0)  //Gx < 0, Gy >= 0
            {
                  if ((Gx + Gy) >= 0)    //  方向3： SW-S
                  {
                      alpha = (float)-Gx / Gy;
                      mag1 = (1 - alpha) * img_in[(idy + 1) * img_width + idx] + alpha * img_in[(idy + 1) * img_width + idx - 1];
                      mag2 = (1 - alpha) * img_in[(idy - 1) * img_width + idx] + alpha * img_in[(idy - 1) * img_width + idx + 1];
                  }
                  else                   // 方向4： SW-W
                  {
                       alpha = (float)Gy / -Gx;
                       mag1 = (1 - alpha) * img_in[idy * img_width + idx - 1] + alpha * img_in[(idy + 1) * img_width + idx - 1];
                       mag2 = (1 - alpha) * img_in[idy * img_width + idx + 1] + alpha * img_in[(idy - 1) * img_width + idx + 1];
                  }
            }
            else    //Gx < 0, Gy < 0
            {
                  if ((-Gx + Gy) >= 0)   //  方向5： NW-W
                  {
                       alpha = (float)Gy / Gx;
                       mag1 = (1 - alpha) * img_in[idy * img_width + idx - 1] + alpha * img_in[(idy - 1) * img_width + idx - 1];
                       mag2 = (1 - alpha) * img_in[idy * img_width + idx + 1] + alpha * img_in[(idy + 1) * img_width + idx + 1];
                  }
                  else                   //  方向6： NW-N
                  {
                       alpha = (float)Gx / Gy;
                       mag1 = (1 - alpha) * img_in[(idy - 1) * img_width + idx] + alpha * img_in[(idy - 1) * img_width + idx - 1];
                       mag2 = (1 - alpha) * img_in[(idy + 1) * img_width + idx] + alpha * img_in[(idy + 1) * img_width + idx + 1];
                  }
            }
        }
        if ((img_in[local_index] < mag1) || (img_in[local_index] < mag2))
           img_nms[local_index] = SUPPRESSED;
        else
        {
            img_nms[local_index] = img_in[local_index]; 
        }
      } // END OF ELSE (mag != 0)
    } // END OF FOR(j)
} // END OF FOR(i)
extern cudaError_t non_max(unsigned char* img_in, unsigned char* img_nms, int* Gx, int* Gy, int img_width, int img_height) {

    dim3 block_dim(16, 16);   //定义线程块
    dim3 grid_dim = dim3((img_width + block_dim.x - 1) / block_dim.x,
        (img_height + block_dim.y - 1) / block_dim.y);

    kernel_non_max << <grid_dim, block_dim >> > (img_in, img_nms, Gx, Gy, img_width, img_height);
    return cudaDeviceSynchronize();
}
//阈值滞后处理
__device__ void trace_immed_neighbors(unsigned char* img_in, unsigned char* img_trace, int idx, int idy, int t_low, int img_width, int img_height)
{
    unsigned n, s, e, w;   //上下左右
    unsigned nw, ne, sw, se;  //左上，右上，左下，右下

    n = (idy - 1)* img_width + idx;
    nw = n - 1;
    ne = n + 1;
    s = (idy + 1) * img_width + idx;
    sw = s - 1;
    se = s + 1;
    w = idy * img_width + idx - 1;
    e = idy * img_width + idx + 1;

    if (img_in[nw] >= t_low) {
        img_trace[nw] = STRONG_EDGE;
    }
    if (img_in[n] >= t_low) {
        img_trace[n] = STRONG_EDGE;
    }
    if (img_in[ne] >= t_low) {
        img_trace[ne] = STRONG_EDGE;
    }
    if (img_in[w] >= t_low) {
        img_trace[w] = STRONG_EDGE;
    }
    if (img_in[e] >= t_low) {
        img_trace[e] = STRONG_EDGE;
    }
    if (img_in[sw] >= t_low) {
        img_trace[sw] = STRONG_EDGE;
    }
    if (img_in[s] >= t_low) {
        img_trace[s] = STRONG_EDGE;
    }
    if (img_in[se] >= t_low) {
        img_trace[se] = STRONG_EDGE;
    }
}
//滞后高阈值
__global__ void kernel_hysteresis_high(unsigned char* img_in, unsigned char* img_trace, unsigned* strong_edge_mask, int t_high, int img_width, int img_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= img_width || idy >= img_height) {
        return;
    }
    int local_index = idy * img_width + idx;
    if (img_in[local_index] > t_high) {
        strong_edge_mask[local_index] = 1;
        img_trace[local_index] = STRONG_EDGE;
    }
    else {
        strong_edge_mask[local_index] = 0;
        img_trace[local_index] = NON_EDGE;
        }
    }
//滞后低阈值
__global__ void kernel_hysteresis_low(unsigned char* img_in, unsigned char* img_trace, unsigned* strong_edge_mask, int t_low, int img_width, int img_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= img_width || idy >= img_height) {
        return;
    }
    int local_index = idy * img_width + idx;
    //排除像素越界
    if ((idy > 0) && (idy < img_height - 1)     //排除首尾行
        && (idx > 0) && (idx < img_width - 1))  //排除首尾列
    {
        if (1 == strong_edge_mask[local_index]) { /* if this pixel was previously found to be a strong edge */
            trace_immed_neighbors(img_in, img_trace, idx, idy, t_low, img_width, img_height);
        }
    }
}
extern cudaError_t hysteresis(unsigned char* img_in, unsigned char* img_high, unsigned char* img_trace, unsigned* strong_edge_mask, int t_high, int t_low, int img_width, int img_height) {

    dim3 block_dim(16, 16);   //定义线程块
    dim3 grid_dim = dim3((img_width + block_dim.x - 1) / block_dim.x,
        (img_height + block_dim.y - 1) / block_dim.y);

    kernel_hysteresis_high << <grid_dim, block_dim >> > (img_in, img_high, strong_edge_mask, t_high, img_width, img_height);
    kernel_hysteresis_low << <grid_dim, block_dim >> > (img_high, img_trace, strong_edge_mask, t_low, img_width, img_height);
    return cudaDeviceSynchronize();
}