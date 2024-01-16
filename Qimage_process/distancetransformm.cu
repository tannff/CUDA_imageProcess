#include "cuda_runtime.h"

#define INF 1e20

/*CUDA�汾�ľ���任*/
//��ʼ������������е�Ԫ������Ϊ0������Ԫ������Ϊ����
__global__ void preparefordt(unsigned char* bwimg, float* gpudtimg, const int img_width, const int img_height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int imgid = idy * img_width + idx;
	if (bwimg[imgid] == 0)
	{
		gpudtimg[imgid] = 0;
	}
	else
	{
		gpudtimg[imgid] = INF;
	}
}

/*
f----�����һά����
d----һά����f��Ӧ�ľ���任���
*/
__device__ void dt1dimgpu(float* f, float* d)
{
	const int n = 500;
	//float d[n];
	int v[n];
	float z[n + 1];
	int k = 0;
	v[0] = 0;
	z[0] = -INF;
	z[1] = +INF;
	for (int q = 1; q <= n - 1; q++)
	{
		float s = ((f[q] + q * q) - (f[v[k]] + (v[k]) * (v[k]))) / (2 * q - 2 * v[k]);
		while (s <= z[k])
		{
			k--;
			s = ((f[q] + q * q) - (f[v[k]] + (v[k]) * (v[k]))) / (2 * q - 2 * v[k]);
		}
		k++;
		v[k] = q;
		z[k] = s;
		z[k + 1] = +INF;
	}

	k = 0;
	for (int q = 0; q <= n - 1; q++)
	{
		while (z[k + 1] < q)
		{
			k++;
		}
		d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
	}
}

/*
f----�����һά����
d----һά����f��Ӧ�ľ���任���
*/
__device__ void dt1dimrowgpu(float* f, float* d)
{
	const int n = 1216;
	int v[n];
	float z[n + 1];
	int k = 0;
	v[0] = 0;
	z[0] = -INF;
	z[1] = +INF;
	for (int q = 1; q <= n - 1; q++)
	{
		float s = ((f[q] + q * q) - (f[v[k]] + (v[k]) * (v[k]))) / (2 * q - 2 * v[k]);
		while (s <= z[k])
		{
			k--;
			s = ((f[q] + q * q) - (f[v[k]] + (v[k]) * (v[k]))) / (2 * q - 2 * v[k]);
		}
		k++;
		v[k] = q;
		z[k] = s;
		z[k + 1] = +INF;
	}

	k = 0;
	for (int q = 0; q <= n - 1; q++)
	{
		while (z[k + 1] < q)
		{
			k++;
		}
		d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
	}
}

//up to down, down to up
__device__ void likedt1dimvec(unsigned char* dim1data, unsigned char* dim1result, const int img_width, const int img_height)
{
	for (int i = 1; i != img_height; i++)
	{
		if (dim1data[i] > 0)
		{
			dim1result[i] = dim1data[i] + dim1result[i - 1];     //����Ԫ����ǰһ��Ԫ�ؽ����ۼ�
		}
	}
}

//left to right , right to left
__device__ void likedt1dimhor(unsigned char* dim1data, unsigned char* dim1result, const int img_width, const int img_height)
{
	for (int i = 1; i != img_width; i++)
	{
		if (dim1data[i] > 0)
		{
			dim1result[i] = dim1data[i] + dim1result[i - 1];
		}
	}
}

/*Ȼ���colpassimg���������һ�δ�����ɨ��*/
__global__ void left_to_rightpass(unsigned char* colpassimg, unsigned char* leftright, const int img_width, const int img_height)
{
	//const int rows = 500;
	//const int cols = 1216;
	
	//block�������̺߳��������һ�д�ȫ�ְ��˵������ڴ�
	__shared__ unsigned char rowdata[img_width];
	int tid = threadIdx.x;
	while (tid < img_width)
	{
		int thid = tid + img_width * blockIdx.y;
		rowdata[tid] = colpassimg[thid];
		tid += blockDim.x;
	}
	__syncthreads();

	tid = threadIdx.x;
	//һ���߳� �Թ����ڴ�����ݽ���һά����任
	__shared__ unsigned char rowdataresult[img_width];
	if (tid == 0)
	{
		likedt1dimhor(rowdata, rowdataresult, img_width, img_height);
	}
	__syncthreads();

	//block�������̺߳������������ڴ�ľ���任����ᵽȫ��ĳ��
	while (tid < img_width)
	{
		int thid = tid + img_width * blockIdx.y;
		leftright[thid] = rowdataresult[tid];
		tid += blockDim.x;
	}
}

/*Ȼ���colpassimg���������һ�δ��ҵ���ɨ��*/
__global__ void right_to_leftpass(unsigned char* colpassimg, unsigned char* rightleft, const int img_width, const int img_height)
{
	//const int rows = 500;
	//const int cols = 1216;

	//block�������̺߳��������һ�д�ȫ�ְ��˵������ڴ�
	__shared__ unsigned char rowdata[img_width];
	int tid = threadIdx.x;
	while (tid < img_width)
	{
		int thid = tid + img_width * blockIdx.y;
		rowdata[img_width - 1 - tid] = colpassimg[thid];
		tid += blockDim.x;
	}
	__syncthreads();

	tid = threadIdx.x;
	//һ���߳� �Թ����ڴ�����ݽ���һά����任
	__shared__ unsigned char rowdataresult[img_width];
	if (tid == 0)
	{
		likedt1dimhor(rowdata, rowdataresult, img_width, img_height);
	}
	__syncthreads();

	//block�������̺߳������������ڴ�ľ���任����ᵽȫ��ĳ��
	while (tid < img_width)
	{
		int thid = tid + img_width * blockIdx.y;
		rightleft[thid] = rowdataresult[img_width - 1 - tid];
		tid += blockDim.x;
	}
}


/*Ȼ���gpudtimg���������һ�δ��ϵ���ɨ��*/
__global__ void up_to_downscan(unsigned char* gpudtimg, unsigned char* updownpassimg, const int img_width, const int img_height)
{
	/*const int rows = 500;
	const int cols = 1216;*/
	int id = blockIdx.y * gridDim.x + blockIdx.x;
	int rowid = threadIdx.x;
	int globalid = id + img_width * rowid;

	//block�������̺߳��������һ�д�ȫ�ְ��˵������ڴ�
	__shared__ unsigned char coldata[img_height];
	coldata[rowid] = gpudtimg[globalid];
	__syncthreads();


	//һ���߳� �Թ����ڴ�����ݽ���һά����任
	__shared__ unsigned char coldataresult[img_height];
	if (rowid == 0)
	{
		likedt1dimvec(coldata, coldataresult, img_width, img_height);
	}
	__syncthreads();

	//block�������̺߳������������ڴ�ľ���任����ᵽȫ��ĳ��
	updownpassimg[globalid] = coldataresult[rowid];
}

/*Ȼ���gpudtimg���������һ�δ��µ���ɨ��*/
__global__ void down_to_upscan(unsigned char* gpudtimg, unsigned char* downuppassimg, const int img_width, const int img_height)
{
	/*const int rows = 500;
	const int cols = 1216;*/
	int id = blockIdx.y * gridDim.x + blockIdx.x;
	int rowid = threadIdx.x;
	int globalid = id + img_width * rowid;

	//block�������̺߳��������һ�д�ȫ�ְ��˵������ڴ�
	__shared__ unsigned char coldata[img_height];
	coldata[img_height - 1 - rowid] = gpudtimg[globalid];
	__syncthreads();


	//һ���߳� �Թ����ڴ�����ݽ���һά����任
	__shared__ unsigned char coldataresult[img_height];
	__syncthreads();
	if (rowid == 0)
	{
		likedt1dimvec(coldata, coldataresult, img_width, img_height);
	}
	__syncthreads();

	//block�������̺߳������������ڴ�ľ���任����ᵽȫ��ĳ��
	downuppassimg[globalid] = coldataresult[img_height - 1 - rowid];
}

/*����ͼ����ÿ�����ݿ����ţ��õ����ս��*/
__global__ void likedtresult(unsigned char* updown, unsigned char* downup, unsigned char* leftright, unsigned char* rightleft, unsigned char* dtresult, const int img_width, const int img_height)
{
	/*int rows = 500;
	int cols = 1216;*/
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int imgid = idy * img_width + idx;

	unsigned char udvalue = updown[imgid];
	unsigned char minvalue = udvalue;
	unsigned char duvalue = downup[imgid];
	if (minvalue > duvalue)
	{
		minvalue = duvalue;
	}
	unsigned char lrvalue = leftright[imgid];
	if (minvalue > lrvalue)
	{
		minvalue = lrvalue;
	}
	unsigned char rlvalue = rightleft[imgid];
	if (minvalue > rlvalue)
	{
		minvalue = rlvalue;
	}

	dtresult[imgid] = minvalue;
}

extern cudaError_t distancetransform(unsigned char* img_in, unsigned char* updown, unsigned char* downup, unsigned char* leftright, unsigned char* rightleft, unsigned char* dtresult, const int img_width, const int img_height) {

	dim3 block_dim(2, 500);   //�����߳̿�
	dim3 grid_dim = dim3((img_width + block_dim.x - 1) / block_dim.x,
		(img_height + block_dim.y - 1) / block_dim.y);

	left_to_rightpass << <grid_dim, block_dim, 0 >> > (img_in, leftright, img_width, img_height);
	right_to_leftpass << <grid_dim, block_dim, 0 >> > (img_in, rightleft, img_width, img_height);
	up_to_downscan << <grid_dim, block_dim, 0 >> > (img_in, updown, img_width, img_height);
	down_to_upscan << <grid_dim, block_dim, 0 >> > (img_in, downup, img_width, img_height);
	likedtresult << <grid_dim, block_dim, 0 >> > (updown, downup, leftright, rightleft, dtresult, img_width, img_height);
	return cudaDeviceSynchronize();
}