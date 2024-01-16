#include "cuda_runtime.h"

#define INF 1e20

/*CUDA版本的距离变换*/

//up to down, down to up
__device__ void likedt1dimvec(unsigned char* dim1data, unsigned char* dim1result, const int img_width, const int img_height)
{
	for (int i = 1; i != img_height; i++)
	{
		if (dim1data[i] > 0)
		{
			dim1result[i] = dim1data[i] + dim1result[i - 1];     //非零元素与前一个元素进行累加
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

/*然后对colpassimg所有行完成一次从左到右扫描*/
__global__ void left_to_rightpass(unsigned char* colpassimg, unsigned char* leftright, const int img_width, const int img_height)
{
	//const int rows = 500;
	//const int cols = 1216;
	
	//block内所有线程合作，完成一行从全局搬运到共享内存
	__shared__ unsigned char rowdata[500];
	int tid = threadIdx.x;
	
	while (tid < img_width)
	{
		int thid = tid + img_width * blockIdx.y;
		rowdata[tid] = colpassimg[thid];
		tid += blockDim.x;
	}
	__syncthreads();

	tid = threadIdx.x;
	//一个线程 对共享内存的数据进行一维距离变换
	__shared__ unsigned char rowdataresult[500];
	if (tid == 0)
	{
		likedt1dimhor(rowdata, rowdataresult, img_width, img_height);
	}
	__syncthreads();

	//block内所有线程合作，将共享内存的距离变换结果搬到全局某行
	while (tid < img_width)
	{
		int thid = tid + img_width * blockIdx.y;
		leftright[thid] = rowdataresult[tid];
		tid += blockDim.x;
	}
}

/*然后对colpassimg所有行完成一次从右到左扫描*/
__global__ void right_to_leftpass(unsigned char* colpassimg, unsigned char* rightleft, const int img_width, const int img_height)
{
	//const int rows = 500;
	//const int cols = 1216;

	//block内所有线程合作，完成一行从全局搬运到共享内存
	__shared__ unsigned char rowdata[500];
	int tid = threadIdx.x;
	while (tid < img_width)
	{
		int thid = tid + img_width * blockIdx.y;
		rowdata[img_width - 1 - tid] = colpassimg[thid];
		tid += blockDim.x;
	}
	__syncthreads();

	tid = threadIdx.x;
	//一个线程 对共享内存的数据进行一维距离变换
	__shared__ unsigned char rowdataresult[500];
	if (tid == 0)
	{
		likedt1dimhor(rowdata, rowdataresult, img_width, img_height);
	}
	__syncthreads();

	//block内所有线程合作，将共享内存的距离变换结果搬到全局某行
	while (tid < img_width)
	{
		int thid = tid + img_width * blockIdx.y;
		rightleft[thid] = rowdataresult[img_width - 1 - tid];
		tid += blockDim.x;
	}
}


/*然后对gpudtimg所有列完成一次从上到下扫描*/
__global__ void up_to_downscan(unsigned char* gpudtimg, unsigned char* updownpassimg, const int img_width, const int img_height)
{
	/*const int rows = 500;
	const int cols = 1216;*/
	int id = blockIdx.y * gridDim.x + blockIdx.x;
	int rowid = threadIdx.x;
	int globalid = id + img_width * rowid;

	//block内所有线程合作，完成一列从全局搬运到共享内存
	__shared__ unsigned char coldata[1216];
	coldata[rowid] = gpudtimg[globalid];
	__syncthreads();


	//一个线程 对共享内存的数据进行一维距离变换
	__shared__ unsigned char coldataresult[1216];
	if (rowid == 0)
	{
		likedt1dimvec(coldata, coldataresult, img_width, img_height);
	}
	__syncthreads();

	//block内所有线程合作，将共享内存的距离变换结果搬到全局某列
	updownpassimg[globalid] = coldataresult[rowid];
}

/*然后对gpudtimg所有列完成一次从下到上扫描*/
__global__ void down_to_upscan(unsigned char* gpudtimg, unsigned char* downuppassimg, const int img_width, const int img_height)
{
	/*const int rows = 500;
	const int cols = 1216;*/
	int id = blockIdx.y * gridDim.x + blockIdx.x;
	int rowid = threadIdx.x;
	int globalid = id + img_width * rowid;

	//block内所有线程合作，完成一列从全局搬运到共享内存
	__shared__ unsigned char coldata[1216];
	coldata[img_height - 1 - rowid] = gpudtimg[globalid];
	__syncthreads();


	//一个线程 对共享内存的数据进行一维距离变换
	__shared__ unsigned char coldataresult[1216];
	__syncthreads();
	if (rowid == 0)
	{
		likedt1dimvec(coldata, coldataresult, img_width, img_height);
	}
	__syncthreads();

	//block内所有线程合作，将共享内存的距离变换结果搬到全局某列
	downuppassimg[globalid] = coldataresult[img_height - 1 - rowid];
}

/*最后对图像结果每个数据开根号，得到最终结果*/
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

	dim3 block_dim(16, 16);   //定义线程块
	dim3 grid_dim = dim3((img_width + block_dim.x - 1) / block_dim.x,
		(img_height + block_dim.y - 1) / block_dim.y);

	left_to_rightpass << <grid_dim, block_dim, 0 >> > (img_in, leftright, img_width, img_height);
	right_to_leftpass << <grid_dim, block_dim, 0 >> > (img_in, rightleft, img_width, img_height);
	up_to_downscan << <grid_dim, block_dim, 0 >> > (img_in, updown, img_width, img_height);
	down_to_upscan << <grid_dim, block_dim, 0 >> > (img_in, downup, img_width, img_height);
	likedtresult << <grid_dim, block_dim, 0 >> > (updown, downup, leftright, rightleft, dtresult, img_width, img_height);
	return cudaDeviceSynchronize();
}
