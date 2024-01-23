#include <iostream>
#include "GalaxyException.h"
#include "camera_on.h"
#include "filter.h"

using namespace cv;
using namespace std;

std::mutex imageMutex;
Mat capturedImage;
ori_image* ori;

int frameCount = 0;
auto startTime = std::chrono::steady_clock::now();

void CSampleDeviceOfflineEventHandler::DoOnDeviceOfflineEvent(void* pUserParam)
{
	std::cout << "收到设备掉线事件!" << std::endl;
}

void CSampleFeatureEventHandler::DoOnFeatureEvent(const GxIAPICPP::gxstring& strFeatureName, void* pUserParam)
	{
	std::cout << "收到曝光结束事件!" << endl;
	}


void CSampleCaptureEventHandler ::DoOnImageCaptured(CImageDataPointer& objImageDataPointer, void* pUserParam)
	{

		std::cout << "收到一帧图像!" << endl;
		std::cout << "ImageInfo: " << objImageDataPointer->GetStatus() << endl;
		std::cout << "ImageInfo: " << objImageDataPointer->GetWidth() << endl;
		std::cout << "ImageInfo: " << objImageDataPointer->GetHeight() << endl;
		std::cout << "ImageInfo: " << objImageDataPointer->GetPayloadSize() << endl;

		Mat img;
		int size_max;

		img.create(objImageDataPointer->GetHeight(), objImageDataPointer->GetWidth(), CV_8UC3);                   //创建一个新的Mat,用于图像数据
		void* pRGB24Buffer = NULL;
		//假设原始数据是BayerRG8图像，用于储存图像数据的指针
		pRGB24Buffer = objImageDataPointer->ConvertToRGB24(GX_BIT_0_7, GX_RAW2RGB_NEIGHBOUR, true);               //将原始图像数据转换为RGB24格式，并将结果赋值给了pRGB24Buffer指针变量
		memcpy(img.data, pRGB24Buffer, (objImageDataPointer->GetHeight()) * (objImageDataPointer->GetWidth()) * 3);
		flip(img, img, 0);                  //沿x轴翻转

		// 如果图像的颜色通道顺序是BGR而不是RGB，则需要进行颜色通道顺序转换

		QImage qimage(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888);
		qimage = qimage.rgbSwapped();
		
		std::cout << "帧数：" << objImageDataPointer->GetFrameID() << endl;

		++frameCount;

		auto endTime = std::chrono::steady_clock::now();
		double elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		if (elapsedSeconds >= 1.0) {
			double fps = frameCount / elapsedSeconds;

			startTime = endTime;
			frameCount = 0;
		}
		ori->showImage(qimage, objImageDataPointer->GetWidth(), objImageDataPointer->GetHeight(), elapsedSeconds);
	}

int camera_on(ori_image* ori_img)
{	

	//声明事件回调对象指针
	IDeviceOfflineEventHandler* pDeviceOfflineEventHandler = NULL;///<掉线事件回调对象
	IFeatureEventHandler* pFeatureEventHandler = NULL;///<远端设备事件回调对象
	ICaptureEventHandler* pCaptureEventHandler = NULL;///<采集回调对象

	//初始化
	IGXFactory::GetInstance().Init();

	try
	{
		do
		{
			//枚举设备
			gxdeviceinfo_vector vectorDeviceInfo;
			IGXFactory::GetInstance().UpdateDeviceList(1000, vectorDeviceInfo);
			cout << "线程ID: " << std::this_thread::get_id() << std::endl;
			if (0 == vectorDeviceInfo.size())
			{
				cout << "无可用设备!" << endl;  
				break;
			}
			cout << vectorDeviceInfo[0].GetVendorName() << endl;
			cout << vectorDeviceInfo[0].GetSN() << endl;
			//打开第一台设备以及设备下面第一个流
			CGXDevicePointer ObjDevicePtr = IGXFactory::GetInstance().OpenDeviceBySN(
				vectorDeviceInfo[0].GetSN(),
				GX_ACCESS_EXCLUSIVE);
			CGXStreamPointer ObjStreamPtr = ObjDevicePtr->OpenStream(0);


			//注册设备掉线事件【目前只有千兆网系列相机支持此事件通知】
			GX_DEVICE_OFFLINE_CALLBACK_HANDLE hDeviceOffline = NULL;
			pDeviceOfflineEventHandler = new CSampleDeviceOfflineEventHandler();
			hDeviceOffline = ObjDevicePtr->RegisterDeviceOfflineCallback(pDeviceOfflineEventHandler, NULL);

			//获取远端设备属性控制器
			CGXFeatureControlPointer ObjFeatureControlPtr = ObjDevicePtr->GetRemoteFeatureControl();

			//设置曝光时间(示例中写死us,只是示例,并不代表真正可工作参数)
			ObjFeatureControlPtr->GetFloatFeature("ExposureTime")->SetValue(30000);

			//注册远端设备事件:曝光结束事件【目前只有千兆网系列相机支持曝光结束事件】
			//选择事件源
			//ObjFeatureControlPtr->GetEnumFeature("EventSelector")->SetValue("ExposureEnd");

			//使能事件
			//ObjFeatureControlPtr->GetEnumFeature("EventNotification")->SetValue("On");
			//GX_FEATURE_CALLBACK_HANDLE hFeatureEvent = NULL;
			//pFeatureEventHandler = new CSampleFeatureEventHandler();
			//hFeatureEvent = ObjFeatureControlPtr->RegisterFeatureCallback(
			//	"EventExposureEnd",
			//	pFeatureEventHandler,
			//	NULL);

			//注册回调采集
			pCaptureEventHandler = new CSampleCaptureEventHandler();
			ori = ori_img;
			ObjStreamPtr->RegisterCaptureCallback(pCaptureEventHandler, NULL);

			//发送开采命令
			ObjStreamPtr->StartGrab();
			ObjFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();

			//此时开采成功,控制台打印信息,直到输入任意键继续
			getchar();

			//发送停采命令
			ObjFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
			ObjStreamPtr->StopGrab();

			//注销采集回调
			ObjStreamPtr->UnregisterCaptureCallback();

			//注销远端设备事件
			//ObjFeatureControlPtr->UnregisterFeatureCallback(hFeatureEvent);

			//注销设备掉线事件
			//ObjDevicePtr->UnregisterDeviceOfflineCallback(hDeviceOffline);

			//释放资源
			ObjStreamPtr->Close();
			ObjDevicePtr->Close();
		}   while (0);
	}

	catch (CGalaxyException& e)
	{
		cout << "错误码: " << e.GetErrorCode() << endl;
		cout << "错误描述信息: " << e.what() << endl;
	}
	catch (std::exception& e)
	{
		cout << "错误描述信息: " << e.what() << endl;
	}

	//反初始化库
	IGXFactory::GetInstance().Uninit();

	//销毁事件回调指针
	if (NULL != pCaptureEventHandler)
	{
		delete pCaptureEventHandler;
		pCaptureEventHandler = NULL;
	}
	if (NULL != pDeviceOfflineEventHandler)
	{
		delete pDeviceOfflineEventHandler;
		pDeviceOfflineEventHandler = NULL;
	}
	if (NULL != pFeatureEventHandler)
	{
		delete pFeatureEventHandler;
		pFeatureEventHandler = NULL;
	}
	return 0;

}
