#pragma once

#include <GalaxyIncludes.h>
#include <opencv2/opencv.hpp>
#include "ori_image.h"

extern bool isProcessing;
extern std::mutex mtx;

int camera_on(ori_image* img);

//用户继承掉线事件处理类
class CSampleDeviceOfflineEventHandler : public IDeviceOfflineEventHandler
{
public:
	void DoOnDeviceOfflineEvent(void* pUserParam);
};

//用户继承属性更新事件处理类

class CSampleFeatureEventHandler : public IFeatureEventHandler
{
public:
	void DoOnFeatureEvent(const GxIAPICPP::gxstring& strFeatureName, void* pUserParam);
};

//用户继承采集事件处理类
class CSampleCaptureEventHandler : public ICaptureEventHandler
{
public:
	void DoOnImageCaptured(CImageDataPointer& objImageDataPointer, void* pUserParam);

};


