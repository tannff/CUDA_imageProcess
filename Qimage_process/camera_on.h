#pragma once

#include <GalaxyIncludes.h>
#include <opencv2/opencv.hpp>
#include "ori_image.h"

extern bool isProcessing;
extern std::mutex mtx;

int camera_on(ori_image* img);

//�û��̳е����¼�������
class CSampleDeviceOfflineEventHandler : public IDeviceOfflineEventHandler
{
public:
	void DoOnDeviceOfflineEvent(void* pUserParam);
};

//�û��̳����Ը����¼�������

class CSampleFeatureEventHandler : public IFeatureEventHandler
{
public:
	void DoOnFeatureEvent(const GxIAPICPP::gxstring& strFeatureName, void* pUserParam);
};

//�û��̳вɼ��¼�������
class CSampleCaptureEventHandler : public ICaptureEventHandler
{
public:
	void DoOnImageCaptured(CImageDataPointer& objImageDataPointer, void* pUserParam);

};


