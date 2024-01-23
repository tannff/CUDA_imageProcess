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
	std::cout << "�յ��豸�����¼�!" << std::endl;
}

void CSampleFeatureEventHandler::DoOnFeatureEvent(const GxIAPICPP::gxstring& strFeatureName, void* pUserParam)
	{
	std::cout << "�յ��ع�����¼�!" << endl;
	}


void CSampleCaptureEventHandler ::DoOnImageCaptured(CImageDataPointer& objImageDataPointer, void* pUserParam)
	{

		std::cout << "�յ�һ֡ͼ��!" << endl;
		std::cout << "ImageInfo: " << objImageDataPointer->GetStatus() << endl;
		std::cout << "ImageInfo: " << objImageDataPointer->GetWidth() << endl;
		std::cout << "ImageInfo: " << objImageDataPointer->GetHeight() << endl;
		std::cout << "ImageInfo: " << objImageDataPointer->GetPayloadSize() << endl;

		Mat img;
		int size_max;

		img.create(objImageDataPointer->GetHeight(), objImageDataPointer->GetWidth(), CV_8UC3);                   //����һ���µ�Mat,����ͼ������
		void* pRGB24Buffer = NULL;
		//����ԭʼ������BayerRG8ͼ�����ڴ���ͼ�����ݵ�ָ��
		pRGB24Buffer = objImageDataPointer->ConvertToRGB24(GX_BIT_0_7, GX_RAW2RGB_NEIGHBOUR, true);               //��ԭʼͼ������ת��ΪRGB24��ʽ�����������ֵ����pRGB24Bufferָ�����
		memcpy(img.data, pRGB24Buffer, (objImageDataPointer->GetHeight()) * (objImageDataPointer->GetWidth()) * 3);
		flip(img, img, 0);                  //��x�ᷭת

		// ���ͼ�����ɫͨ��˳����BGR������RGB������Ҫ������ɫͨ��˳��ת��

		QImage qimage(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888);
		qimage = qimage.rgbSwapped();
		
		std::cout << "֡����" << objImageDataPointer->GetFrameID() << endl;

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

	//�����¼��ص�����ָ��
	IDeviceOfflineEventHandler* pDeviceOfflineEventHandler = NULL;///<�����¼��ص�����
	IFeatureEventHandler* pFeatureEventHandler = NULL;///<Զ���豸�¼��ص�����
	ICaptureEventHandler* pCaptureEventHandler = NULL;///<�ɼ��ص�����

	//��ʼ��
	IGXFactory::GetInstance().Init();

	try
	{
		do
		{
			//ö���豸
			gxdeviceinfo_vector vectorDeviceInfo;
			IGXFactory::GetInstance().UpdateDeviceList(1000, vectorDeviceInfo);
			cout << "�߳�ID: " << std::this_thread::get_id() << std::endl;
			if (0 == vectorDeviceInfo.size())
			{
				cout << "�޿����豸!" << endl;  
				break;
			}
			cout << vectorDeviceInfo[0].GetVendorName() << endl;
			cout << vectorDeviceInfo[0].GetSN() << endl;
			//�򿪵�һ̨�豸�Լ��豸�����һ����
			CGXDevicePointer ObjDevicePtr = IGXFactory::GetInstance().OpenDeviceBySN(
				vectorDeviceInfo[0].GetSN(),
				GX_ACCESS_EXCLUSIVE);
			CGXStreamPointer ObjStreamPtr = ObjDevicePtr->OpenStream(0);


			//ע���豸�����¼���Ŀǰֻ��ǧ����ϵ�����֧�ִ��¼�֪ͨ��
			GX_DEVICE_OFFLINE_CALLBACK_HANDLE hDeviceOffline = NULL;
			pDeviceOfflineEventHandler = new CSampleDeviceOfflineEventHandler();
			hDeviceOffline = ObjDevicePtr->RegisterDeviceOfflineCallback(pDeviceOfflineEventHandler, NULL);

			//��ȡԶ���豸���Կ�����
			CGXFeatureControlPointer ObjFeatureControlPtr = ObjDevicePtr->GetRemoteFeatureControl();

			//�����ع�ʱ��(ʾ����д��us,ֻ��ʾ��,�������������ɹ�������)
			ObjFeatureControlPtr->GetFloatFeature("ExposureTime")->SetValue(30000);

			//ע��Զ���豸�¼�:�ع�����¼���Ŀǰֻ��ǧ����ϵ�����֧���ع�����¼���
			//ѡ���¼�Դ
			//ObjFeatureControlPtr->GetEnumFeature("EventSelector")->SetValue("ExposureEnd");

			//ʹ���¼�
			//ObjFeatureControlPtr->GetEnumFeature("EventNotification")->SetValue("On");
			//GX_FEATURE_CALLBACK_HANDLE hFeatureEvent = NULL;
			//pFeatureEventHandler = new CSampleFeatureEventHandler();
			//hFeatureEvent = ObjFeatureControlPtr->RegisterFeatureCallback(
			//	"EventExposureEnd",
			//	pFeatureEventHandler,
			//	NULL);

			//ע��ص��ɼ�
			pCaptureEventHandler = new CSampleCaptureEventHandler();
			ori = ori_img;
			ObjStreamPtr->RegisterCaptureCallback(pCaptureEventHandler, NULL);

			//���Ϳ�������
			ObjStreamPtr->StartGrab();
			ObjFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();

			//��ʱ���ɳɹ�,����̨��ӡ��Ϣ,ֱ���������������
			getchar();

			//����ͣ������
			ObjFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
			ObjStreamPtr->StopGrab();

			//ע���ɼ��ص�
			ObjStreamPtr->UnregisterCaptureCallback();

			//ע��Զ���豸�¼�
			//ObjFeatureControlPtr->UnregisterFeatureCallback(hFeatureEvent);

			//ע���豸�����¼�
			//ObjDevicePtr->UnregisterDeviceOfflineCallback(hDeviceOffline);

			//�ͷ���Դ
			ObjStreamPtr->Close();
			ObjDevicePtr->Close();
		}   while (0);
	}

	catch (CGalaxyException& e)
	{
		cout << "������: " << e.GetErrorCode() << endl;
		cout << "����������Ϣ: " << e.what() << endl;
	}
	catch (std::exception& e)
	{
		cout << "����������Ϣ: " << e.what() << endl;
	}

	//����ʼ����
	IGXFactory::GetInstance().Uninit();

	//�����¼��ص�ָ��
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
