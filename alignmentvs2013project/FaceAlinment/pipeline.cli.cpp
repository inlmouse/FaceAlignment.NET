#include "pipeline.cli.h"


using namespace cv;
using namespace std;
using namespace FastDetection;

#define TO_NATIVE_STRING(str) msclr::interop::marshal_as<std::string>(str)

namespace FacePipeline
{
	Pipeline::Pipeline(System::String^ modelFile)
	{
		Cascador = new lbf::LbfCascador();
		FILE *model = fopen(TO_NATIVE_STRING(modelFile).c_str(), "rb");
		Cascador->Read(model);
		fclose(model); 
		fastdetection = gcnew FastFace(1.2f, 3, 24);

		
	}

	Pipeline::~Pipeline()
	{
		delete Cascador;
	}

	FrameInfo Pipeline::GetFrameInfo(System::Drawing::Bitmap^ bmpImg)
	{
		FaceInfo info = fastdetection->Facedetect_Frontal(bmpImg);
		vector<Rect> rects;
		for (size_t i = 0; i < info.count; i++)
		{
			Rect temp = Rect(info.r[i].X, info.r[i].Y, info.r[i].Width, info.r[i].Height);
			rects.push_back(temp);
		}
		array<array<int, 2>^>^ landmarks = gcnew array<array<int, 2>^>(info.count);
		//****
		lbf::Config &config = lbf::Config::GetInstance();
		int landmark_n = config.landmark_n;
		double bbox[4]; // the bounding box
		vector<double> x(landmark_n), y(landmark_n); //used to strore the shape
		//****
		Mat img;
		int issuccess = ConvertBitmapToMat(bmpImg, img);

		if (issuccess==0)
		{
			for (size_t i = 0; i < info.count; i++)
			{

				Rect r = rects[i];
				Rect rec = enlarge(r, 1.2);
				// double check to guarentee that
				// the bounding box is contained by the img.
				rec.x = max(0, rec.x);
				rec.y = max(0, rec.y);
				rec.width = min(img.cols - rec.x, rec.width);
				rec.height = min(img.rows - rec.y, rec.height);

				Mat img_t = img(rec).clone();
				lbf::BBox bbox_(abs(rec.x - r.x), abs(rec.y - r.y), r.width, r.height);

				Mat gray;
				cvtColor(img_t, gray, CV_BGR2GRAY);
				vector<vector<double>> output = Cascador->Predict(gray, bbox_,true);
				landmarks[i] = gcnew array<int, 2>(68, 2);
				for (size_t j = 0; j < 68; j++)
				{
					landmarks[i][j, 0] = Convert::ToInt32(output[j][0]) + rec.x;
					landmarks[i][j, 1] = Convert::ToInt32(output[j][1]) + rec.y;
					//Console::WriteLine(landmarks[i][j, 1]);
				}
				gray.release();
				img_t.release();
			}
			img.release();
			return FrameInfo(info.count, info.r, landmarks);
		}
		else
		{
			img.release();
			return FrameInfo(-1);
		}
	}
	
	int Pipeline::ConvertBitmapToMat(System::Drawing::Bitmap^ bmpImg, cv::Mat& cvImg)
	{
		int retVal = 0;

		//锁定Bitmap数据  
		System::Drawing::Imaging::BitmapData^ bmpData = bmpImg->LockBits(
			System::Drawing::Rectangle(0, 0, bmpImg->Width, bmpImg->Height),
			System::Drawing::Imaging::ImageLockMode::ReadWrite, bmpImg->PixelFormat);

		//若cvImg非空，则清空原有数据  
		if (!cvImg.empty())
		{
			cvImg.release();
		}

		//将 bmpImg 的数据指针复制到 cvImg 中，不拷贝数据  
		if (bmpImg->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed)  // 灰度图像  
		{
			cvImg = cv::Mat(bmpImg->Height, bmpImg->Width, CV_8UC1, (char*)bmpData->Scan0.ToPointer());
		}
		else if (bmpImg->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)   // 彩色图像  
		{
			cvImg = cv::Mat(bmpImg->Height, bmpImg->Width, CV_8UC3, (char*)bmpData->Scan0.ToPointer());
		}
		else if (bmpImg->PixelFormat == System::Drawing::Imaging::PixelFormat::Format32bppArgb)	//RGBA
		{
			cvImg = cv::Mat(bmpImg->Height, bmpImg->Width, CV_8UC4, (char*)bmpData->Scan0.ToPointer());
		}
		else
		{
			retVal = -1;
		}

		//解锁Bitmap数据  
		bmpImg->UnlockBits(bmpData);

		return (retVal);
	}

	Rect Pipeline::enlarge(Rect r, double s)
	{
		Rect rec;
		assert(s > 1.0);
		rec.width = r.width * s;
		rec.height = r.height * s;
		rec.x = r.x - (rec.width - r.width) / 2;
		rec.y = r.y - (rec.height - r.height) / 2;

		return rec;
	}
}